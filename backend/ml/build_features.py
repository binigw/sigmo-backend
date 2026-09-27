"""Sigmo V2 — offline feature-corpus builder (v4 mechanical optimization).

Builds the three training corpora with the unified extractor from
``backend.ml.features``:

  ZZU   : streams ZZU-MCC5 3-phase motor current CSVs straight out of the
          8 GB 7-Zip archive (no raw data ever touches disk) via a py7zr
          ``WriterFactory``, cuts twenty 1.0 s, 12 800-sample windows per
          recording (uniformly spread across the recording's continuous
          segments, skipping the first/last 5 % of usable samples) and
          writes ``models/features_zzu.csv``.
  SEED  : SEED_PQD_v1 (Zenodo 11843312) XPQRS CSVs from
          ``datasets/seed_pqd.zip`` -> 17 000 single-cycle grid signals.
  IEEE  : HanShaoll/PQD-1D-model IEEE-1159 synthetic xlsx (0.2 s @ 3.2 kHz)
          from ``datasets/pqd_ieee_raw`` -> 80 000 grid signals.

Label assignments for the PQD sources are recovered from the v3 corpus
(``models/features_pqd_v3_legacy.csv``), so the new build is identical in
class balance and only changes feature definitions.

Usage:
  python3 -m backend.ml.build_features zzu [--start N] [--end M]
  python3 -m backend.ml.build_features seed
  python3 -m backend.ml.build_features ieee
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/sigmo_v2")

from backend.ml.features import (
    FEATURE_NAMES,
    META_COLUMNS,
    PQD_NOMINAL_RMS_PU,
    extract_signal_features,
    extract_window_features,
)

MODELS = Path("/home/user/sigmo_v2/models")
DATASETS = Path("/home/user/datasets")

ZZU_ARCHIVE = DATASETS / "zzu_dataset.7z"
ZZU_FILE_LIST = DATASETS / "zzu_used_files.txt"
ZZU_OUT = MODELS / "features_zzu.csv"
ZZU_LOG = MODELS / "zzu_build.log"

PQD_LEGACY = MODELS / "features_pqd_v3_legacy.csv"
SEED_ZIP = DATASETS / "seed_pqd.zip"
PQD_NEW_OUT = MODELS / "features_pqd_new.csv"
PQD_BUILD_LOG = MODELS / "pqd_build.log"

ZZU_FS = 12_800.0
ZZU_WINDOW = 32_000            # 2.5 s — v5: probes showed 3 k-rpm mechanical
                               # fingerprints are SNR-starved at 1.0 s (defect
                               # sidebands sit inside the 1 Hz skirt); 2.5 s
                               # (0.4 Hz resolution) restores them and matches
                               # the 16,384-pt Zoom-FFT scale of the DSP spec.
ZZU_WINDOWS_PER_FILE = 20
ZZU_SKIP_HEAD_TAIL = 0.05      # 5 % of usable samples each side
# One corrupt recording identified during the v3 audit (truncated payload).
ZZU_CORRUPT_EXCLUDE = {
    "bearing_inner_L_speed_circulation_20Nm_3000rpm_250707101400.csv",
}

ZZU_LABEL_RULES: tuple[tuple[str, str], ...] = (
    ("health", "HEALTHY"),
    ("bearing_inner", "BEARING_INNER_RACE"),
    ("bearing_outer", "BEARING_OUTER_RACE"),
    ("bearing_ball", "BEARING_BALL"),
    ("Rotor_misalignment", "SHAFT_MISALIGNMENT"),
    ("Rotor_unbalance", "MECH_UNBALANCE_ECCENTRICITY"),
    ("bend", "MECH_UNBALANCE_ECCENTRICITY"),
    ("broken_bar", "BROKEN_ROTOR_BAR"),
    ("winding", "STATOR_WINDING_INTERTURN"),
    ("voltage_unbalance", "PHASE_CURRENT_UNBALANCE"),
)

_FILENAME_RE = re.compile(
    r"^(?P<classev>.+?)_(?:speed|torque)_(?:circulation|ciculation)_"
    r"(?P<torque>\d+)Nm_(?P<rpm>\d+)rpm_\d+\.csv$"
)


def _log(line: str, path: Path) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp} {line}\n")


def _zzu_parse_name(basename: str) -> dict[str, object]:
    m = _FILENAME_RE.match(basename)
    if m is None:
        raise ValueError(f"unparseable ZZU filename: {basename}")
    classev = m.group("classev")
    sev_m = re.search(r"_(H|L)$", classev)
    severity = sev_m.group(1) if sev_m else "NA"
    base = classev[: -2] if sev_m else classev
    label = None
    for token, lab in ZZU_LABEL_RULES:
        if base.startswith(token):
            label = lab
            break
    if label is None:
        raise ValueError(f"no label rule for ZZU class token: {base}")
    return {
        "label": label,
        "severity": severity,
        "rpm": int(m.group("rpm")),
        "torque_nm": int(m.group("torque")),
    }


def _zzu_valid_starts(t: np.ndarray, dt_med: float) -> list[int]:
    """20 window starts spread over 5%..95% of the usable recording.

    ZZU-MCC5 timestamp audit finding: the time column repeats every
    16,384 samples (hardware DMA block is stamped with its START time, so
    time "runs backwards" exactly one block at every boundary).  The
    current channels themselves are one continuous stream (verified by
    spectral continuity across block edges), so windowing is driven by
    row indices; the time column is only used to estimate fs.
    """
    n = t.shape[0]
    if n < ZZU_WINDOW:
        return []
    last_start = n - ZZU_WINDOW
    head = int(ZZU_SKIP_HEAD_TAIL * last_start)
    tail = int((1.0 - ZZU_SKIP_HEAD_TAIL) * last_start)
    picks = np.linspace(head, tail, ZZU_WINDOWS_PER_FILE)
    return [int(round(p)) for p in picks]


def _zzu_rows_from_bytes(relpath: str, payload: bytes) -> list[dict[str, object]]:
    basename = relpath.rsplit("/", 1)[-1]
    meta = _zzu_parse_name(basename)
    df = pd.read_csv(
        io.BytesIO(payload),
        header=None,
        usecols=[0, 5, 6, 7],
        na_values=["NaN"],
        engine="c",
    )
    arr = df.to_numpy(dtype=np.float64, na_value=np.nan)
    good = np.all(np.isfinite(arr[:, 1:4]), axis=1)
    arr = arr[good]
    if arr.shape[0] < ZZU_WINDOW:
        raise ValueError(f"{basename}: only {arr.shape[0]} valid rows")
    t = arr[:, 0]
    cur = arr[:, 1:4]
    dt_med = float(np.median(np.diff(t)))
    if not (6e-5 < dt_med < 1e-3):
        raise ValueError(f"{basename}: implausible dt {dt_med:.3e}")
    fs = 1.0 / dt_med
    starts = _zzu_valid_starts(t, dt_med)
    rows: list[dict[str, object]] = []
    for s in starts:
        feats = extract_window_features(cur[s : s + ZZU_WINDOW], fs)
        tries = 0
        while np.isnan(feats["fundamental_hz"]) and tries < 30:
            s = min(s + ZZU_WINDOW // 10, arr.shape[0] - ZZU_WINDOW)
            feats = extract_window_features(cur[s : s + ZZU_WINDOW], fs)
            tries += 1
        if np.isnan(feats["fundamental_hz"]):
            continue
        rows.append(
            {
                "source": "ZZU_MCC5",
                "file": basename,
                "label": meta["label"],
                "severity": meta["severity"],
                "rpm": meta["rpm"],
                "torque_nm": meta["torque_nm"],
                "window_idx": int(s),
                **feats,
            }
        )
    return rows


class _CsvSink(io.BytesIO):
    """Receives one decompressed CSV during the streaming 7-Zip pass."""

    def __init__(self, relpath: str, builder: "_ZzuStreamBuilder") -> None:
        super().__init__()
        self.relpath = relpath
        self.builder = builder

    def close(self) -> None:
        try:
            payload = self.getvalue()
            self.builder.process(self.relpath, payload)
        finally:
            super().close()


class _ZzuStreamBuilder:
    def __init__(self, out_path: Path, log_path: Path, total_start: int) -> None:
        self.out_path = out_path
        self.log_path = log_path
        self.rows_total = total_start
        self.files_done = 0
        self._lock = threading.Lock()  # py7zr delivers files from worker threads

    def make_sink(self, relpath: str) -> _CsvSink:
        return _CsvSink(relpath, self)

    def process(self, relpath: str, payload: bytes) -> None:
        basename = relpath.rsplit("/", 1)[-1]
        try:
            rows = _zzu_rows_from_bytes(relpath, payload)
        except Exception as exc:  # noqa: BLE001 - corrupt payload must not kill the pass
            _log(f"ERROR {basename}: {exc}", self.log_path)
            return
        if not rows:
            _log(f"WARN {basename}: 0 valid windows (skipped)", self.log_path)
            return
        frame = pd.DataFrame(rows)[META_COLUMNS + FEATURE_NAMES]
        with self._lock:
            needs_header = not (
                self.out_path.exists() and self.out_path.stat().st_size > 0
            )
            frame.to_csv(self.out_path, mode="a", header=needs_header, index=False)
        self.rows_total += len(rows)
        self.files_done += 1
        _log(
            f"  [{self.files_done}] {basename}: {len(rows)} windows (total {self.rows_total})",
            self.log_path,
        )


class _StreamFactory:
    """py7zr WriterFactory: hands every decompressed target to the builder."""

    def __init__(self, builder: _ZzuStreamBuilder, wanted: set[str]) -> None:
        self.builder = builder
        self.wanted = wanted

    def create(self, filename: str) -> io.BytesIO:
        return self.builder.make_sink(filename)


def build_zzu(start: int, end: int) -> None:
    import py7zr

    wanted_names = [
        ln.strip() for ln in ZZU_FILE_LIST.read_text().splitlines() if ln.strip()
    ]
    wanted_names = [n for n in wanted_names if n not in ZZU_CORRUPT_EXCLUDE]
    slice_names = wanted_names[start:end]

    if ZZU_OUT.exists():
        done = set(pd.read_csv(ZZU_OUT, usecols=["file"])["file"].unique())
    else:
        done = set()
    todo = [n for n in slice_names if n not in done]
    total_start = len(pd.read_csv(ZZU_OUT, usecols=["file"])) if ZZU_OUT.exists() else 0
    _log(
        f"ZZU build slice [{start}:{end}] -> {len(todo)} files to process "
        f"({len(done)} already in corpus)",
        ZZU_LOG,
    )
    if not todo:
        print("ZZU slice already complete.")
        return

    builder = _ZzuStreamBuilder(ZZU_OUT, ZZU_LOG, total_start)
    with py7zr.SevenZipFile(ZZU_ARCHIVE, "r") as zf:
        base_to_rel = {n.rsplit("/", 1)[-1]: n for n in zf.getnames() if n.endswith(".csv")}
        rel_targets = [
            base_to_rel[n] for n in todo if n in base_to_rel and base_to_rel[n]
        ]
        missing = [n for n in todo if n not in base_to_rel]
        for n in missing:
            _log(f"MISSING in archive: {n}", ZZU_LOG)
        t0 = time.time()
        zf.extract(targets=rel_targets, factory=_StreamFactory(builder, set(rel_targets)))
    _log(
        f"ZZU slice done: {builder.files_done} files, {builder.rows_total} rows total, "
        f"{time.time() - t0:.0f}s",
        ZZU_LOG,
    )
    print(f"ZZU slice done: files={builder.files_done} rows_total={builder.rows_total}")


def _pqd_label_map() -> dict[str, tuple[str, str]]:
    """file basename -> (source, label) recovered from the v3 corpus."""
    legacy = pd.read_csv(PQD_LEGACY, usecols=["source", "file", "label"])
    legacy = legacy.drop_duplicates("file")
    out: dict[str, tuple[str, str]] = {}
    for _, r in legacy.iterrows():
        base = str(r["file"]).rsplit("/", 1)[-1]
        out[base] = (str(r["source"]), str(r["label"]))
    return out


def _append_pqd(rows: list[dict[str, object]], log_note: str) -> None:
    if not rows:
        return
    header = not PQD_NEW_OUT.exists()
    pd.DataFrame(rows)[META_COLUMNS + FEATURE_NAMES].to_csv(
        PQD_NEW_OUT, mode="a", header=header, index=False
    )
    _log(f"{log_note}: {len(rows)} rows -> {PQD_NEW_OUT.name}", PQD_BUILD_LOG)


def build_seed() -> None:
    labels = _pqd_label_map()
    t0 = time.time()
    done = 0
    with zipfile.ZipFile(SEED_ZIP) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".csv"):
                continue
            base = name.rsplit("/", 1)[-1]
            tag = base.replace(".csv", "")
            hit = labels.get(tag) or labels.get(base)
            if hit is None:
                _log(f"SEED: no v3 label for {base} (skipped)", PQD_BUILD_LOG)
                continue
            source, label = hit
            raw = zf.read(name)
            mat = pd.read_csv(io.BytesIO(raw), header=None).to_numpy(np.float64)
            if mat.shape[0] < mat.shape[1]:
                mat = mat.T  # signals must be ROWS (n_signals, n_samples)
            fs = 5000.0
            rows = []
            for i in range(mat.shape[0]):
                x = mat[i]
                if not np.all(np.isfinite(x)):
                    continue
                feats = extract_signal_features(x, fs, nominal_rms=PQD_NOMINAL_RMS_PU)
                if np.isnan(feats["fundamental_hz"]):
                    continue
                rows.append(
                    {
                        "source": "SEED_PQD",
                        "file": f"XPQRS/{base}",
                        "label": label,
                        "severity": "NA",
                        "rpm": np.nan,
                        "torque_nm": np.nan,
                        "window_idx": int(i),
                        **feats,
                    }
                )
            _append_pqd(rows, f"SEED {base} [{label}]")
            done += 1
    _log(f"SEED done: {done} files in {time.time() - t0:.0f}s", PQD_BUILD_LOG)
    print(f"SEED done: {done} files")


def build_ieee() -> None:
    import openpyxl

    raw_dir = DATASETS / "pqd_ieee_raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"{raw_dir} missing — place the IEEE_PQD xlsx files there first"
        )
    labels = _pqd_label_map()
    xlsx_files = sorted(raw_dir.glob("**/*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"no xlsx files under {raw_dir}")
    t0 = time.time()
    for x_idx, path in enumerate(xlsx_files):
        hit = labels.get(path.stem) or labels.get(path.name)
        if hit is None:
            _log(f"IEEE: no v3 label for {path.name} (skipped)", PQD_BUILD_LOG)
            continue
        source, label = hit
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = []
        for i, row_vals in enumerate(ws.iter_rows(values_only=True)):
            try:
                x = np.asarray(row_vals, dtype=np.float64)
            except (TypeError, ValueError):
                continue  # tolerate a header row if present
            if x.shape[0] != 640 or not np.all(np.isfinite(x)):
                continue
            feats = extract_signal_features(x, 3200.0, nominal_rms=PQD_NOMINAL_RMS_PU)
            if np.isnan(feats["fundamental_hz"]):
                continue
            rows.append(
                {
                    "source": "IEEE_PQD",
                    "file": path.name,
                    "label": label,
                    "severity": "NA",
                    "rpm": np.nan,
                    "torque_nm": np.nan,
                    "window_idx": int(i),
                    **feats,
                }
            )
        wb.close()
        _append_pqd(rows, f"IEEE [{x_idx + 1}/{len(xlsx_files)}] {path.name} [{label}]")
    _log(f"IEEE done: {len(xlsx_files)} files in {time.time() - t0:.0f}s", PQD_BUILD_LOG)
    print(f"IEEE done: {len(xlsx_files)} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sigmo V2 feature-corpus builder")
    parser.add_argument("source", choices=["zzu", "seed", "ieee"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10**9)
    args = parser.parse_args()
    if args.source == "zzu":
        build_zzu(args.start, args.end)
    elif args.source == "seed":
        build_seed()
    else:
        build_ieee()


if __name__ == "__main__":
    main()
