"""Sigmo V2 — model training & versioning pipeline (v6).

Stage-1 (v5 design, locked, all experiments documented in
models/OPTIMIZATION_REPORT.md):

  * SINGLE 13-class XGBoost over the feature vector (a two-specialist
    hierarchy was tried in v4b and degraded the bearing classes; fixed
    700-round budget — early stopping on a ~700-row internal fold
    collapsed learning in v4b);
  * tempered inverse-frequency class weights (exponent 0.6) + targeted
    class boosts for the mechanical classes + severity boosts for the
    physically weaker light-severity ("L") recordings;
  * corpus: ZZU-MCC5 3-phase windows rebuilt at 2.5 s (0.4 Hz envelope
    resolution — restoring 3 k-rpm defect sidebands starved at 1 s),
    SEED_PQD and IEEE_PQD grid signals (IEEE rows retained under the
    documented definition-drift disclosure);
  * evaluation: StratifiedGroupKFold(5, seed 42) with ZZU grouped BY
    RECORDING.  Fold 0 = production artifact + gate metrics.  The
    remaining 4 folds are also trained+scored (cross-validation summary)
    so reporting carries per-class stability margins instead of
    single-fold luck; those boosters are discarded.

Stage-2 (v6, V6 ADDENDUM A2) — bearing specialist with HONEST
enable-if-better gating:

  * a 3-class specialist (OUTER/INNER/BALL) trained ONLY on bearing rows
    of the train split, over the 117 base features + 8 NaN-safe
    rpm×signature interaction columns (125 inputs);
  * composed system = the stage-1 verdict stands UNLESS it is a bearing
    class, in which case the specialist re-decides;
  * the specialist is persisted (model_bearing.json) but wired into
    production inference ONLY when the composed end-to-end macro-F1 on
    the held-out fold EXCEEDS the single-model macro-F1.  Ties disable.
    No hierarchy ships on vibes; the flag lives in features.json.

The Isolation-Forest novelty gate (Cold Start signal 1 of 4) and the
population baseline are rebuilt every run.  No synthetic fault data is
ever generated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

import sys

sys.path.insert(0, "/home/user/sigmo_v2")

from backend.ml.features import CLASS_NAMES, FEATURE_NAMES, ISO_MOTOR_FEATURES, META_COLUMNS

MODELS = Path("/home/user/sigmo_v2/models")
REGISTRY = MODELS / "registry.json"

# v6c training corpus (Phase-2 assembly, 2026-09-24; 13-class restore
# 2026-09-25): the single concatenated matrix produced by
# scripts/build_corpus_external.py `assemble` (MCC5 speed + MCC5 torque +
# KAIST bearing + KAIST varying-load + SEED_PQD + IEEE_PQD, all on the
# current 124-column v6 contract).  Corpus CSVs rest as .csv.gz
# (snapshot-cap safety — see builder).  sha256 and per-part audit gates
# are recorded in models/corpus_build_audit.jsonl (action
# "v6c-assembled").  The legacy part files remain on disk for provenance
# but are no longer training inputs.
V6C_FEATURES = MODELS / "features_v6c.csv"


def _read_corpus_csv(path: Path, **kw) -> pd.DataFrame:
    """Read a corpus CSV that rests as .csv.gz (plain fallback)."""
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return pd.read_csv(gz, **kw)
    return pd.read_csv(path, **kw)

# ---------------------------------------------------------------------------
# Training configuration (v5, locked)
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
WEIGHT_TEMPER = 0.6
CLASS_BOOST = {
    "BEARING_OUTER_RACE": 1.5,
    "BEARING_INNER_RACE": 1.5,
    "MECH_UNBALANCE_ECCENTRICITY": 1.25,
    "BROKEN_ROTOR_BAR": 1.1,
    "STATOR_WINDING_INTERTURN": 1.4,
}
SEVERITY_BOOST = {
    # NOTE: "SHAFT_MISALIGNMENT|L" is inert for v6c — the restored class
    # (KAIST varying-load) uses numeric severity codes (1/3/5), not H/L.
    "SHAFT_MISALIGNMENT|L": 1.5,
    "STATOR_WINDING_INTERTURN|L": 1.4,
    "BEARING_OUTER_RACE|L": 1.3,
    "BEARING_INNER_RACE|L": 1.5,
    "BEARING_BALL|L": 1.2,
}
XGB_PARAMS: dict[str, object] = {
    "objective": "multi:softprob",
    "num_class": len(CLASS_NAMES),
    "max_depth": 9,
    "eta": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "colsample_bynode": 0.6,
    "min_child_weight": 2,
    "gamma": 0.05,
    "lambda": 1.5,
    "tree_method": "hist",
    "seed": SEED,
    "nthread": -1,
}
XGB_ROUNDS = 700      # fixed budget; see module docstring
ISO_CONTAMINATION = 0.02
ISO_N_ESTIMATORS = 300
SPEED_BUCKET_RPM = 1000.0

# ---------------------------------------------------------------------------
# v6 Stage-2 bearing specialist (V6 ADDENDUM A2) — honest enable-if-better
# ---------------------------------------------------------------------------
BEARING_CLASSES: tuple[str, ...] = (
    "BEARING_OUTER_RACE",
    "BEARING_INNER_RACE",
    "BEARING_BALL",
)
# 8 NaN-safe rpm×signature interaction columns appended for the
# specialist (rpm_est × base): bearing envelopes scale with shaft speed,
# so the interactions expose speed-conditional signature strength.
RPM_INTERACT_BASE: tuple[str, ...] = (
    "eband_1p0x", "eband_2p0x", "eband_3p0x", "eband_6p0x",
    "hfenv_opk1_sb1x_eng", "hfenv_1x_rel", "eratio_2p0x", "esnr_6p0x",
)
BEARING_WEIGHT_TEMPER = 0.3
BEARING_CLASS_BOOST: dict[str, float] = {
    "BEARING_OUTER_RACE": 1.25,
    "BEARING_INNER_RACE": 1.25,
}
# Same regularization family as stage-1, shallower trees (depth 6) and a
# fixed 400-round budget (A2 spec).
BEARING_XGB_PARAMS: dict[str, object] = {
    **{k: v for k, v in XGB_PARAMS.items() if k not in ("num_class", "max_depth")},
    "max_depth": 6,
    "num_class": len(BEARING_CLASSES),
}
BEARING_XGB_ROUNDS = 400


# ---------------------------------------------------------------------------
# Corpus assembly (row order is part of the evaluation contract)
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cols = META_COLUMNS + FEATURE_NAMES

    if not (V6C_FEATURES.exists()
            or Path(str(V6C_FEATURES) + ".gz").exists()):
        raise FileNotFoundError(
            f"v6c corpus missing: {V6C_FEATURES} — build it with "
            f"`python3 scripts/build_corpus_external.py assemble`"
        )
    # reindex (NOT strict selection) as a defensive no-op: the v6c corpus
    # is already on the current 117-feature contract, and XGBoost handles
    # NaN natively (single-channel PQD rows carry NaN motor-only columns).
    df = _read_corpus_csv(V6C_FEATURES, low_memory=False)
    return df.reindex(columns=cols)


def load_features_v5_lineage() -> pd.DataFrame:
    """The v5-lineage corpus (ZZU_MCC5 + SEED_PQD + v3-legacy IEEE rows).

    Kept verbatim so any still-active v5-era model can be re-evaluated on
    EXACTLY the corpus it was trained from (regression test M4); the v6c
    corpus is a different recording set and would misrepresent it.
    """
    frames: list[pd.DataFrame] = []
    cols = META_COLUMNS + FEATURE_NAMES
    zzu = _read_corpus_csv(MODELS / "features_zzu.csv", low_memory=False)
    frames.append(zzu.reindex(columns=cols))

    legacy = _read_corpus_csv(MODELS / "features_pqd_v3_legacy.csv",
                              low_memory=False)
    legacy = legacy[legacy["source"] == "IEEE_PQD"]
    frames.append(legacy.reindex(columns=cols))  # v5-only columns -> NaN

    pqd_new = MODELS / "features_pqd_new.csv"
    if pqd_new.exists() or Path(str(pqd_new) + ".gz").exists():
        # reindex: motor-only columns are NaN for single-channel grid rows
        frames.append(_read_corpus_csv(pqd_new, low_memory=False)
                      .reindex(columns=cols))
    return pd.concat(frames, ignore_index=True)


def load_features_for_version(metrics: dict) -> pd.DataFrame:
    """Select the corpus a model version was actually trained from, using
    the ``corpus.sources`` block of its saved metrics."""
    sources = set(metrics.get("corpus", {}).get("sources", []))
    if sources & {"MCC5_THU", "KAIST_BEARING", "KAIST_LOAD"}:
        return load_features()          # v6c lineage
    return load_features_v5_lineage()   # v5 lineage (ZZU_MCC5 era)


MCSA_SOURCES = frozenset(
    {"ZZU_MCC5", "MCC5_THU", "KAIST_BEARING", "KAIST_LOAD"}
)


def _group_keys(df: pd.DataFrame) -> pd.Series:
    keys = df["source"].astype(str) + "::" + df["file"].astype(str)
    # MCSA corpora are windowed recordings: windows from one file are
    # strongly correlated, so the whole FILE must land on one side of the
    # split.  PQD rows are independent single-cycle synthetic signals, so
    # each row is its own group (v5 lineage semantics).
    mcsa_mask = df["source"].isin(MCSA_SOURCES)
    pqd_mask = ~mcsa_mask
    if pqd_mask.any():
        keys = keys.copy()
        keys[pqd_mask] = (
            keys[pqd_mask] + "::row" + df.index.to_series()[pqd_mask].astype(str)
        )
    return keys


def split_grouped(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic grouped 80/20 split (StratifiedGroupKFold fold 0)."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    labels = df["label"].to_numpy()
    groups = _group_keys(df).to_numpy()
    train_idx, test_idx = next(iter(sgkf.split(np.zeros(len(df)), labels, groups)))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
def _weights_from_counts(y: np.ndarray, temper: float, boost: dict[str, float] | None,
                         class_names: list[str]) -> np.ndarray:
    counts = np.bincount(y, minlength=len(class_names)).astype(np.float64)
    counts[counts == 0.0] = 1.0
    inv = counts ** (-temper)
    inv /= float(np.median(inv))
    if boost:
        for idx, name in enumerate(class_names):
            inv[idx] *= boost.get(name, 1.0)
    return inv[y]


def _sample_weights(frame: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    w = _weights_from_counts(y, WEIGHT_TEMPER, CLASS_BOOST, CLASS_NAMES)
    sev = frame["severity"].astype(str).to_numpy()
    lab = frame["label"].astype(str).to_numpy()
    mult = np.ones(len(frame))
    for key, m in SEVERITY_BOOST.items():
        l_part, _, s_part = key.partition("|")
        mult[(lab == l_part) & (sev == s_part)] *= m
    return w * mult


def _fit_booster(train_df: pd.DataFrame) -> xgb.Booster:
    label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
    y = train_df["label"].map(label_map).to_numpy()
    x = train_df[FEATURE_NAMES].to_numpy(np.float64)
    dtrain = xgb.DMatrix(x, label=y, feature_names=FEATURE_NAMES, nthread=-1)
    dtrain.set_weight(_sample_weights(train_df, y))
    return xgb.train(XGB_PARAMS, dtrain, num_boost_round=XGB_ROUNDS, verbose_eval=False)


# ---------------------------------------------------------------------------
# v6 Stage-2 bearing specialist (V6 ADDENDUM A2)
# ---------------------------------------------------------------------------
def _bearing_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """117 base features + 8 NaN-safe rpm×signature interactions (125 cols).

    NaN propagates naturally through the products (missing rpm_est or a
    NaN base feature yields a NaN interaction) — XGBoost handles NaN
    natively and the trainer never imputes.
    """
    rpm = frame["rpm_est"].to_numpy(np.float64)
    cols: list[str] = list(FEATURE_NAMES)
    data = [frame[f].to_numpy(np.float64) for f in FEATURE_NAMES]
    for base in RPM_INTERACT_BASE:
        cols.append(f"rpm_x_{base}")
        data.append(rpm * frame[base].to_numpy(np.float64))
    return np.column_stack(data), cols


def _fit_bearing_specialist(train_df: pd.DataFrame) -> xgb.Booster:
    """Train the 3-class specialist on bearing rows of the TRAIN split only."""
    sub = train_df[train_df["label"].isin(BEARING_CLASSES)]
    label_map = {c: i for i, c in enumerate(BEARING_CLASSES)}
    y = sub["label"].map(label_map).to_numpy()
    x, cols = _bearing_matrix(sub)
    w = _weights_from_counts(
        y, BEARING_WEIGHT_TEMPER, BEARING_CLASS_BOOST, list(BEARING_CLASSES)
    )
    dtrain = xgb.DMatrix(x, label=y, feature_names=cols, nthread=-1)
    dtrain.set_weight(w)
    return xgb.train(
        BEARING_XGB_PARAMS, dtrain, num_boost_round=BEARING_XGB_ROUNDS,
        verbose_eval=False,
    )



def _bearing_submodel_metrics(
    booster: xgb.Booster,
    booster_b: xgb.Booster,
    test_df: pd.DataFrame,
    stage1_macro_f1: float,
) -> dict[str, object]:
    """Honest enable-if-better evaluation of the Stage-2 reroute.

    Composed system: the stage-1 verdict stands UNLESS it is a bearing
    class, in which case the specialist (3-class, bearing-only) re-decides
    among OUTER/INNER/BALL.  Enablement requires the composed end-to-end
    macro-F1 (full held-out fold, all 13 classes) to EXCEED the
    single-model macro-F1; ties disable.  The decision and every number
    behind it are persisted in metrics.json.
    """
    label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_test = test_df["label"].map(label_map).to_numpy()
    x_test = test_df[FEATURE_NAMES].to_numpy(np.float64)
    proba = booster.predict(
        xgb.DMatrix(x_test, feature_names=FEATURE_NAMES, nthread=-1)
    )
    pred1 = np.asarray(np.argmax(proba, axis=1), dtype=np.int64)

    bearing_global_idx = [label_map[c] for c in BEARING_CLASSES]
    reroute_mask = np.isin(pred1, bearing_global_idx)

    xb, bcols = _bearing_matrix(test_df)
    predb = np.asarray(
        np.argmax(
            booster_b.predict(xgb.DMatrix(xb, feature_names=bcols, nthread=-1)),
            axis=1,
        ),
        dtype=np.int64,
    )
    b_local_to_global = {i: label_map[c] for i, c in enumerate(BEARING_CLASSES)}

    composed = pred1.copy()
    composed[reroute_mask] = [b_local_to_global[int(i)] for i in predb[reroute_mask]]

    composed_macro = float(f1_score(y_test, composed, average="macro", zero_division=0))
    composed_acc = float(accuracy_score(y_test, composed))

    all_idx = np.arange(len(CLASS_NAMES))
    prec = precision_score(y_test, composed, labels=all_idx, average=None, zero_division=0)
    rec = recall_score(y_test, composed, labels=all_idx, average=None, zero_division=0)
    f1s = f1_score(y_test, composed, labels=all_idx, average=None, zero_division=0)
    composed_per_bearing = {
        c: {
            "precision": float(prec[label_map[c]]),
            "recall": float(rec[label_map[c]]),
            "f1": float(f1s[label_map[c]]),
        }
        for c in BEARING_CLASSES
    }

    # Specialist-only accuracy on TRUE-bearing held-out windows (what the
    # 3-class model resolves when the routing decision is perfect).
    # predb is in the specialist's LOCAL label space — map to global
    # class indices before scoring (mixed label spaces would compare
    # apples to oranges).
    predb_global = np.asarray(
        [b_local_to_global[int(i)] for i in predb], dtype=np.int64
    )
    true_bearing = np.isin(y_test, bearing_global_idx)
    stage2_only = (
        float(accuracy_score(y_test[true_bearing], predb_global[true_bearing]))
        if bool(true_bearing.any())
        else float("nan")
    )

    enabled = bool(composed_macro > stage1_macro_f1)
    return {
        "enabled": enabled,
        "decision_rule": (
            "composed_end_to_end.macro_f1 > stage1.macro_f1 (strict; ties disable)"
        ),
        "stage1_macro_f1": float(stage1_macro_f1),
        "composed_end_to_end": {
            "accuracy": composed_acc,
            "macro_f1": composed_macro,
            "per_bearing": composed_per_bearing,
        },
        "stage2_only_accuracy_true_bearing_windows": stage2_only,
        "params": dict(BEARING_XGB_PARAMS),
        "rounds": BEARING_XGB_ROUNDS,
        "weight_temper": BEARING_WEIGHT_TEMPER,
        "class_boost": dict(BEARING_CLASS_BOOST),
        "rpm_interact_base": list(RPM_INTERACT_BASE),
        "input_columns": len(FEATURE_NAMES) + len(RPM_INTERACT_BASE),
    }


def _evaluate(booster: xgb.Booster, test_df: pd.DataFrame) -> dict[str, object]:
    label_map = {c: i for i, c in enumerate(CLASS_NAMES)}
    y_test = test_df["label"].map(label_map).to_numpy()
    x_test = test_df[FEATURE_NAMES].to_numpy(np.float64)
    proba = booster.predict(xgb.DMatrix(x_test, feature_names=FEATURE_NAMES, nthread=-1))
    pred = np.asarray(np.argmax(proba, axis=1), dtype=np.int64)
    all_idx = np.arange(len(CLASS_NAMES))
    prec = precision_score(y_test, pred, labels=all_idx, average=None, zero_division=0)
    rec = recall_score(y_test, pred, labels=all_idx, average=None, zero_division=0)
    f1s = f1_score(y_test, pred, labels=all_idx, average=None, zero_division=0)
    support = np.bincount(y_test, minlength=len(CLASS_NAMES))
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1s[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_test, pred, labels=all_idx).tolist(),
        "confusion_labels": CLASS_NAMES,
    }


def cross_validation_summary(df: pd.DataFrame) -> dict[str, object]:
    """Train+score all 5 grouped folds; report per-class F1 stability."""
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    labels = df["label"].to_numpy()
    groups = _group_keys(df).to_numpy()
    fold_rows: list[dict[str, object]] = []
    f1_by_class: dict[str, list[float]] = {c: [] for c in CLASS_NAMES}
    for k, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(df)), labels, groups)):
        booster = _fit_booster(df.iloc[tr_idx])
        m = _evaluate(booster, df.iloc[te_idx])
        fold_rows.append({"fold": k, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        for c in CLASS_NAMES:
            f1_by_class[c].append(m["per_class"][c]["f1"])
        del booster
    per_class = {
        c: {
            "f1_mean": float(np.mean(v)),
            "f1_min": float(np.min(v)),
            "f1_max": float(np.max(v)),
        }
        for c, v in f1_by_class.items()
    }
    return {
        "folds": fold_rows,
        "accuracy_mean": float(np.mean([f["accuracy"] for f in fold_rows])),
        "macro_f1_mean": float(np.mean([f["macro_f1"] for f in fold_rows])),
        "per_class_f1": per_class,
    }


# ---------------------------------------------------------------------------
# Isolation-Forest novelty gate (Cold Start signal 1 of 4)
# ---------------------------------------------------------------------------
def _speed_bucket(rpm: pd.Series) -> pd.Series:
    return (rpm / SPEED_BUCKET_RPM).round() * SPEED_BUCKET_RPM


def _train_gate(train_df: pd.DataFrame) -> dict[str, object]:
    healthy = train_df[
        # v6c: healthy MOTOR windows from every MCSA source (v5 hardcode
        # was ZZU_MCC5-only, which is empty in the v6c corpus)
        (train_df["label"] == "HEALTHY")
        & (train_df["source"].isin(MCSA_SOURCES))
    ]
    feats = list(ISO_MOTOR_FEATURES)
    buckets: dict[str, dict[str, list[float]]] = {}
    for bkt, grp in healthy.groupby(_speed_bucket(healthy["rpm_est"])):
        mu = grp[feats].mean(numeric_only=True)
        sd = grp[feats].std(numeric_only=True).replace(0.0, np.nan)
        buckets[str(int(bkt))] = {
            "mean": [float(v) for v in mu.fillna(0.0).to_numpy()],
            "std": [float(v) if np.isfinite(v) and v > 0 else 1.0 for v in sd.to_numpy()],
        }

    def _transform(frame: pd.DataFrame) -> np.ndarray:
        x = frame[feats].to_numpy(np.float64).copy()
        bkt = _speed_bucket(frame["rpm_est"]).to_numpy()
        available = sorted(int(k) for k in buckets)
        for i in range(len(frame)):
            b = int(bkt[i])
            stats_b = buckets.get(str(b))
            if stats_b is None:
                nearest = min(available, key=lambda k: abs(k - b))
                stats_b = buckets[str(nearest)]
            mu = np.asarray(stats_b["mean"])
            sd = np.asarray(stats_b["std"])
            x[i] = (x[i] - mu) / sd
        return np.clip(np.nan_to_num(x, nan=0.0), -10.0, 10.0)

    forest = IsolationForest(
        n_estimators=ISO_N_ESTIMATORS,
        contamination=ISO_CONTAMINATION,
        random_state=SEED,
        n_jobs=-1,
    )
    forest.fit(_transform(healthy))
    return {"forest": forest, "features": feats, "buckets": buckets, "transform": _transform}


def _gate_metrics(gate: dict[str, object], train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, object]:
    forest: IsolationForest = gate["forest"]
    tf = gate["transform"]
    healthy_train = train_df[
        (train_df["label"] == "HEALTHY") & (train_df["source"].isin(MCSA_SOURCES))]
    healthy_test = test_df[
        (test_df["label"] == "HEALTHY") & (test_df["source"].isin(MCSA_SOURCES))]
    fault_train = train_df[
        (train_df["label"] != "HEALTHY") & (train_df["source"].isin(MCSA_SOURCES))]
    return {
        "healthy_pass_rate": float((forest.predict(tf(healthy_train)) == 1).mean()),
        "healthy_pass_rate_heldout": float((forest.predict(tf(healthy_test)) == 1).mean()) if len(healthy_test) else float("nan"),
        "fault_trip_rate": float((forest.predict(tf(fault_train)) == -1).mean()),
        "role": "novelty gate only — signal 1 of 4 in the Cold Start 11.6 confirmation rule; never an alert source by itself",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def _write_registry(entry: dict[str, object]) -> None:
    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"versions": []}
    pruned = []
    for v in reg["versions"]:
        vdir = MODELS / str(v["version"])
        if (vdir / "metrics.json").exists():
            v["active"] = False
            pruned.append(v)
    entry["active"] = True
    pruned.append(entry)
    REGISTRY.write_text(json.dumps({"versions": pruned}, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def train(with_cv_summary: bool = True) -> dict[str, object]:
    t0 = datetime.now(timezone.utc)
    df = load_features()
    train_df, test_df = split_grouped(df)

    booster = _fit_booster(train_df)
    xgb_metrics = _evaluate(booster, test_df)

    cv_summary = cross_validation_summary(df) if with_cv_summary else None

    gate = _train_gate(train_df)
    iso_metrics = _gate_metrics(gate, train_df, test_df)

    # v6 Stage-2 bearing specialist: always trained and persisted, wired
    # into production ONLY if the composed system measurably wins (A2).
    booster_b = _fit_bearing_specialist(train_df)
    bearing_metrics = _bearing_submodel_metrics(
        booster, booster_b, test_df, float(xgb_metrics["macro_f1"])
    )

    baseline_rows: list[dict[str, object]] = []
    healthy_all = train_df[train_df["label"] == "HEALTHY"]
    for source, grp in healthy_all.groupby("source"):
        for feat in FEATURE_NAMES:
            vals = grp[feat].dropna().to_numpy(np.float64)
            if vals.size < 20:
                continue
            baseline_rows.append(
                {
                    "source_dataset": str(source),
                    "feature": feat,
                    "healthy_mean": float(np.mean(vals)),
                    "healthy_std": float(np.std(vals)),
                    "p05": float(np.percentile(vals, 5)),
                    "p95": float(np.percentile(vals, 95)),
                    "n_samples": int(vals.size),
                }
            )

    version = "v6." + t0.strftime("%Y%m%d.%H%M%S")
    vdir = MODELS / version
    vdir.mkdir(parents=True, exist_ok=False)
    booster.save_model(vdir / "model.json")
    booster_b.save_model(vdir / "model_bearing.json")
    joblib.dump(
        {"forest": gate["forest"], "features": gate["features"], "buckets": gate["buckets"]},
        vdir / "gate.joblib",
    )
    (vdir / "features.json").write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "class_names": CLASS_NAMES,
                "iso_motor_features": list(ISO_MOTOR_FEATURES),
                "bearing_classes": list(BEARING_CLASSES),
                "rpm_interact_base": list(RPM_INTERACT_BASE),
                "bearing_submodel_enabled": bool(bearing_metrics["enabled"]),
            },
            indent=2,
        )
    )
    metrics = {
        "version": version,
        "trained_at": t0.isoformat(),
        "architecture": (
            "stage-1 single 13-class XGBoost (117 features) + Stage-2 "
            "bearing specialist (honest enable-if-better, see bearing_submodel)"
        ),
        "corpus": {
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "zzu_rows": int((df["source"] == "ZZU_MCC5").sum()),
            "sources": sorted(df["source"].unique().tolist()),
        },
        "xgboost": {
            **xgb_metrics,
            "rounds": XGB_ROUNDS,
            "params": dict(XGB_PARAMS),
            "class_boost": CLASS_BOOST,
            "severity_boost": SEVERITY_BOOST,
            "weight_temper": WEIGHT_TEMPER,
        },
        "cross_validation_5fold": cv_summary,
        "isolation_forest": iso_metrics,
        "bearing_submodel": bearing_metrics,
    }
    (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (vdir / "population_baseline.json").write_text(json.dumps(baseline_rows, indent=2))
    (vdir / "train_config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_splits": N_SPLITS,
                "xgb_rounds": XGB_ROUNDS,
                "iso_contamination": ISO_CONTAMINATION,
                "iso_n_estimators": ISO_N_ESTIMATORS,
            },
            indent=2,
        )
    )
    _write_registry(
        {
            "version": version,
            "trained_at": t0.isoformat(),
            "accuracy": xgb_metrics["accuracy"],
            "macro_f1": xgb_metrics["macro_f1"],
            "active": True,
        }
    )

    print(f"version {version}")
    print(f"corpus {len(df)} rows | train {len(train_df)} / test {len(test_df)}")
    print(f"accuracy {xgb_metrics['accuracy']:.4f}  macro-F1 {xgb_metrics['macro_f1']:.4f}")
    for name in CLASS_NAMES:
        m = xgb_metrics["per_class"][name]
        print(f"  {name:32s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f} n={m['support']}")
    if cv_summary is not None:
        print(f"5-fold CV: acc {cv_summary['accuracy_mean']:.4f} macro-F1 {cv_summary['macro_f1_mean']:.4f}")
    print(f"  ISO healthy pass {iso_metrics['healthy_pass_rate']:.3f} "
          f"heldout {iso_metrics['healthy_pass_rate_heldout']:.3f} "
          f"fault trip {iso_metrics['fault_trip_rate']:.3f}")
    bm = bearing_metrics["composed_end_to_end"]
    print(f"bearing specialist: composed macro-F1 {bm['macro_f1']:.4f} "
          f"vs stage-1 {bearing_metrics['stage1_macro_f1']:.4f} "
          f"-> enabled={bearing_metrics['enabled']} "
          f"(stage2-only on true-bearing windows "
          f"{bearing_metrics['stage2_only_accuracy_true_bearing_windows']:.3f})")
    for c in BEARING_CLASSES:
        m = bm["per_bearing"][c]
        print(f"  composed {c:26s} P {m['precision']:.2f} R {m['recall']:.2f} F1 {m['f1']:.2f}")
    return metrics


if __name__ == "__main__":
    train()
