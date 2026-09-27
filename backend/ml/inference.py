"""Sigmo V2 — model runtime for inference inside the FastAPI backend.

``ModelRuntime`` loads the ACTIVE registry version's artifact bundle and
serves:

  * ``classify(feature_dict)`` — 13-class XGBoost verdict with full
    probability vector; guaranteed to reproduce the training evaluation
    on the same feature rows (test M4).  When the loaded version ships
    with ``bearing_submodel_enabled=true`` (V6 ADDENDUM A2), bearing
    verdicts are re-routed through the persisted 3-class specialist
    (model_bearing.json) and the payload carries a ``bearing_submodel``
    block with the stage-1 verdict preserved for audit;
  * ``classify_temporal(stream_id, feature_dict)`` — per-window verdict
    plus the §11.6 temporal aggregation (TemporalVoteBuffer: rolling
    majority over the last 5 window verdicts per stream, LRU-capped).
    The buffer NEVER raises alerts by itself (Cold Start 11.6: 4-signal
    confirmation lives in the gating layer, not here);
  * ``anomaly_gate(feature_dict)`` — Isolation-Forest novelty flag in
    the same per-speed-bucket z-space the gate was trained in
    (Cold Start signal 1 of 4; never an alert source by itself),
  * ``rollback(version)`` — registry surgery + artifact pre-check +
    reload, keeping exactly one active version.

All state lives in the repo's ``models/`` directory; no model state is ever
cached outside the versioned artifact bundles.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

# Repository root = three levels up from this file (backend/ml/), so the
# bundle resolves identically in the dev sandbox, a Docker image
# (WORKDIR /app), and any clone of the project.
MODELS = Path(__file__).resolve().parents[2] / "models"
SPEED_BUCKET_RPM = 1000.0
ARTIFACTS = ("model.json", "features.json", "gate.joblib", "metrics.json")
# Optional artifact: required ONLY when features.json sets
# bearing_submodel_enabled=true (V6 ADDENDUM A2).
BEARING_ARTIFACT = "model_bearing.json"


class TemporalVoteBuffer:
    """§11.6 temporal aggregation: rolling-majority verdict voting.

    One bounded deque of the last ``depth`` window verdicts per
    ``stream_id`` (a stream = one motor's continuous acquisition
    sequence).  ``max_streams`` caps memory with strict LRU eviction —
    the least recently updated stream is forgotten first, so a plant
    rotating thousands of short streams cannot grow the buffer without
    bound.

    Semantics (V6 ADDENDUM A3):
      * aggregated_class = modal verdict over the buffered window;
        ties are broken by RECENCY (the tied class with the most recent
        vote wins);
      * vote_share = winning count / buffered count;
      * stable = vote_share > 0.5 (a strict majority — a plurality
        verdict is NOT stable);
      * the buffer is an aggregation aid only: it never raises alerts
        and never overrides the per-window classifier — Cold Start
        §11.6's 4-signal confirmation is enforced downstream.
    """

    def __init__(self, depth: int = 5, max_streams: int = 2048) -> None:
        if depth < 3 or depth % 2 == 0:
            raise ValueError(
                f"TemporalVoteBuffer depth must be an odd int >= 3, got {depth}"
            )
        if max_streams < 1:
            raise ValueError("max_streams must be >= 1")
        self.depth = depth
        self.max_streams = max_streams
        self._streams: "OrderedDict[str, list[str]]" = OrderedDict()

    def push(self, stream_id: str, verdict: str) -> dict[str, object]:
        """Record one window verdict and return the aggregation block."""
        buf = self._streams.get(stream_id)
        if buf is None:
            if len(self._streams) >= self.max_streams:
                self._streams.popitem(last=False)  # evict least-recent
            buf = []
            self._streams[stream_id] = buf
        else:
            self._streams.move_to_end(stream_id)  # LRU touch
        buf.append(verdict)
        if len(buf) > self.depth:
            del buf[: len(buf) - self.depth]
        return self.aggregate(stream_id)

    def aggregate(self, stream_id: str) -> dict[str, object]:
        """Current aggregation for ``stream_id`` (no state change)."""
        buf = self._streams.get(stream_id, [])
        n = len(buf)
        if n == 0:
            return {
                "aggregated_class": None,
                "vote_count": 0,
                "windows_buffered": 0,
                "vote_share": 0.0,
                "stable": False,
            }
        counts: dict[str, int] = {}
        for v in buf:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values())
        # Recency tie-break: scanning newest -> oldest, the first tied
        # class to reach `best` running votes is the most recently
        # reinforced one and wins the tie.
        winner: str | None = None
        seen: dict[str, int] = {}
        for v in reversed(buf):
            seen[v] = seen.get(v, 0) + 1
            if counts[v] == best and seen[v] == best:
                winner = v
                break
        share = best / n
        return {
            "aggregated_class": winner,
            "vote_count": best,
            "windows_buffered": n,
            "vote_share": round(share, 4),
            "stable": share > 0.5,
        }

    def reset(self, stream_id: str) -> bool:
        """Forget a stream's history (e.g., motor re-commissioned).
        Returns True if the stream was known."""
        return self._streams.pop(stream_id, None) is not None


class ModelRuntime:
    """Loads and serves the active Sigmo V2 model bundle."""

    def __init__(self, models_root: Path = MODELS) -> None:
        self.models_root = Path(models_root)
        self.registry_path = self.models_root / "registry.json"
        self.version: str | None = None
        self.feature_names: list[str] = []
        self.class_names: list[str] = []
        # v6 Stage-2 bearing specialist state (V6 ADDENDUM A2/A3)
        self.bearing_classes: list[str] = []
        self.rpm_interact_base: list[str] = []
        self.bearing_submodel_enabled: bool = False
        self._bearing: xgb.Booster | None = None
        # §11.6 temporal aggregation (V6 ADDENDUM A3)
        self.vote_buffer = TemporalVoteBuffer(depth=5, max_streams=2048)
        self._booster: xgb.Booster | None = None
        self._gate: dict[str, object] | None = None

    # ------------------------------------------------------------------ load
    def _registry(self) -> dict[str, list[dict[str, object]]]:
        return json.loads(self.registry_path.read_text())

    def load(self, version: str | None = None) -> str:
        """Load ``version`` (default: the registry's active one) and return it."""
        reg = self._registry()
        if version is None:
            active = [v["version"] for v in reg["versions"] if v.get("active")]
            if not active:
                raise RuntimeError("registry has no active model version")
            version = str(active[-1])
        vdir = self.models_root / version
        # model.ubj (UBJSON) is the memory-lean serialization: loading
        # the 13-class v6c booster from JSON spikes ~194 MB higher RSS
        # than the identical UBJSON bytes (measured 2026-09-25; JSON
        # builds a full DOM before converting trees). Bundles may carry
        # either or both; UBJSON is preferred whenever present.
        for artifact in ARTIFACTS:
            if artifact == "model.json" and (vdir / "model.ubj").exists():
                continue  # UBJSON variant supersedes the JSON artifact
            if not (vdir / artifact).exists():
                raise FileNotFoundError(f"{version}: missing artifact {artifact}")

        feat = json.loads((vdir / "features.json").read_text())
        booster = xgb.Booster()
        booster.load_model(
            vdir / "model.ubj" if (vdir / "model.ubj").exists()
            else vdir / "model.json"
        )
        gate = joblib.load(vdir / "gate.joblib")

        # v6 optional Stage-2 specialist (defaults keep v5 bundles valid)
        bearing_classes = list(feat.get("bearing_classes", []))
        rpm_interact_base = list(feat.get("rpm_interact_base", []))
        enabled = bool(feat.get("bearing_submodel_enabled", False))
        bearing: xgb.Booster | None = None
        if enabled:
            bpath = vdir / BEARING_ARTIFACT
            if not bpath.exists():
                raise FileNotFoundError(
                    f"{version}: features.json sets bearing_submodel_enabled "
                    f"but {BEARING_ARTIFACT} is missing — the bundle is "
                    "inconsistent; refusing to load"
                )
            bearing = xgb.Booster()
            bearing.load_model(bpath)

        self.version = version
        self.feature_names = list(feat["feature_names"])
        self.class_names = list(feat["class_names"])
        self.bearing_classes = bearing_classes
        self.rpm_interact_base = rpm_interact_base
        self.bearing_submodel_enabled = enabled
        self._bearing = bearing
        self._booster = booster
        self._gate = gate
        return self.version

    def _require_loaded(self) -> None:
        if self._booster is None or self.version is None:
            raise RuntimeError("ModelRuntime.load() must be called first")

    # -------------------------------------------------------------- classify
    def _feature_vector(self, features: dict[str, float]) -> np.ndarray:
        missing = [n for n in self.feature_names if n not in features]
        if missing:
            raise KeyError(f"missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return np.asarray(
            [[float(features[n]) for n in self.feature_names]], dtype=np.float64
        )

    def _bearing_vector(self, features: dict[str, float]) -> np.ndarray:
        """125-column specialist input: base features + rpm interactions.

        Interactions are NaN-safe: a missing/NaN rpm_est or base feature
        yields NaN (XGBoost-native), exactly mirroring the trainer's
        ``_bearing_matrix``.
        """
        cols: list[float] = [float(features[n]) for n in self.feature_names]
        rpm = float(features.get("rpm_est", float("nan")))
        for base in self.rpm_interact_base:
            v = features.get(base, float("nan"))
            cols.append(rpm * float(v) if v == v and rpm == rpm else float("nan"))
        return np.asarray([cols], dtype=np.float64)

    def classify(self, features: dict[str, float]) -> dict[str, object]:
        """13-class verdict for one feature window.

        With ``bearing_submodel_enabled`` (V6 A2): when the stage-1
        verdict is a bearing class, the 3-class specialist re-decides —
        the top-level predicted_class/class_index/confidence carry the
        FINAL verdict, and the ``bearing_submodel`` block preserves the
        stage-1 verdict and the specialist's probabilities for audit.
        """
        self._require_loaded()
        row = self._feature_vector(features)
        proba = np.asarray(self._booster.inplace_predict(row), dtype=np.float64)[0]
        idx = int(np.argmax(proba))
        result: dict[str, object] = {
            "predicted_class": self.class_names[idx],
            "class_index": idx,
            "confidence": float(proba[idx]),
            "probabilities": {c: float(proba[i]) for i, c in enumerate(self.class_names)},
            "model_version": self.version,
        }

        if (
            self.bearing_submodel_enabled
            and self._bearing is not None
            and result["predicted_class"] in self.bearing_classes
        ):
            assert self._bearing is not None  # narrow for the type checker
            row_b = self._bearing_vector(features)
            proba_b = np.asarray(
                self._bearing.inplace_predict(row_b), dtype=np.float64
            )[0]
            b_idx = int(np.argmax(proba_b))
            stage1_class = str(result["predicted_class"])
            stage1_conf = float(result["confidence"])
            result["bearing_submodel"] = {
                "rerouted": True,
                "stage1_predicted_class": stage1_class,
                "stage1_confidence": stage1_conf,
                "predicted_class": self.bearing_classes[b_idx],
                "confidence": float(proba_b[b_idx]),
                "probabilities": {
                    c: float(proba_b[i]) for i, c in enumerate(self.bearing_classes)
                },
            }
            result["predicted_class"] = self.bearing_classes[b_idx]
            result["class_index"] = self.class_names.index(self.bearing_classes[b_idx])
            result["confidence"] = float(proba_b[b_idx])
        return result

    # -------------------------------------------------------- §11.6 temporal
    def classify_temporal(
        self, stream_id: str, features: dict[str, float]
    ) -> dict[str, object]:
        """Per-window verdict + §11.6 rolling-majority aggregation.

        The aggregation block reflects the last ``depth`` verdicts of
        this stream (LRU-capped across streams).  It NEVER replaces the
        per-window verdict and NEVER raises alerts by itself.
        """
        base = self.classify(features)
        aggregation = self.vote_buffer.push(stream_id, str(base["predicted_class"]))
        return {**base, "stream_id": stream_id, "temporal_aggregation": aggregation}

    def reset_stream(self, stream_id: str) -> bool:
        """Clear one stream's vote history (motor recommission / test).
        Returns True if the stream had buffered history."""
        return self.vote_buffer.reset(stream_id)

    # ------------------------------------------------------------------ gate
    def _gate_z(self, features: dict[str, float]) -> np.ndarray:
        gate = self._gate
        feats: list[str] = gate["features"]
        buckets: dict[str, dict[str, list[float]]] = gate["buckets"]
        rpm = float(features.get("rpm_est", 0.0) or 0.0)
        bkt = int(round(rpm / SPEED_BUCKET_RPM) * SPEED_BUCKET_RPM)
        stats_b = buckets.get(str(bkt))
        if stats_b is None:
            available = sorted(int(k) for k in buckets)
            nearest = min(available, key=lambda k: abs(k - bkt))
            stats_b = buckets[str(nearest)]
        mu = np.asarray(stats_b["mean"])
        sd = np.asarray(stats_b["std"])
        x = np.asarray([[float(features.get(f, float("nan"))) for f in feats]])
        z = (x - mu) / sd
        return np.clip(np.nan_to_num(z, nan=0.0), -10.0, 10.0)

    def anomaly_gate(self, features: dict[str, float]) -> tuple[bool, str]:
        """Return (is_novelty, human-readable detail)."""
        self._require_loaded()
        assert self._gate is not None
        z = self._gate_z(features)
        forest = self._gate["forest"]
        pred = int(forest.predict(z)[0])
        score = float(forest.score_samples(z)[0])
        flag = pred == -1
        detail = (
            f"novelty gate {'TRIPPED' if flag else 'clear'} "
            f"(iso score {score:.3f}, model {self.version})"
        )
        return flag, detail

    # -------------------------------------------------------------- rollback
    def rollback(self, version: str) -> str:
        """Re-activate ``version`` after artifact pre-check; reload; return it."""
        vdir = self.models_root / version
        if not vdir.is_dir():
            raise FileNotFoundError(f"no artifact directory for {version}")
        for artifact in ARTIFACTS:
            if not (vdir / artifact).exists():
                raise FileNotFoundError(f"{version}: missing artifact {artifact}")
        reg = self._registry()
        known = {str(v["version"]) for v in reg["versions"]}
        if version not in known:
            reg["versions"].append(
                {"version": version, "trained_at": "unknown", "accuracy": None,
                 "macro_f1": None, "active": False}
            )
        for v in reg["versions"]:
            v["active"] = str(v["version"]) == version
        self.registry_path.write_text(json.dumps(reg, indent=2))
        return self.load(version)
