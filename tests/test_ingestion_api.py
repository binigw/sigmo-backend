"""Sigmo V2 — Step 2 integration tests: full FastAPI ingestion path.

Drives the real ASGI app over HTTP (httpx ASGITransport) with
physically-modeled three-phase waveforms chunked into ESP32-sized 2048-
sample frames. The Supabase repository is exercised through a
capture-layer subclass whose only override is the network write — every
validation, cadence, and gating rule in the real code path executes.
The outage simulation verifies Section 9 store-and-forward end to end.

Test matrix:
  A1: contract rejection — wrong frame length, wrong rate, unequal phases
  A2: healthy motor — 16 frames -> 1 window, HEALTHY, persisted with
      the v6c verdict columns (predicted_class/confidence/version)
  A3: faulty motor (7% THD + 4% unbalance) -> ANOMALY with named breaches
      AND a dominant-fault verdict with 13-class probabilities
  A4: inrush window -> INRUSH_SUPPRESSED, no breaches, NO verdict
      (the classifier never sees startup transients)
  A5: DB outage -> row buffered (verdict preserved through the outage);
      recovery -> buffer flushed with original timestamps and
      bypass_cadence
  A6: sequence gap resets the accumulator (no corrupted stitched windows)
  A7: health endpoint reflects buffer depth and streaming motors
  A8: assessment endpoint — ONE structured JSON with snapshot +
      fault_assessment (live_runtime source) + deterministic Section 14
      RUL (CALCULATING_INSUFFICIENT_DATA under 24 h of history)
  A9: assessment 404 for a motor with no telemetry at all
  A10: CORS — Origin echo on GET, full preflight on OPTIONS
  A11: /api/v2/models/active reports the ACTIVE registry version
  A12: authentication — no key -> 401, invalid key -> 401 (all protected)
  A13: authentication — device key on an admin route -> 403 (scope)
  A14: authentication — health + models/active + docs stay OPEN
  A15: authentication — dashboard key unlocks assessment/views reads
  A16: authentication — fail-closed: zero keys configured -> 503
  A17: commissioning gate — telemetry for an UNREGISTERED motor is
       refused on the FIRST frame (422, actionable message, nothing
       buffered) per Section 11 Cold Start
  A18: foreign-key resilience — a row accepted during an outage for a
       motor never commissioned is DROPPED loudly at flush time and
       never blocks the rows behind it
  A19: restart survival — with the in-RAM live verdicts cleared (the
       production state after every service restart / free-tier wake),
       the assessment endpoint serves the persisted Supabase row
       (snapshot_source == "supabase") with the full evidence set
  A20: motors overview — GET /api/v2/motors lists every motor with
       telemetry (live union persisted, freshest wins) with health
       index, Stage and verdict; dashboard scope required
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import asyncpg
import numpy as np

sys.path.insert(0, "/home/user/sigmo_v2")

import httpx

from backend import main as sigmo_main
from backend.api import auth
from backend.analysis.rul import HealthIndexHistory, HealthIndexSample
from backend.config import DSP
from backend.db.repository import SigmoRepository
from backend.dsp.pipeline import ThreePhaseFeatures
from backend.ml.features import CLASS_NAMES

FS = DSP.sampling_rate_hz
FRAME = 2048
WINDOW = DSP.analysis_window_samples
FRAMES_PER_WINDOW = WINDOW // FRAME  # 16

# ACTIVE classifier version, read from the registry like production does
# (never hardcoded — the test stays valid across future model promotions).
import json
from pathlib import Path as _Path
_REGISTRY = json.loads(
    (_Path(__file__).resolve().parent.parent / "models" / "registry.json")
    .read_text(encoding="utf-8")
)
ACTIVE_VERSION = next(
    e["version"] for e in _REGISTRY["versions"] if e.get("active") is True
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    tag = "PASS" if condition else "FAIL"
    print(f"[{tag}] {name}: {detail}")
    if not condition:
        failures.append(name)


class CaptureRepository(SigmoRepository):
    """Real repository logic with only the wire-level write captured.

    `outage=True` raises ConnectionError exactly as asyncpg would during
    a Supabase outage, driving the service's real buffering branch.
    """

    async def get_motor_baseline(self, motor_id: str):
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        return self.baselines.get(motor_id, {})

    async def fetch_latest_snapshot(self, motor_id: str):
        """Latest captured steady-state row per motor (same ordering and
        INRUSH exclusion as the SQL read)."""
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        rows = [
            r for r in self.telemetry_rows
            if r["motor_id"] == motor_id and r["status"] != "INRUSH_SUPPRESSED"
        ]
        if not rows:
            return None
        latest = max(rows, key=lambda r: r["recorded_at"])
        return {
            "motor_id": latest["motor_id"],
            "recorded_at": latest["recorded_at"],
            "status": latest["status"],
            "fundamental_hz": latest["fundamental_hz"],
            "rms_a": latest["rms_a"], "rms_b": latest["rms_b"],
            "rms_c": latest["rms_c"],
            "thd_percent_a": latest["thd_percent_a"],
            "thd_percent_b": latest["thd_percent_b"],
            "thd_percent_c": latest["thd_percent_c"],
            "crest_factor_a": latest["crest_factor_a"],
            "crest_factor_b": latest["crest_factor_b"],
            "crest_factor_c": latest["crest_factor_c"],
            "rotor_sideband_db_a": latest["rotor_sideband_db_a"],
            "rotor_sideband_db_b": latest["rotor_sideband_db_b"],
            "rotor_sideband_db_c": latest["rotor_sideband_db_c"],
            "envelope_peak_hz_a": latest["envelope_peak_hz_a"],
            "envelope_peak_hz_b": latest["envelope_peak_hz_b"],
            "envelope_peak_hz_c": latest["envelope_peak_hz_c"],
            "unbalance_percent": latest["unbalance"],
            "zscore_max": latest["zscore_max"],
            "predicted_class": latest.get("predicted_class"),
            "model_confidence": latest.get("model_confidence"),
            "model_version": latest.get("model_version"),
        }

    async def fetch_telemetry_history(
        self, motor_id: str, hours: int, limit: int
    ):
        """Same semantics as the SQL read: 404-bound ValueError when the
        motor has no telemetry at all, otherwise the newest non-inrush
        captured rows inside the hours window, capped by limit,
        oldest-first (SQL-shape keys)."""
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        rows = [
            r for r in self.telemetry_rows
            if r["motor_id"] == motor_id
            and r["status"] != "INRUSH_SUPPRESSED"
        ]
        if not rows:
            raise ValueError(
                f"No telemetry recorded for motor {motor_id!r}"
            )
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        def _sql_shape(r: dict) -> dict:
            return {
                "recorded_at": r["recorded_at"],
                "status": r["status"],
                "fundamental_hz": r["fundamental_hz"],
                "rms_a": r["rms_a"], "rms_b": r["rms_b"], "rms_c": r["rms_c"],
                "thd_percent_a": r["thd_percent_a"],
                "thd_percent_b": r["thd_percent_b"],
                "thd_percent_c": r["thd_percent_c"],
                "crest_factor_a": r["crest_factor_a"],
                "crest_factor_b": r["crest_factor_b"],
                "crest_factor_c": r["crest_factor_c"],
                "rotor_sideband_db_a": r["rotor_sideband_db_a"],
                "rotor_sideband_db_b": r["rotor_sideband_db_b"],
                "rotor_sideband_db_c": r["rotor_sideband_db_c"],
                "unbalance_percent": r["unbalance"],
                "zscore_max": r["zscore_max"],
                "predicted_class": r.get("predicted_class"),
                "model_confidence": r.get("model_confidence"),
                "model_version": r.get("model_version"),
            }
        recent = sorted(
            (r for r in rows if r["recorded_at"] >= cutoff),
            key=lambda r: r["recorded_at"],
        )
        newest = recent[-limit:] if limit < len(recent) else recent
        return [_sql_shape(r) for r in newest]

    async def log_fault(
        self,
        motor_id: str,
        taxonomy_code: str,
        urgency_stage: int,
        model_confidence: float,
        population_sigma: float,
        absolute_threshold_breached: bool,
        trend_confirmed: bool,
        spectral_evidence: dict,
    ) -> int:
        """Capture the fault_logs wire write; computes the generated
        four_signal_confirmed column exactly as the SQL schema does."""
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        # Re-run the REAL validation from the parent class.
        if not 1 <= urgency_stage <= 4:
            raise ValueError(
                f"urgency_stage must be 1-4, got {urgency_stage}"
            )
        if not 0.0 <= model_confidence <= 1.0:
            raise ValueError(
                f"model_confidence must be 0-1, got {model_confidence}"
            )
        self._fault_seq += 1
        self.fault_rows.append({
            "id": self._fault_seq,
            "motor_id": motor_id,
            "detected_at": datetime.now(timezone.utc),
            "taxonomy_code": taxonomy_code,
            "urgency_stage": urgency_stage,
            "model_confidence": model_confidence,
            "population_sigma": population_sigma,
            "absolute_threshold_breached": absolute_threshold_breached,
            "trend_confirmed": trend_confirmed,
            "four_signal_confirmed": (
                population_sigma > 5.0
                and absolute_threshold_breached
                and model_confidence > 0.70
                and trend_confirmed
            ),
            "physically_verified": False,
            "verified_by": None,
            "spectral_evidence": spectral_evidence,
            "resolved_at": None,
        })
        return self._fault_seq

    async def get_latest_fault_episode(self, motor_id: str):
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        rows = [
            r for r in self.fault_rows if r["motor_id"] == motor_id
        ]
        if not rows:
            return None
        latest = max(rows, key=lambda r: r["detected_at"])
        return {
            "taxonomy_code": latest["taxonomy_code"],
            "detected_at": latest["detected_at"],
            "resolved_at": latest["resolved_at"],
        }

    async def fetch_fault_logs(self, motor_id, limit):
        """Same semantics as the SQL read: newest first, optional
        per-motor filter, capped at limit."""
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        rows = [
            r for r in self.fault_rows
            if motor_id is None or r["motor_id"] == motor_id
        ]
        newest = sorted(
            rows, key=lambda r: r["detected_at"], reverse=True
        )[:limit]
        return [dict(r) for r in newest]

    async def fetch_health_index_history(self, motor_id: str, lookback_days: int):
        """Same semantics as the SQL read: INRUSH rows excluded, earliest
        row, latest row, and the lookback-window samples ascending."""
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        rows = sorted(
            (r for r in self.telemetry_rows
             if r["motor_id"] == motor_id
             and r["status"] != "INRUSH_SUPPRESSED"),
            key=lambda r: r["recorded_at"],
        )
        if not rows:
            return HealthIndexHistory(None, None, [])
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        window_samples = [
            HealthIndexSample(r["recorded_at"], r["zscore_max"])
            for r in rows if r["recorded_at"] >= cutoff
        ]
        last = rows[-1]
        return HealthIndexHistory(
            first_recorded_at=rows[0]["recorded_at"],
            current_sample=HealthIndexSample(
                last["recorded_at"], last["zscore_max"]
            ),
            window_samples=window_samples,
        )

    # ---- Phase-5 asset registry + plant config emulation -------------
    def __init__(self) -> None:
        super().__init__()
        self.outage = False
        self.telemetry_rows: list[dict] = []
        self.baselines: dict[str, dict[str, tuple[float, float, float]]] = {}
        self.assets: dict[str, dict] = {}
        self.config: dict[str, dict] = {}
        # Motors for which the wire raises the REAL Supabase FK
        # error (unregistered at flush time after an outage).
        self.fk_motors: set[str] = set()
        # Fault-episode capture (Section 10 audit trail): mirrors
        # the fault_logs wire, including the DB-computed
        # four_signal_confirmed generated column.
        self.fault_rows: list[dict] = []
        self._fault_seq = 0
        # Alert dismissals (live alert hygiene): mirrors the
        # alert_dismissals wire in memory.
        self.alert_dismissal_rows: list[dict] = []

    async def fetch_alert_dismissals(self) -> list[dict]:
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        return [dict(r) for r in self.alert_dismissal_rows]

    async def upsert_alert_dismissals(self, pairs, dismissed_by: str) -> None:
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        now = datetime.now(timezone.utc)
        for mid, key in pairs:
            for row in self.alert_dismissal_rows:
                if row["motor_id"] == mid and row["alert_key"] == key:
                    row["dismissed_at"] = now
                    row["dismissed_by"] = dismissed_by
                    break
            else:
                self.alert_dismissal_rows.append(
                    {
                        "motor_id": mid,
                        "alert_key": key,
                        "dismissed_at": now,
                        "dismissed_by": dismissed_by,
                    }
                )

    async def delete_alert_dismissals(self, pairs) -> int:
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        removed = 0
        for mid, key in pairs:
            before = len(self.alert_dismissal_rows)
            self.alert_dismissal_rows = [
                r
                for r in self.alert_dismissal_rows
                if not (r["motor_id"] == mid and r["alert_key"] == key)
            ]
            removed += before - len(self.alert_dismissal_rows)
        return removed

    async def upsert_motor_asset(self, asset: dict) -> bool:
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        self.assets[asset["motor_id"]] = {
            "motor_id": asset["motor_id"],
            "mcc_panel_id": asset["mcc_panel_id"],
            "cabinet_number": asset["cabinet_number"],
            "line_zone_id": asset["line_zone_id"],
            "nameplate_kw": float(asset["nameplate_kw"]),
            "rated_rpm": int(asset["rated_rpm"]),
            "drive_type": asset["drive_type"],
            "asset_tag": asset.get("asset_tag"),
            "bearing_part_number": asset.get("bearing_part_number"),
            "spare_capacitor_spec": asset.get("spare_capacitor_spec"),
            "downtime_cost_etb_per_hour": (
                float(asset["downtime_cost_etb_per_hour"])
                if asset.get("downtime_cost_etb_per_hour") is not None
                else None
            ),
        }
        return True

    async def get_motor_asset(self, motor_id: str):
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        return self.assets.get(motor_id)

    async def list_motor_assets(self) -> list[dict]:
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        return [self.assets[k] for k in sorted(self.assets)]

    async def get_plant_config(self) -> dict:
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        return dict(self.config)

    async def set_plant_config(self, key, value, unit, description) -> bool:
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        self.config[key] = {
            "value": float(value), "unit": unit, "description": description,
        }
        return True

    async def fetch_all_latest_telemetry(self) -> list[dict]:
        """Same semantics as the SQL DISTINCT ON read: latest non-inrush
        row per motor."""
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        latest: dict[str, dict] = {}
        for r in self.telemetry_rows:
            if r["status"] == "INRUSH_SUPPRESSED":
                continue
            cur = latest.get(r["motor_id"])
            if cur is None or r["recorded_at"] > cur["recorded_at"]:
                latest[r["motor_id"]] = {
                    "motor_id": r["motor_id"],
                    "recorded_at": r["recorded_at"],
                    "status": r["status"],
                    "fundamental_hz": r["fundamental_hz"],
                    "thd_percent_a": r["thd_percent_a"],
                    "thd_percent_b": r["thd_percent_b"],
                    "thd_percent_c": r["thd_percent_c"],
                    "unbalance_percent": r["unbalance"],
                    "zscore_max": r["zscore_max"],
                    "predicted_class": r.get("predicted_class"),
                    "model_confidence": r.get("model_confidence"),
                    "model_version": r.get("model_version"),
                    "envelope_peak_hz_a": r["envelope_peak_hz_a"],
                    "envelope_peak_hz_b": r["envelope_peak_hz_b"],
                    "envelope_peak_hz_c": r["envelope_peak_hz_c"],
                    "rotor_sideband_db_a": r["rotor_sideband_db_a"],
                    "rotor_sideband_db_b": r["rotor_sideband_db_b"],
                    "rotor_sideband_db_c": r["rotor_sideband_db_c"],
                    "crest_factor_a": r["crest_factor_a"],
                    "crest_factor_b": r["crest_factor_b"],
                    "crest_factor_c": r["crest_factor_c"],
                }
        return [latest[k] for k in sorted(latest)]

    async def log_telemetry(
        self,
        motor_id: str,
        features: ThreePhaseFeatures,
        status: str,
        zscore_max: float,
        recorded_at: datetime | None = None,
        bypass_cadence: bool = False,
        predicted_class: str | None = None,
        model_confidence: float | None = None,
        model_version: str | None = None,
    ) -> bool:
        # Re-run the REAL validation + cadence logic from the parent class
        if status not in ("HEALTHY", "ANOMALY", "INRUSH_SUPPRESSED"):
            raise ValueError(f"Invalid telemetry status: {status!r}")
        now = recorded_at or datetime.now(timezone.utc)
        if not bypass_cadence and not self._due_for_logging(motor_id, status, now):
            return False
        if self.outage:
            raise ConnectionError("simulated Supabase outage")
        if motor_id in self.fk_motors:
            # Byte-identical to live Supabase: telemetry_features.motor_id
            # references motors(motor_id) (verified against production).
            raise asyncpg.ForeignKeyViolationError(
                'insert or update on table "telemetry_features" violates '
                'foreign key constraint "telemetry_features_motor_id_fkey"'
            )
        self.telemetry_rows.append(
            {
                "motor_id": motor_id,
                "status": status,
                "recorded_at": now,
                "bypass_cadence": bypass_cadence,
                "unbalance": features.current_unbalance_percent,
                "zscore_max": zscore_max,
                "predicted_class": predicted_class,
                "model_confidence": model_confidence,
                "model_version": model_version,
                # physical columns needed by fetch_latest_snapshot
                "fundamental_hz": features.mean_fundamental_hz,
                "rms_a": features.phase_a.rms_a,
                "rms_b": features.phase_b.rms_a,
                "rms_c": features.phase_c.rms_a,
                "thd_percent_a": features.phase_a.thd_percent,
                "thd_percent_b": features.phase_b.thd_percent,
                "thd_percent_c": features.phase_c.thd_percent,
                "crest_factor_a": features.phase_a.crest_factor,
                "crest_factor_b": features.phase_b.crest_factor,
                "crest_factor_c": features.phase_c.crest_factor,
                "rotor_sideband_db_a": features.phase_a.rotor_sideband_ratio_db,
                "rotor_sideband_db_b": features.phase_b.rotor_sideband_ratio_db,
                "rotor_sideband_db_c": features.phase_c.rotor_sideband_ratio_db,
                "envelope_peak_hz_a": features.phase_a.envelope_peak_hz,
                "envelope_peak_hz_b": features.phase_b.envelope_peak_hz,
                "envelope_peak_hz_c": features.phase_c.envelope_peak_hz,
            }
        )
        if not bypass_cadence:
            self._last_logged[motor_id] = now
        return True


def three_phase_window(
    amps: tuple[float, float, float],
    thd_frac: float = 0.0,
    inrush: bool = False,
    noise_scale: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physically-modeled 50 Hz three-phase set for the test rig."""
    rng = np.random.default_rng(7)
    t = np.arange(WINDOW) / FS
    phases = []
    for i, a in enumerate(amps):
        shift = -2.0 * np.pi * i / 3.0
        sig = a * np.sin(2 * np.pi * 50.0 * t + shift)
        if thd_frac > 0.0:
            sig += a * thd_frac * np.sin(2 * np.pi * 150.0 * t + shift)
        sig += rng.normal(0.0, noise_scale, WINDOW)
        if inrush:
            envelope = np.ones(WINDOW)
            envelope[: WINDOW // 4] = 6.0
            sig = sig * envelope
        phases.append(sig)
    return phases[0], phases[1], phases[2]


async def send_window(
    client: httpx.AsyncClient,
    motor_id: str,
    waves: tuple[np.ndarray, np.ndarray, np.ndarray],
    start_seq: int = 0,
    skip_seq_at: int | None = None,
) -> list[dict]:
    """Chunk a window into 2048-sample frames and POST them in order."""
    responses = []
    seq = start_seq
    for k in range(FRAMES_PER_WINDOW):
        s = slice(k * FRAME, (k + 1) * FRAME)
        if skip_seq_at is not None and k == skip_seq_at:
            seq += 1  # simulate a dropped frame
        payload = {
            "motor_id": motor_id,
            "device_id": "esp32-node-01",
            "sequence": seq,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "sampling_rate_hz": FS,
            "i_a": waves[0][s].tolist(),
            "i_b": waves[1][s].tolist(),
            "i_c": waves[2][s].tolist(),
        }
        r = await client.post("/api/v2/telemetry", json=payload)
        responses.append({"status_code": r.status_code, "body": r.json()})
        seq += 1
    return responses


# Section-7 test keys (one per scope). Loaded directly into the auth
# module because ASGITransport does not run the app lifespan.
TEST_DEVICE_KEY = "test-device-key-000"
TEST_DASHBOARD_KEY = "test-dashboard-key-000"
TEST_ADMIN_KEY = "test-admin-key-000"
DASH_HEADERS = {"X-API-Key": TEST_DASHBOARD_KEY}


async def run() -> None:
    repo = CaptureRepository()
    sigmo_main.service._repo = repo  # inject capture layer into real service
    # Commissioning precedes telemetry (Section 11 Cold Start): every
    # motor that will stream must exist in the registry first — exactly
    # what the production FK on telemetry_features.motor_id enforces.
    # Minimal assets only (no rates/bearing) so downstream expectations
    # about unconfigured business fields stay untouched.
    for mid in ("MTR-HEALTHY", "MTR-FAULTY", "MTR-STARTUP",
                "MTR-OUTAGE", "MTR-GAPPED"):
        await repo.upsert_motor_asset({
            "motor_id": mid, "mcc_panel_id": "MCC-01",
            "cabinet_number": "C-01", "line_zone_id": "LINE-A/ZONE-1",
            "nameplate_kw": 75.0, "rated_rpm": 2970, "drive_type": "DOL",
        })
    auth.load_keys(
        f"{TEST_DEVICE_KEY}:device,{TEST_DASHBOARD_KEY}:dashboard,"
        f"{TEST_ADMIN_KEY}:admin"
    )
    transport = httpx.ASGITransport(app=sigmo_main.app)
    async with (
        httpx.AsyncClient(
            transport=transport, base_url="http://sigmo-test", timeout=60.0,
            headers={"X-API-Key": TEST_DEVICE_KEY},
        ) as client,
        httpx.AsyncClient(
            transport=transport, base_url="http://sigmo-test", timeout=60.0
        ) as noauth,
    ):

        # A1 — contract rejections
        bad = {
            "motor_id": "M1", "device_id": "d", "sequence": 0,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "sampling_rate_hz": FS,
            "i_a": [0.0] * 1000, "i_b": [0.0] * 1000, "i_c": [0.0] * 1000,
        }
        r1 = await client.post("/api/v2/telemetry", json=bad)
        bad_rate = dict(bad, i_a=[0.0] * 2048, i_b=[0.0] * 2048,
                        i_c=[0.0] * 2048, sampling_rate_hz=8000.0)
        r2 = await client.post("/api/v2/telemetry", json=bad_rate)
        bad_uneven = dict(bad, i_a=[0.0] * 2048, i_b=[0.0] * 2048, i_c=[0.0] * 4096)
        r3 = await client.post("/api/v2/telemetry", json=bad_uneven)
        check(
            "A1 contract rejection",
            r1.status_code == 422 and r2.status_code == 422 and r3.status_code == 422,
            f"lengths={r1.status_code}, rate={r2.status_code}, uneven={r3.status_code}",
        )

        # A2 — healthy motor window
        healthy = three_phase_window((10.0, 10.0, 10.0))
        resp = await send_window(client, "MTR-HEALTHY", healthy)
        finals = [r for r in resp if r["body"].get("window_analyzed")]
        a = finals[-1]["body"]["analysis"] if finals else None
        check(
            "A2 healthy window",
            len(finals) == 1 and a is not None and a["status"] == "HEALTHY"
            and a["persisted"] and not a["buffered_offline"]
            and abs(a["fundamental_hz"] - 50.0) < 0.05,
            f"status={a['status'] if a else None}, "
            f"fund={a['fundamental_hz'] if a else 0:.3f} Hz, "
            f"rows={len(repo.telemetry_rows)}",
        )
        row = repo.telemetry_rows[-1]
        check(
            "A2b verdict persisted",
            a is not None and a["dominant_fault"] in CLASS_NAMES
            and a["model_version"] == ACTIVE_VERSION
            and row["predicted_class"] == a["dominant_fault"]
            and row["model_confidence"] == a["model_confidence"]
            and row["model_version"] == ACTIVE_VERSION,
            f"row verdict={row['predicted_class']} "
            f"conf={row['model_confidence']:.3f} ver={row['model_version']}",
        )

        # A3 — faulty motor: 7% third harmonic + unbalanced phases
        faulty = three_phase_window((10.0, 10.0, 8.8), thd_frac=0.07)
        resp = await send_window(client, "MTR-FAULTY", faulty)
        a = [r for r in resp if r["body"].get("window_analyzed")][-1]["body"]["analysis"]
        breach_text = " | ".join(a["absolute_threshold_breaches"])
        check(
            "A3 anomaly gating",
            a["status"] == "ANOMALY"
            and any("THD" in b for b in a["absolute_threshold_breaches"])
            and any("unbalance" in b.lower() for b in a["absolute_threshold_breaches"]),
            f"breaches: {breach_text}",
        )
        probs = a.get("probabilities") or {}
        check(
            "A3b verdict on faulty window",
            a["dominant_fault"] in CLASS_NAMES
            and a["dominant_fault"] != "HEALTHY"
            and len(probs) == len(CLASS_NAMES)
            and abs(sum(probs.values()) - 1.0) < 1e-6
            and a["temporal_aggregation"]["windows_buffered"] == 1,
            f"dominant={a['dominant_fault']} conf={a['model_confidence']:.3f} "
            f"probs={len(probs)} keys",
        )

        # A4 — inrush suppression
        inrush = three_phase_window((10.0, 10.0, 10.0), inrush=True)
        resp = await send_window(client, "MTR-STARTUP", inrush)
        a = [r for r in resp if r["body"].get("window_analyzed")][-1]["body"]["analysis"]
        check(
            "A4 inrush suppressed",
            a["status"] == "INRUSH_SUPPRESSED" and not a["absolute_threshold_breaches"]
            and a["dominant_fault"] is None and a["probabilities"] is None,
            f"status={a['status']}, breaches={a['absolute_threshold_breaches']}, "
            f"verdict={a['dominant_fault']}",
        )

        # A5 — outage buffering then recovery flush
        repo.outage = True
        rows_before = len(repo.telemetry_rows)
        resp = await send_window(client, "MTR-OUTAGE", healthy)
        a = [r for r in resp if r["body"].get("window_analyzed")][-1]["body"]["analysis"]
        buffered_ok = (
            not a["persisted"] and a["buffered_offline"]
            and sigmo_main.service.offline_buffer_depth == 1
        )
        outage_time = datetime.now(timezone.utc)
        repo.outage = False
        resp = await send_window(client, "MTR-OUTAGE", healthy, start_seq=100)
        a2 = [r for r in resp if r["body"].get("window_analyzed")][-1]["body"]["analysis"]
        flushed = [r for r in repo.telemetry_rows[rows_before:] if r["bypass_cadence"]]
        check(
            "A5 store-and-forward",
            buffered_ok and a2["persisted"]
            and sigmo_main.service.offline_buffer_depth == 0
            and len(flushed) == 1
            and flushed[0]["recorded_at"] <= outage_time,
            f"buffered_ok={buffered_ok}, flushed={len(flushed)}, "
            f"depth_after={sigmo_main.service.offline_buffer_depth}",
        )
        check(
            "A5b verdict survives outage",
            len(flushed) == 1
            and flushed[0]["predicted_class"] in CLASS_NAMES
            and flushed[0]["model_version"] == ACTIVE_VERSION,
            f"flushed verdict={flushed[0]['predicted_class'] if flushed else None}",
        )

        # A6 — sequence gap resets accumulator (no window completes)
        resp = await send_window(client, "MTR-GAPPED", healthy, skip_seq_at=8)
        analyzed = [r for r in resp if r["body"].get("window_analyzed")]
        last_fill = resp[-1]["body"]["window_fill_percent"]
        check(
            "A6 sequence-gap reset",
            len(analyzed) == 0 and 0.0 < last_fill < 100.0,
            f"windows_completed={len(analyzed)}, final fill={last_fill:.1f}%",
        )

        # A7 — health endpoint
        h = (await client.get("/api/v2/health")).json()
        check(
            "A7 health endpoint",
            h["service"] == "sigmo-v2-backend" and h["offline_buffer_depth"] == 0
            and h["motors_streaming"] >= 4,
            f"db={h['database_connected']}, buffer={h['offline_buffer_depth']}, "
            f"motors={h['motors_streaming']}",
        )

        # A8 — comprehensive assessment payload in ONE structured JSON
        r = await client.get(
            "/api/v2/motors/MTR-FAULTY/assessment", headers=DASH_HEADERS
        )
        m = r.json()
        fa = m.get("fault_assessment") or {}
        rul = m.get("rul") or {}
        check(
            "A8 assessment payload",
            r.status_code == 200
            and m["motor_id"] == "MTR-FAULTY"
            and m["model_version"] == ACTIVE_VERSION
            and m["database_connected"] is True
            and m["snapshot"] is not None
            and m["snapshot_source"] == "live_runtime"
            and fa.get("source") == "live_runtime"
            and fa.get("dominant_fault") in CLASS_NAMES
            and fa.get("is_fault") is True
            and len(fa.get("probabilities") or {}) == len(CLASS_NAMES)
            and fa.get("temporal_aggregation", {}).get("windows_buffered") == 1
            and rul.get("rul_status") == "CALCULATING_INSUFFICIENT_DATA"
            and rul.get("rul_days_estimate") == "Calculating... (Insufficient Data)",
            f"snapshot={m['snapshot_source']}, fault={fa.get('dominant_fault')} "
            f"({fa.get('source')}), rul={rul.get('rul_status')}",
        )

        # A9 — unknown motor -> 404 (dashboard scope)
        r = await client.get(
            "/api/v2/motors/MTR-NONE/assessment", headers=DASH_HEADERS
        )
        check(
            "A9 assessment 404",
            r.status_code == 404 and "No telemetry recorded" in r.json()["detail"],
            f"status={r.status_code}, detail={r.json().get('detail', '')[:60]}",
        )

        # A10 — CORS for the external Replit frontend
        r = await client.get(
            "/api/v2/motors/MTR-FAULTY/assessment",
            headers={"Origin": "https://sigmo-dashboard.replit.app",
                     **DASH_HEADERS},
        )
        echo = r.headers.get("access-control-allow-origin")
        pre = await client.options(
            "/api/v2/motors/MTR-FAULTY/assessment",
            headers={
                "Origin": "https://sigmo-dashboard.replit.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        check(
            "A10 CORS",
            echo is not None and pre.status_code == 200
            and pre.headers.get("access-control-allow-methods") is not None,
            f"echo={echo}, preflight={pre.status_code}, "
            f"methods={pre.headers.get('access-control-allow-methods')}",
        )

        # A11 — active model metadata endpoint (open path, no key)
        r = await noauth.get("/api/v2/models/active")
        check(
            "A11 active model",
            r.status_code == 200
            and r.json()["model_version"] == ACTIVE_VERSION,
            f"body={r.json()}",
        )

        # A12 — no key / invalid key -> 401 on every protected route
        r1 = await noauth.post("/api/v2/telemetry", json={})
        r2 = await noauth.get("/api/v2/motors/MTR-HEALTHY/assessment")
        r3 = await client.get(  # valid-format header, unknown key
            "/api/v2/motors/MTR-HEALTHY/assessment",
            headers={"X-API-Key": "not-a-real-key"},
        )
        check(
            "A12 auth required",
            r1.status_code == 401 and r2.status_code == 401
            and r3.status_code == 401,
            f"telemetry={r1.status_code}, read={r2.status_code}, "
            f"badkey={r3.status_code}",
        )

        # A13 — device key must NOT reach the admin surface
        r = await client.get("/api/v2/admin/motor-assets")
        r2 = await client.put("/api/v2/admin/plant-config/x", json={
            "value": 1.0, "unit": "u", "description": "d"
        })
        check(
            "A13 scope enforcement",
            r.status_code == 403 and r2.status_code == 403,
            f"admin-get={r.status_code}, admin-put={r2.status_code}",
        )

        # A14 — open paths stay reachable without any key
        h = await noauth.get("/api/v2/health")
        m = await noauth.get("/api/v2/models/active")
        d = await noauth.get("/openapi.json")
        check(
            "A14 open paths",
            h.status_code == 200 and m.status_code == 200
            and d.status_code == 200,
            f"health={h.status_code}, model={m.status_code}, "
            f"openapi={d.status_code}",
        )

        # A15 — dashboard key unlocks the read surface (401 vs 200)
        r_no = await noauth.get("/api/v2/views/executive")
        r_yes = await client.get(
            "/api/v2/views/executive", headers=DASH_HEADERS
        )
        check(
            "A15 dashboard scope reads",
            r_no.status_code == 401 and r_yes.status_code == 200,
            f"without-key={r_no.status_code}, "
            f"with-dashboard-key={r_yes.status_code}",
        )

        # A16 — fail-closed: zero keys configured -> 503 on protected
        auth.reset_keys()
        r = await noauth.post("/api/v2/telemetry", json={})
        check(
            "A16 fail-closed without keys",
            r.status_code == 503 and "not configured" in r.json()["detail"],
            f"status={r.status_code}",
        )
        auth.load_keys(
            f"{TEST_DEVICE_KEY}:device,{TEST_DASHBOARD_KEY}:dashboard,"
            f"{TEST_ADMIN_KEY}:admin"
        )  # restore for any later use

        # A17 — commissioning gate: an UNREGISTERED motor is refused on
        # the FIRST frame with an actionable 422 (never 15 silent frames
        # then a failure; never a silently-buffered un-writable row).
        resp17 = await send_window(client, "MTR-UNREGISTERED", healthy)
        d17 = resp17[0]["body"].get("detail", "")
        check(
            "A17 commissioning gate",
            resp17[0]["status_code"] == 422
            and all(r["status_code"] == 422 for r in resp17)
            and "not registered" in d17
            and "/api/v2/admin/motor-assets/MTR-UNREGISTERED" in d17
            and "Section 11" in d17
            and sigmo_main.service.offline_buffer_depth == 0,
            f"first={resp17[0]['status_code']}, depth="
            f"{sigmo_main.service.offline_buffer_depth}, detail={d17[:70]}",
        )

        # A18 — FK resilience: telemetry for a never-commissioned motor
        # accepted during an outage (fail-open, Section 9) is DROPPED
        # loudly at flush time and does NOT block the rows behind it.
        repo.fk_motors.add("MTR-UNREG-FK")
        repo.outage = True
        await send_window(client, "MTR-UNREG-FK", healthy)
        buffered_depth = sigmo_main.service.offline_buffer_depth
        repo.outage = False
        await repo.upsert_motor_asset({
            "motor_id": "MTR-FLUSH-01", "mcc_panel_id": "MCC-01",
            "cabinet_number": "C-02", "line_zone_id": "LINE-A/ZONE-1",
            "nameplate_kw": 75.0, "rated_rpm": 2970, "drive_type": "DOL",
        })
        resp18 = await send_window(client, "MTR-FLUSH-01", healthy)
        a18 = [r for r in resp18 if r["body"].get("window_analyzed")][-1]
        a18 = a18["body"]["analysis"]
        rows18 = {r["motor_id"] for r in repo.telemetry_rows}
        check(
            "A18 FK drop at flush",
            buffered_depth == 1
            and a18["persisted"] and not a18["buffered_offline"]
            and sigmo_main.service.offline_buffer_depth == 0
            and "MTR-FLUSH-01" in rows18
            and "MTR-UNREG-FK" not in rows18,
            f"buffered={buffered_depth}, persisted={a18['persisted']}, "
            f"depth_after={sigmo_main.service.offline_buffer_depth}",
        )

        # A20 — motors overview list (dashboard status grid source)
        r = await client.get("/api/v2/motors", headers=DASH_HEADERS)
        ov = r.json()
        ov_by_id = {m["motor_id"]: m for m in ov.get("motors", [])}
        entry = ov_by_id.get("MTR-FAULTY", {})
        check(
            "A20 motors overview",
            r.status_code == 200
            and ov.get("database_connected") is True
            and ov.get("model_version") == ACTIVE_VERSION
            and {"MTR-HEALTHY", "MTR-FAULTY", "MTR-OUTAGE",
                 "MTR-FLUSH-01"}.issubset(set(ov_by_id))
            and entry.get("stage") == 2
            and entry.get("is_fault") is True
            and entry.get("predicted_class") in CLASS_NAMES
            and isinstance(entry.get("health_percent"), (int, float))
            and entry.get("source") in ("live_runtime", "supabase"),
            f"motors={sorted(ov_by_id)}, faulty stage={entry.get('stage')} "
            f"fault={entry.get('predicted_class')} src={entry.get('source')}",
        )

        # A19 — restart survival: the production state after every
        # service restart (free-tier wake) has NO live verdicts in RAM,
        # so every read must come from the persisted Supabase row. This
        # is the regression test for the KeyError('envelope_peak_hz_a')
        # that broke both views after the first free-tier sleep.
        sigmo_main.service._live_verdicts.clear()
        r = await client.get(
            "/api/v2/motors/MTR-FAULTY/assessment", headers=DASH_HEADERS
        )
        m = r.json()
        fa = m.get("fault_assessment") or {}
        rul19 = m.get("rul") or {}
        check(
            "A19 restart survival (supabase source)",
            r.status_code == 200
            and m["snapshot_source"] == "supabase"
            and fa.get("source") == "supabase"
            and fa.get("dominant_fault") in CLASS_NAMES
            and fa.get("model_version") == ACTIVE_VERSION
            and m["snapshot"] is not None
            and m["snapshot"]["thd_percent_max"] > 0
            and rul19.get("rul_status") is not None,
            f"source={m.get('snapshot_source')}, fault={fa.get('dominant_fault')}, "
            f"thd={m['snapshot']['thd_percent_max'] if m.get('snapshot') else None}",
        )

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("RESULT: ALL INGESTION API TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run())
