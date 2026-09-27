"""Sigmo V2 — Supabase PostgreSQL repository layer.

The ONLY persistence path in the system (SIGMO_RULES.md Section 4.2).
Connects to the external Supabase Postgres instance via asyncpg using
DATABASE_URL from Secrets. Implements:

  - adaptive-cadence telemetry logging (Section 10)
  - fault log persistence with 4-signal confirmation fields (Section 11.6)
  - personalized baseline reads/writes (Section 11.4)
  - the 90-day healthy-telemetry cleanup call (Section 10)
  - Health-Index history fetch for deterministic RUL (Section 14.2)

Raw waveform arrays never reach this module by design: it accepts only
the feature dataclasses produced by the DSP pipeline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import asyncpg

from ..analysis.rul import HealthIndexHistory, HealthIndexSample
from ..config import DB
from ..dsp.pipeline import ThreePhaseFeatures


class SigmoRepository:
    """Async repository over the external Supabase PostgreSQL database."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._last_logged: dict[str, datetime] = {}

    async def connect(self) -> None:
        """Create the connection pool. Fails hard without DATABASE_URL."""
        dsn = DB.require_dsn()
        self._pool = await asyncpg.create_pool(
            dsn=dsn,
            min_size=DB.min_pool_size,
            max_size=DB.max_pool_size,
            command_timeout=30.0,
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "Repository not connected. Call connect() during FastAPI "
                "startup before handling telemetry."
            )
        return self._pool

    async def apply_schema(self, schema_sql: str) -> None:
        """Apply schema.sql idempotently (deployment bootstrap)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(schema_sql)

    # ------------------------------------------------------------------
    # Adaptive-cadence telemetry logging (Section 10)
    # ------------------------------------------------------------------
    def _due_for_logging(self, motor_id: str, status: str, now: datetime) -> bool:
        """Healthy: 30-min cadence. Anomaly: 1-min cadence. Inrush windows
        are logged at healthy cadence so commissioning history exists."""
        interval = (
            DB.anomaly_log_interval_s
            if status == "ANOMALY"
            else DB.healthy_log_interval_s
        )
        last = self._last_logged.get(motor_id)
        return last is None or (now - last).total_seconds() >= interval

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
        """Persist one feature row if the adaptive cadence allows it.

        `recorded_at` / `bypass_cadence` support store-and-forward replay
        (SIGMO_RULES.md Section 9): rows buffered during a database outage
        are flushed later with their original capture timestamps and must
        not be re-filtered by the live cadence gate.

        Returns True if a row was written, False if skipped by cadence.
        """
        if status not in ("HEALTHY", "ANOMALY", "INRUSH_SUPPRESSED"):
            raise ValueError(f"Invalid telemetry status: {status!r}")
        now = recorded_at or datetime.now(timezone.utc)
        if not bypass_cadence and not self._due_for_logging(motor_id, status, now):
            return False

        pa, pb, pc = features.phase_a, features.phase_b, features.phase_c
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO telemetry_features (
                    motor_id, recorded_at, status, fundamental_hz,
                    rms_a, rms_b, rms_c,
                    thd_percent_a, thd_percent_b, thd_percent_c,
                    crest_factor_a, crest_factor_b, crest_factor_c,
                    kurtosis_a, kurtosis_b, kurtosis_c,
                    rotor_sideband_db_a, rotor_sideband_db_b, rotor_sideband_db_c,
                    envelope_peak_hz_a, envelope_peak_hz_b, envelope_peak_hz_c,
                    unbalance_percent, zscore_max,
                    predicted_class, model_confidence, model_version
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                          $15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27)
                """,
                motor_id, now, status, features.mean_fundamental_hz,
                pa.rms_a, pb.rms_a, pc.rms_a,
                pa.thd_percent, pb.thd_percent, pc.thd_percent,
                pa.crest_factor, pb.crest_factor, pc.crest_factor,
                pa.kurtosis, pb.kurtosis, pc.kurtosis,
                pa.rotor_sideband_ratio_db, pb.rotor_sideband_ratio_db,
                pc.rotor_sideband_ratio_db,
                pa.envelope_peak_hz, pb.envelope_peak_hz, pc.envelope_peak_hz,
                features.current_unbalance_percent, zscore_max,
                predicted_class, model_confidence, model_version,
            )
        if not bypass_cadence:
            self._last_logged[motor_id] = now
        return True

    # ------------------------------------------------------------------
    # Fault logs (Section 10 permanent retention, Section 11.6 gating)
    # ------------------------------------------------------------------
    async def log_fault(
        self,
        motor_id: str,
        taxonomy_code: str,
        urgency_stage: int,
        model_confidence: float,
        population_sigma: float,
        absolute_threshold_breached: bool,
        trend_confirmed: bool,
        spectral_evidence: dict[str, Any],
    ) -> int:
        """Insert a fault record; the DB computes four_signal_confirmed."""
        if not 1 <= urgency_stage <= 4:
            raise ValueError(f"urgency_stage must be 1-4, got {urgency_stage}")
        if not 0.0 <= model_confidence <= 1.0:
            raise ValueError(f"model_confidence must be 0-1, got {model_confidence}")
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO fault_logs (
                    motor_id, taxonomy_code, urgency_stage, model_confidence,
                    population_sigma, absolute_threshold_breached,
                    trend_confirmed, spectral_evidence
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
                RETURNING id
                """,
                motor_id, taxonomy_code, urgency_stage, model_confidence,
                population_sigma, absolute_threshold_breached,
                trend_confirmed, json.dumps(spectral_evidence),
            )
        return int(row["id"])

    # ------------------------------------------------------------------
    # Baselines (Cold Start Protocol, Section 11)
    # ------------------------------------------------------------------
    async def get_motor_baseline(self, motor_id: str) -> dict[str, tuple[float, float, float]]:
        """Return {feature_name: (mean, std, z_threshold)} for a motor."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT feature_name, baseline_mean, baseline_std,
                       zscore_alert_threshold
                FROM motor_baselines WHERE motor_id = $1
                """,
                motor_id,
            )
        return {
            r["feature_name"]: (
                float(r["baseline_mean"]),
                float(r["baseline_std"]),
                float(r["zscore_alert_threshold"]),
            )
            for r in rows
        }

    async def upsert_motor_baseline(
        self,
        motor_id: str,
        feature_stats: dict[str, tuple[float, float]],
    ) -> None:
        """Write personalized baseline — caller MUST have verified the
        motor's baseline_state is HEALTHY_CONFIRMED (Section 11.4)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            state = await conn.fetchval(
                "SELECT baseline_state FROM motors WHERE motor_id = $1", motor_id
            )
            if state != "HEALTHY_CONFIRMED":
                raise PermissionError(
                    f"Motor {motor_id} baseline_state is {state!r}; personalized "
                    "baselines may only be written for HEALTHY_CONFIRMED motors "
                    "(SIGMO_RULES.md Section 11.4)."
                )
            async with conn.transaction():
                for name, (mean, std) in feature_stats.items():
                    await conn.execute(
                        """
                        INSERT INTO motor_baselines
                            (motor_id, feature_name, baseline_mean, baseline_std)
                        VALUES ($1,$2,$3,$4)
                        ON CONFLICT (motor_id, feature_name) DO UPDATE
                        SET baseline_mean = EXCLUDED.baseline_mean,
                            baseline_std = EXCLUDED.baseline_std,
                            established_at = now()
                        """,
                        motor_id, name, mean, std,
                    )

    # ------------------------------------------------------------------
    # Retention (Section 10)
    # ------------------------------------------------------------------
    async def run_retention_cleanup(self) -> int:
        """Execute the 90-day healthy-telemetry cleanup. Returns rows removed."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            return int(await conn.fetchval("SELECT cleanup_healthy_telemetry()"))

    # ------------------------------------------------------------------
    # Asset registry + plant configuration (Phase 5, SIGMO_RULES.md 6)
    # ------------------------------------------------------------------
    async def upsert_motor_asset(self, asset: dict[str, Any]) -> bool:
        """Register or update one motor's asset record (Section 6.b).

        Physical identity (MCC panel, cabinet, line/zone, nameplate kW,
        rated RPM, drive type) plus Phase-5 spares/cost extension
        columns. Full-row upsert — commissioning is an explicit,
        audited act, so the caller supplies every field.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO motors (
                    motor_id, mcc_panel_id, cabinet_number, line_zone_id,
                    nameplate_kw, rated_rpm, drive_type,
                    asset_tag, bearing_part_number, spare_capacitor_spec,
                    downtime_cost_etb_per_hour
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                ON CONFLICT (motor_id) DO UPDATE SET
                    mcc_panel_id = EXCLUDED.mcc_panel_id,
                    cabinet_number = EXCLUDED.cabinet_number,
                    line_zone_id = EXCLUDED.line_zone_id,
                    nameplate_kw = EXCLUDED.nameplate_kw,
                    rated_rpm = EXCLUDED.rated_rpm,
                    drive_type = EXCLUDED.drive_type,
                    asset_tag = EXCLUDED.asset_tag,
                    bearing_part_number = EXCLUDED.bearing_part_number,
                    spare_capacitor_spec = EXCLUDED.spare_capacitor_spec,
                    downtime_cost_etb_per_hour =
                        EXCLUDED.downtime_cost_etb_per_hour
                """,
                asset["motor_id"], asset["mcc_panel_id"],
                asset["cabinet_number"], asset["line_zone_id"],
                asset["nameplate_kw"], int(asset["rated_rpm"]),
                asset["drive_type"], asset.get("asset_tag"),
                asset.get("bearing_part_number"),
                asset.get("spare_capacitor_spec"),
                asset.get("downtime_cost_etb_per_hour"),
            )
        return True

    async def get_motor_asset(self, motor_id: str) -> dict[str, Any] | None:
        """One motor's full asset record, or None when not registered."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT motor_id, mcc_panel_id, cabinet_number, line_zone_id,
                       nameplate_kw, rated_rpm, drive_type, baseline_state,
                       asset_tag, bearing_part_number, spare_capacitor_spec,
                       downtime_cost_etb_per_hour
                FROM motors WHERE motor_id = $1
                """,
                motor_id,
            )
        return dict(row) if row is not None else None

    async def list_motor_assets(self) -> list[dict[str, Any]]:
        """All registered motor asset records (executive aggregation)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT motor_id, mcc_panel_id, cabinet_number, line_zone_id,
                       nameplate_kw, rated_rpm, drive_type, baseline_state,
                       asset_tag, bearing_part_number, spare_capacitor_spec,
                       downtime_cost_etb_per_hour
                FROM motors ORDER BY motor_id
                """
            )
        return [dict(r) for r in rows]

    async def get_plant_config(self) -> dict[str, dict[str, Any]]:
        """All explicitly-configured plant business rates (Section 6.a).

        Returns {key: {value, unit, description}}; an absent key simply
        means "not configured" — callers must surface that honestly.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key, value, unit, description
                FROM plant_config ORDER BY key
                """
            )
        return {
            r["key"]: {
                "value": float(r["value"]),
                "unit": r["unit"],
                "description": r["description"],
            }
            for r in rows
        }

    async def set_plant_config(
        self, key: str, value: float, unit: str, description: str
    ) -> bool:
        """Set one business rate (admin action, audited by updated_at)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO plant_config (key, value, unit, description,
                                          updated_at)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    description = EXCLUDED.description,
                    updated_at = now()
                """,
                key, float(value), unit, description,
            )
        return True

    async def fetch_all_latest_telemetry(self) -> list[dict[str, Any]]:
        """Latest non-inrush telemetry row per motor (Section 6.a basis).

        One DISTINCT ON read: per motor, the newest row that is not
        INRUSH_SUPPRESSED (startup transients are not a health state —
        same exclusion as the Section 14.2.2 HI series).
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT ON (motor_id)
                    motor_id, recorded_at, status, fundamental_hz,
                    thd_percent_a, thd_percent_b, thd_percent_c,
                    unbalance_percent, zscore_max,
                    predicted_class, model_confidence, model_version,
                    envelope_peak_hz_a, envelope_peak_hz_b,
                    envelope_peak_hz_c,
                    rotor_sideband_db_a, rotor_sideband_db_b,
                    rotor_sideband_db_c,
                    crest_factor_a, crest_factor_b, crest_factor_c
                FROM telemetry_features
                WHERE status <> 'INRUSH_SUPPRESSED'
                ORDER BY motor_id, recorded_at DESC
                """
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Motor assessment read path (Phase 4, SIGMO_RULES.md 6/14)
    # ------------------------------------------------------------------
    async def fetch_latest_snapshot(self, motor_id: str) -> dict[str, Any] | None:
        """Latest steady-state telemetry row (physical features + gating
        status + the persisted v6c verdict) for one motor, newest first.

        INRUSH_SUPPRESSED rows are excluded — startup transients are not
        a health state (same exclusion as Section 14.2.2 and the
        plant-wide read). None when the motor has no steady-state
        telemetry yet. Pure read, one statement."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT motor_id, recorded_at, status, fundamental_hz,
                       rms_a, rms_b, rms_c,
                       thd_percent_a, thd_percent_b, thd_percent_c,
                       crest_factor_a, crest_factor_b, crest_factor_c,
                       rotor_sideband_db_a, rotor_sideband_db_b,
                       rotor_sideband_db_c,
                       envelope_peak_hz_a, envelope_peak_hz_b,
                       envelope_peak_hz_c,
                       unbalance_percent, zscore_max,
                       predicted_class, model_confidence, model_version
                FROM telemetry_features
                WHERE motor_id = $1 AND status <> 'INRUSH_SUPPRESSED'
                ORDER BY recorded_at DESC
                LIMIT 1
                """,
                motor_id,
            )
        return dict(row) if row is not None else None

    # ------------------------------------------------------------------
    # Analytics trends fetch (steady-state window series)
    # ------------------------------------------------------------------
    async def fetch_telemetry_history(
        self, motor_id: str, hours: int, limit: int
    ) -> list[dict[str, Any]]:
        """Steady-state window history for one motor (Analytics trends).

        The newest ``limit`` non-inrush telemetry rows inside the last
        ``hours`` wall-clock hours, returned OLDEST-FIRST (chart-ready).
        INRUSH_SUPPRESSED rows are excluded — startup transients are not
        a health state (same exclusion as Section 14.2.2, the assessment
        read and the plant-wide read). Raises ValueError("No telemetry
        recorded ...") when the motor has no telemetry row at all (the
        route maps that to 404). Pure read path: one connection, two
        statements, no writes.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            known = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM telemetry_features WHERE motor_id = $1
                )
                """,
                motor_id,
            )
            if not known:
                raise ValueError(
                    f"No telemetry recorded for motor {motor_id!r}"
                )
            rows = await conn.fetch(
                """
                SELECT recorded_at, status, fundamental_hz,
                       rms_a, rms_b, rms_c,
                       thd_percent_a, thd_percent_b, thd_percent_c,
                       crest_factor_a, crest_factor_b, crest_factor_c,
                       rotor_sideband_db_a, rotor_sideband_db_b,
                       rotor_sideband_db_c,
                       unbalance_percent, zscore_max,
                       predicted_class, model_confidence, model_version
                FROM telemetry_features
                WHERE motor_id = $1 AND status <> 'INRUSH_SUPPRESSED'
                  AND recorded_at >= now() - make_interval(hours => $2)
                ORDER BY recorded_at DESC
                LIMIT $3
                """,
                motor_id, int(hours), int(limit),
            )
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------
    # Fault log read/episode path (Section 10 permanent retention)
    # ------------------------------------------------------------------
    async def get_latest_fault_episode(
        self, motor_id: str
    ) -> dict[str, Any] | None:
        """Newest fault_logs row of one motor (episode-dedup probe).

        Pure read: the ingestion path calls this before writing a fault
        row so an ONGOING episode (same taxonomy code, unresolved)
        never produces duplicate rows — one row per fault episode.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT taxonomy_code, detected_at, resolved_at
                FROM fault_logs
                WHERE motor_id = $1
                ORDER BY detected_at DESC
                LIMIT 1
                """,
                motor_id,
            )
        return dict(row) if row is not None else None

    async def fetch_fault_logs(
        self, motor_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """Newest fault log rows (Section 10: permanently retained).

        All motors, or one motor when ``motor_id`` is given, newest
        first, capped at ``limit``. ``spectral_evidence`` is JSONB:
        asyncpg has no jsonb codec installed, so the column arrives as
        a JSON string and is decoded here (str|dict tolerant for the
        test harness). Pure read path.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if motor_id is None:
                rows = await conn.fetch(
                    """
                    SELECT id, motor_id, detected_at, taxonomy_code,
                           urgency_stage, model_confidence, population_sigma,
                           absolute_threshold_breached, trend_confirmed,
                           four_signal_confirmed, physically_verified,
                           verified_by, spectral_evidence, resolved_at
                    FROM fault_logs
                    ORDER BY detected_at DESC
                    LIMIT $1
                    """,
                    int(limit),
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT id, motor_id, detected_at, taxonomy_code,
                           urgency_stage, model_confidence, population_sigma,
                           absolute_threshold_breached, trend_confirmed,
                           four_signal_confirmed, physically_verified,
                           verified_by, spectral_evidence, resolved_at
                    FROM fault_logs
                    WHERE motor_id = $1
                    ORDER BY detected_at DESC
                    LIMIT $2
                    """,
                    motor_id, int(limit),
                )
        out: list[dict[str, Any]] = []
        for row in rows:
            record = dict(row)
            evidence = record.get("spectral_evidence")
            if isinstance(evidence, str):
                try:
                    record["spectral_evidence"] = json.loads(evidence)
                except (TypeError, ValueError):
                    record["spectral_evidence"] = {}
            out.append(record)
        return out

    # ------------------------------------------------------------------
    # Deterministic RUL history fetch (Section 14.2)
    # ------------------------------------------------------------------
    async def fetch_health_index_history(
        self, motor_id: str, lookback_days: int
    ) -> HealthIndexHistory:
        """Read one motor's Health-Index time series for Section 14 RUL.

        INRUSH_SUPPRESSED rows are excluded at the SQL level — startup
        transients are not degradation (Sections 14.2.2 / 9). Returns:

          - first_recorded_at: earliest non-inrush row (cold-start clock
            origin, Section 14.3)
          - current_sample: most recent non-inrush row (current HI,
            Section 14.4.4)
          - window_samples: all non-inrush rows inside the last
            ``lookback_days`` wall-clock days, ascending (trend window,
            Section 14.4.1)

        Pure read path: one connection, three statements, no writes.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn:
            first_recorded_at = await conn.fetchval(
                """
                SELECT MIN(recorded_at) FROM telemetry_features
                WHERE motor_id = $1 AND status <> 'INRUSH_SUPPRESSED'
                """,
                motor_id,
            )
            last_row = await conn.fetchrow(
                """
                SELECT recorded_at, zscore_max FROM telemetry_features
                WHERE motor_id = $1 AND status <> 'INRUSH_SUPPRESSED'
                ORDER BY recorded_at DESC
                LIMIT 1
                """,
                motor_id,
            )
            window_rows = await conn.fetch(
                """
                SELECT recorded_at, zscore_max FROM telemetry_features
                WHERE motor_id = $1
                  AND status <> 'INRUSH_SUPPRESSED'
                  AND recorded_at >= now() - make_interval(days => $2)
                ORDER BY recorded_at ASC
                """,
                motor_id, int(lookback_days),
            )

        current_sample = (
            HealthIndexSample(
                recorded_at=last_row["recorded_at"],
                health_index_sigma=float(last_row["zscore_max"]),
            )
            if last_row is not None
            else None
        )
        window_samples = [
            HealthIndexSample(
                recorded_at=r["recorded_at"],
                health_index_sigma=float(r["zscore_max"]),
            )
            for r in window_rows
        ]
        return HealthIndexHistory(
            first_recorded_at=first_recorded_at,
            current_sample=current_sample,
            window_samples=window_samples,
        )
