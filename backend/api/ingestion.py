"""Sigmo V2 — Ingestion service: RAM window accumulation, DSP dispatch,
gating, adaptive persistence, and offline store-and-forward buffering.

Data flow (SIGMO_RULES.md Sections 4.1, 9, 10):

  ESP32 frame (2048/4096 samples) --> per-motor RAM accumulator
    --> when 32,768 samples/phase collected --> DSP pipeline (RAM only)
    --> gating (absolute thresholds + baseline z-score)
    --> repository.log_telemetry (adaptive cadence, Supabase)
    --> on DB outage: features enter a bounded in-RAM offline buffer,
        flushed automatically on the next successful persistence cycle.

Raw sample arrays are discarded the moment feature extraction completes.
Only feature rows (~200 bytes) are ever buffered or persisted.
"""
from __future__ import annotations

import asyncio
import logging

import asyncpg
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from ..analysis.gating import GatingResult, evaluate_window
from ..analysis.repair_protocols_am import (
    AMHARIC_PROTOCOLS,
    TOOLS_AND_SPARES,
    exact_taxonomy_code,
)
from ..analysis.rul import RulStatus, assess_rul
from ..analysis.staging import (
    classify_stage,
    energy_penalty_monthly_etb,
    executive_decision_directive,
    financial_risk_exposure_etb,
    plant_health_index,
    plant_health_index_from_scores,
    motor_health_percent,
    assessed_health_percent,
)
from ..config import DSP
from ..db.repository import SigmoRepository
from ..dsp.pipeline import ThreePhaseFeatures, analyze_three_phase
from ..ml.features import extract_window_features
from ..ml.inference import ModelRuntime
from .schemas import (
    AlertEntry,
    AlertsPayload,
    CriticalAssetRiskSummary,
    ExecutiveViewPayload,
    FaultAssessment,
    FaultLogEntry,
    FaultLogsPayload,
    FaultUrgencyStage,
    FinancialAmount,
    MccPanelLocation,
    MotorAssessment,
    MotorSnapshot,
    MotorSpecifications,
    RequiredToolsAndSpares,
    RulAssessment,
    RulWindow,
    SpectralEvidenceData,
    TechnicianViewPayload,
    TelemetryFrame,
    TrendPoint,
    TrendsPayload,
    TrendsWindow,
    WindowAnalysis,
    MotorsOverviewPayload,
    MotorOverviewEntry,
)

logger = logging.getLogger("sigmo.ingestion")

OFFLINE_BUFFER_MAX_ROWS = 5_000  # ~1 MB of feature rows; bounded by design


@dataclass
class _BufferedRow:
    """A feature row awaiting flush after a database outage."""

    motor_id: str
    features: ThreePhaseFeatures
    status: str
    zscore_max: float
    recorded_at: datetime
    predicted_class: str | None = None
    model_confidence: float | None = None
    model_version: str | None = None


@dataclass
class _LiveVerdict:
    """Freshest analyzed steady-state window for one motor (RAM only).

    Survives database outages so the assessment endpoint can always
    serve the latest verdict; superseded by the next analyzed window.
    """

    analyzed_at: datetime
    status: str
    dominant_fault: str
    confidence: float
    probabilities: dict[str, float]
    temporal_aggregation: dict[str, object]
    model_version: str
    snapshot: MotorSnapshot
    fft_peak_hz: list[float]
    sideband_50hz_db: list[float]
    crest_factor_max: float


def _analyze_window(
    i_a: np.ndarray, i_b: np.ndarray, i_c: np.ndarray, fs: float
) -> tuple[ThreePhaseFeatures, dict[str, float]]:
    """One threaded pass: gating DSP + the 117-feature v6c vector.

    Both extractions read the same RAM-local raw window; the returned
    feature dict feeds ModelRuntime.classify_temporal verbatim (NaN
    features are legal — XGBoost-native handling).
    """
    features = analyze_three_phase(i_a, i_b, i_c, fs)
    cur = np.column_stack([i_a, i_b, i_c])
    ml_features = extract_window_features(cur, fs)
    return features, ml_features


class _WindowAccumulator:
    """Per-motor RAM accumulator building 32,768-sample analysis windows.

    Frames must arrive with increasing sequence numbers; a sequence gap
    (dropped frame) resets the window to prevent stitching discontinuous
    waveforms, which would corrupt the spectrum.
    """

    def __init__(self) -> None:
        self._a: list[np.ndarray] = []
        self._b: list[np.ndarray] = []
        self._c: list[np.ndarray] = []
        self._count = 0
        self._last_sequence: int | None = None

    @property
    def fill_percent(self) -> float:
        return min(100.0, 100.0 * self._count / DSP.analysis_window_samples)

    def add(self, frame: TelemetryFrame) -> bool:
        """Append a frame. Returns True when a full window is ready."""
        if self._last_sequence is not None and frame.sequence != self._last_sequence + 1:
            logger.warning(
                "Sequence gap on motor %s (%d -> %d); resetting window",
                frame.motor_id, self._last_sequence, frame.sequence,
            )
            self.reset()
        self._last_sequence = frame.sequence
        self._a.append(np.asarray(frame.i_a, dtype=np.float64))
        self._b.append(np.asarray(frame.i_b, dtype=np.float64))
        self._c.append(np.asarray(frame.i_c, dtype=np.float64))
        self._count += len(frame.i_a)
        return self._count >= DSP.analysis_window_samples

    def drain(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the completed window and clear all raw data from RAM."""
        n = DSP.analysis_window_samples
        i_a = np.concatenate(self._a)[:n]
        i_b = np.concatenate(self._b)[:n]
        i_c = np.concatenate(self._c)[:n]
        self.reset()
        return i_a, i_b, i_c

    def reset(self) -> None:
        self._a.clear()
        self._b.clear()
        self._c.clear()
        self._count = 0


class IngestionService:
    """Orchestrates accumulation, analysis, gating, and persistence."""

    def __init__(self, repository: SigmoRepository) -> None:
        self._repo = repository
        self._accumulators: dict[str, _WindowAccumulator] = {}
        self._baseline_cache: dict[str, dict[str, tuple[float, float, float]]] = {}
        self._offline_buffer: deque[_BufferedRow] = deque(maxlen=OFFLINE_BUFFER_MAX_ROWS)
        self._lock = asyncio.Lock()
        self.database_connected = False
        # v6c production runtime (Phase 4): loads the registry's ACTIVE
        # version (v6.20260925.045900). Degraded-tolerant: if no bundle
        # can be loaded, ingestion/gating/RUL continue without verdicts
        # and every ML field is honestly null.
        self._runtime: ModelRuntime | None = None
        self._live_verdicts: dict[str, _LiveVerdict] = {}
        # Commissioning gate (Section 11 Cold Start): motors must be
        # registered in the asset registry before telemetry is accepted
        # (telemetry_features.motor_id has a foreign key to motors).
        # Positive lookups only — an absent motor is re-checked against
        # the database so a mid-run registration takes effect at once.
        self._registered_motors: set[str] = set()
        try:
            runtime = ModelRuntime()
            runtime.load()
            self._runtime = runtime
            logger.info("ML runtime loaded: %s", runtime.version)
        except Exception as exc:
            logger.error(
                "ML runtime unavailable (%s); running in degraded mode "
                "without dominant-fault verdicts.", exc,
            )

    @property
    def model_version(self) -> str | None:
        return self._runtime.version if self._runtime is not None else None

    @property
    def offline_buffer_depth(self) -> int:
        return len(self._offline_buffer)

    @property
    def motors_streaming(self) -> int:
        return len(self._accumulators)

    async def _baseline_for(self, motor_id: str) -> dict[str, tuple[float, float, float]]:
        """Fetch and cache the personalized baseline; empty dict when the
        motor has none (COMMISSIONING / POPULATION_TRACKED states) or the
        database is unreachable (gating then relies on absolute thresholds,
        which is the correct degraded behaviour per Section 11.3)."""
        if motor_id in self._baseline_cache:
            return self._baseline_cache[motor_id]
        try:
            baseline = await self._repo.get_motor_baseline(motor_id)
            self.database_connected = True
        except Exception as exc:  # outage tolerance, Section 9
            logger.error("Baseline fetch failed for %s: %s", motor_id, exc)
            self.database_connected = False
            return {}
        self._baseline_cache[motor_id] = baseline
        return baseline

    async def _persist(
        self,
        motor_id: str,
        features: ThreePhaseFeatures,
        gating: GatingResult,
        verdict: dict[str, object] | None = None,
    ) -> tuple[bool, bool]:
        """Persist a feature row; on failure, buffer it in RAM.

        Returns (persisted, buffered_offline).
        """
        now = datetime.now(timezone.utc)
        predicted_class = (
            str(verdict["predicted_class"]) if verdict is not None else None
        )
        model_confidence = (
            float(verdict["confidence"]) if verdict is not None else None
        )
        model_version = (
            str(verdict["model_version"]) if verdict is not None else None
        )
        try:
            await self._flush_offline_buffer()
            written = await self._repo.log_telemetry(
                motor_id, features, gating.status, gating.zscore_max,
                predicted_class=predicted_class,
                model_confidence=model_confidence,
                model_version=model_version,
            )
            self.database_connected = True
            return written, False
        except Exception as exc:
            self.database_connected = False
            self._offline_buffer.append(
                _BufferedRow(
                    motor_id, features, gating.status, gating.zscore_max, now,
                    predicted_class, model_confidence, model_version,
                )
            )
            logger.error(
                "Supabase unreachable (%s); row buffered (depth=%d/%d)",
                exc, len(self._offline_buffer), OFFLINE_BUFFER_MAX_ROWS,
            )
            return False, True

    async def _flush_offline_buffer(self) -> None:
        """Replay buffered rows oldest-first with original timestamps.

        A row for a motor that was never registered (foreign-key
        violation — telemetry accepted in good faith during a database
        outage) is PERMANENTLY invalid: it is dropped with an explicit
        error naming the motor, never silently and never blocking the
        remaining rows. Any other failure (network, timeout) stops the
        replay so rows keep their order and retry on the next persist.
        """
        while self._offline_buffer:
            row = self._offline_buffer[0]
            try:
                await self._repo.log_telemetry(
                    row.motor_id,
                    row.features,
                    row.status,
                    row.zscore_max,
                    recorded_at=row.recorded_at,
                    bypass_cadence=True,
                    predicted_class=row.predicted_class,
                    model_confidence=row.model_confidence,
                    model_version=row.model_version,
                )
            except asyncpg.ForeignKeyViolationError:
                logger.error(
                    "Dropping buffered row for unregistered motor %s "
                    "(recorded %s) — commission the motor before "
                    "reconnecting its telemetry node.",
                    row.motor_id, row.recorded_at.isoformat(),
                )
                self._offline_buffer.popleft()
                continue
            self._offline_buffer.popleft()

    async def _motor_known(self, motor_id: str) -> bool:
        """True when the motor is commissioned in the asset registry.

        Fail-open during database outages (Section 9): telemetry is
        never refused because the grid/network is down — rows buffer,
        and permanently-unregistrable rows are dropped with an explicit
        error at flush time instead of blocking the pipeline.
        """
        if motor_id in self._registered_motors:
            return True
        try:
            asset = await self._repo.get_motor_asset(motor_id)
        except Exception:
            return True  # database unreachable — fail-open (Section 9)
        if asset is not None:
            self._registered_motors.add(motor_id)
            return True
        return False

    async def _log_fault_episode(
        self,
        motor_id: str,
        features: ThreePhaseFeatures,
        gating: GatingResult,
        verdict: dict[str, object],
    ) -> None:
        """Persist one fault episode row (Section 10 permanent retention).

        Episode trigger: a non-HEALTHY classifier verdict on a
        steady-state window (gate evidence — including a HEALTHY gate —
        is recorded verbatim in the Section 11.6 signals, and
        four_signal_confirmed stays the anti-false-positive gate).
        Episode dedup: the
        newest existing row of this motor is probed first — an ONGOING
        episode (same taxonomy code, unresolved) never duplicates; only
        a NEW episode (different code, or first ever) is written. The row records the four Section 11.6
        anti-false-positive signals verbatim; four_signal_confirmed is
        computed by the database. Fail-soft by design: a fault-log
        write failure (e.g. mid-outage) is logged and swallowed —
        telemetry ingestion must never be blocked by the audit trail.
        """
        predicted = str(verdict["predicted_class"])
        confidence = float(verdict["confidence"])
        zscore = float(gating.zscore_max)
        absolute_breach = bool(gating.breaches)
        aggregation = verdict.get("temporal_aggregation") or {}
        trend_confirmed = bool(
            aggregation
            and aggregation.get("stable")
            and aggregation.get("aggregated_class") == predicted
        )
        stage = classify_stage(
            zscore_max=zscore,
            gating_status=gating.status,
            predicted_class=predicted,
            rul_days_numeric=None,
            rul_status=None,
        )
        taxonomy_code = exact_taxonomy_code(stage.stage, predicted)
        try:
            latest = await self._repo.get_latest_fault_episode(motor_id)
            if latest is not None and latest["taxonomy_code"] == taxonomy_code:
                return  # ongoing episode — no duplicate rows
            await self._repo.log_fault(
                motor_id=motor_id,
                taxonomy_code=taxonomy_code,
                urgency_stage=stage.stage,
                model_confidence=confidence,
                population_sigma=zscore,
                absolute_threshold_breached=absolute_breach,
                trend_confirmed=trend_confirmed,
                spectral_evidence={
                    "fundamental_hz": features.mean_fundamental_hz,
                    "thd_percent_max": max(
                        features.phase_a.thd_percent,
                        features.phase_b.thd_percent,
                        features.phase_c.thd_percent,
                    ),
                    "current_unbalance_percent": (
                        features.current_unbalance_percent
                    ),
                    "crest_factor_max": max(
                        features.phase_a.crest_factor,
                        features.phase_b.crest_factor,
                        features.phase_c.crest_factor,
                    ),
                    "rotor_sideband_db_max": max(
                        features.phase_a.rotor_sideband_ratio_db,
                        features.phase_b.rotor_sideband_ratio_db,
                        features.phase_c.rotor_sideband_ratio_db,
                    ),
                    "envelope_peak_hz": [
                        features.phase_a.envelope_peak_hz,
                        features.phase_b.envelope_peak_hz,
                        features.phase_c.envelope_peak_hz,
                    ],
                    "rotor_sideband_db": [
                        features.phase_a.rotor_sideband_ratio_db,
                        features.phase_b.rotor_sideband_ratio_db,
                        features.phase_c.rotor_sideband_ratio_db,
                    ],
                    "absolute_threshold_breaches": list(gating.breaches),
                    "zscore_max": zscore,
                    "model_version": str(verdict["model_version"]),
                },
            )
            logger.info(
                "Fault episode logged motor=%s code=%s stage=%d "
                "conf=%.2f sigma=%.2f abs=%s trend=%s",
                motor_id, taxonomy_code, stage.stage, confidence,
                zscore, absolute_breach, trend_confirmed,
            )
        except Exception as exc:
            logger.error(
                "Fault episode log failed for motor %s (%s) — ingestion "
                "continues; the audit row for this episode is lost.",
                motor_id, exc,
            )

    async def ingest(self, frame: TelemetryFrame) -> tuple[float, WindowAnalysis | None]:
        """Process one frame; returns (window_fill_percent, analysis or None)."""
        # Commissioning gate: reject the FIRST frame of an unregistered
        # motor with an actionable 422 — never accumulate 15 frames and
        # fail on the 16th. Fail-open during database outages (Section 9).
        if not await self._motor_known(frame.motor_id):
            raise ValueError(
                f"Motor {frame.motor_id!r} is not registered — commission "
                "it first via PUT /api/v2/admin/motor-assets/"
                f"{frame.motor_id} (SIGMO_RULES.md Section 11 Cold Start: "
                "physical identity before monitoring)."
            )

        async with self._lock:
            acc = self._accumulators.setdefault(frame.motor_id, _WindowAccumulator())
            window_ready = acc.add(frame)
            fill = acc.fill_percent
            if not window_ready:
                return fill, None
            i_a, i_b, i_c = acc.drain()  # raw arrays leave the accumulator

        # Heavy DSP off the event loop; raw arrays are RAM-local to this
        # call. One threaded pass produces BOTH the gating features and
        # the 117-feature v6c vector from the same raw window.
        features, ml_features = await asyncio.to_thread(
            _analyze_window, i_a, i_b, i_c, frame.sampling_rate_hz
        )
        del i_a, i_b, i_c  # explicit end-of-life for raw waveforms (Section 10)

        baseline = await self._baseline_for(frame.motor_id)
        gating = evaluate_window(features, baseline)

        # v6c verdict (Phase 4): classify steady-state windows only —
        # the model is trained on steady-state data, so INRUSH_SUPPRESSED
        # startup transients are never classified (same exclusion the
        # Section 14.2.2 RUL protocol applies to the HI series).
        verdict: dict[str, object] | None = None
        if self._runtime is not None and gating.status != "INRUSH_SUPPRESSED":
            verdict = self._runtime.classify_temporal(frame.motor_id, ml_features)

        persisted, buffered = await self._persist(
            frame.motor_id, features, gating, verdict
        )

        analysis = WindowAnalysis(
            motor_id=frame.motor_id,
            window_completed_at=datetime.now(timezone.utc),
            status=gating.status,  # type: ignore[arg-type]
            fundamental_hz=features.mean_fundamental_hz,
            rms_a=features.phase_a.rms_a,
            rms_b=features.phase_b.rms_a,
            rms_c=features.phase_c.rms_a,
            thd_percent_max=max(
                features.phase_a.thd_percent,
                features.phase_b.thd_percent,
                features.phase_c.thd_percent,
            ),
            current_unbalance_percent=features.current_unbalance_percent,
            crest_factor_max=max(
                features.phase_a.crest_factor,
                features.phase_b.crest_factor,
                features.phase_c.crest_factor,
            ),
            rotor_sideband_db_max=max(
                features.phase_a.rotor_sideband_ratio_db,
                features.phase_b.rotor_sideband_ratio_db,
                features.phase_c.rotor_sideband_ratio_db,
            ),
            zscore_max=gating.zscore_max,
            absolute_threshold_breaches=list(gating.breaches),
            persisted=persisted,
            buffered_offline=buffered,
            dominant_fault=(
                str(verdict["predicted_class"]) if verdict is not None else None
            ),
            model_confidence=(
                float(verdict["confidence"]) if verdict is not None else None
            ),
            probabilities=(
                dict(verdict["probabilities"]) if verdict is not None else None
            ),
            temporal_aggregation=(
                dict(verdict["temporal_aggregation"])
                if verdict is not None else None
            ),
            model_version=(
                str(verdict["model_version"]) if verdict is not None else None
            ),
        )
        if verdict is not None:
            self._live_verdicts[frame.motor_id] = _LiveVerdict(
                analyzed_at=analysis.window_completed_at,
                status=gating.status,
                dominant_fault=analysis.dominant_fault or "",
                confidence=analysis.model_confidence or 0.0,
                probabilities=analysis.probabilities or {},
                temporal_aggregation=analysis.temporal_aggregation or {},
                model_version=analysis.model_version or "",
                snapshot=MotorSnapshot(
                    recorded_at=analysis.window_completed_at,
                    status=gating.status,  # type: ignore[arg-type]
                    fundamental_hz=analysis.fundamental_hz,
                    rms_a=analysis.rms_a,
                    rms_b=analysis.rms_b,
                    rms_c=analysis.rms_c,
                    thd_percent_max=analysis.thd_percent_max,
                    current_unbalance_percent=analysis.current_unbalance_percent,
                    crest_factor_max=analysis.crest_factor_max,
                    rotor_sideband_db_max=analysis.rotor_sideband_db_max,
                    zscore_max=analysis.zscore_max,
                ),
                fft_peak_hz=[
                    features.phase_a.envelope_peak_hz,
                    features.phase_b.envelope_peak_hz,
                    features.phase_c.envelope_peak_hz,
                ],
                sideband_50hz_db=[
                    features.phase_a.rotor_sideband_ratio_db,
                    features.phase_b.rotor_sideband_ratio_db,
                    features.phase_c.rotor_sideband_ratio_db,
                ],
                crest_factor_max=analysis.crest_factor_max,
            )
            # Fault episode audit trail (Section 10): one row per NEW
            # fault episode — any non-HEALTHY classifier verdict opens
            # the episode. Mechanical faults (bearing outer race, rotor
            # bars) do NOT breach the electrical gates, so requiring
            # gate ANOMALY here left real Stage 2 detections invisible
            # in Fault Logs while the dashboard alerted on them. The
            # Section 11.6 multi-signal evidence is still recorded
            # verbatim on every row, and four_signal_confirmed (the
            # anti-false-positive gate) still requires ALL four signals
            # — an unconfirmed episode is visible but honestly flagged.
            if analysis.dominant_fault not in (None, "HEALTHY"):
                await self._log_fault_episode(
                    frame.motor_id, features, gating, verdict
                )
        logger.info(
            "Window analyzed motor=%s status=%s fund=%.2fHz thd=%.2f%% unb=%.2f%%"
            " verdict=%s conf=%.2f",
            frame.motor_id, gating.status, features.mean_fundamental_hz,
            analysis.thd_percent_max, features.current_unbalance_percent,
            analysis.dominant_fault,
            analysis.model_confidence if analysis.model_confidence is not None else -1.0,
        )
        return 100.0, analysis

    # ------------------------------------------------------------------
    # Comprehensive motor assessment (Phase 4, SIGMO_RULES.md 6/14)
    # ------------------------------------------------------------------
    async def assessment(self, motor_id: str) -> MotorAssessment:
        """Assemble the complete per-motor payload in one read path.

        Precedence rules (honest freshness):
          snapshot / fault_assessment — the LIVE runtime verdict of the
            latest analyzed steady-state window when this service has
            seen the motor (fresher than the 30-min healthy persistence
            cadence); otherwise the latest persisted Supabase row.
          rul — ALWAYS the deterministic Section 14 protocol over the
            stored Supabase HI series (pure read, Section 14.7.2); null
            with an explicit reason only when Supabase is unreachable.
        """
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        generated_at = _dt.now(_tz.utc)
        live = self._live_verdicts.get(motor_id)

        snapshot: MotorSnapshot | None = None
        snapshot_source: str | None = None
        fault: FaultAssessment | None = None
        db_row: dict[str, object] | None = None
        db_ok = True
        try:
            db_row = await self._repo.fetch_latest_snapshot(motor_id)
            self.database_connected = True
        except Exception as exc:
            db_ok = False
            self.database_connected = False
            logger.error("Snapshot fetch failed for %s: %s", motor_id, exc)

        # Unknown motor: no live verdict in RAM and no persisted row (when
        # the database could be queried). 404 — there is nothing honest to
        # report, not even an RUL status (the Section 14 protocol needs a
        # stored HI series). Checked BEFORE the RUL read on purpose.
        if live is None and db_row is None:
            raise ValueError(
                f"No telemetry recorded for motor {motor_id!r} yet — send "
                "frames to POST /api/v2/telemetry first."
            )

        if live is not None:
            snapshot = live.snapshot
            snapshot_source = "live_runtime"
            fault = FaultAssessment(
                dominant_fault=live.dominant_fault,
                confidence=live.confidence,
                is_fault=live.dominant_fault != "HEALTHY",
                probabilities=live.probabilities,
                temporal_aggregation=live.temporal_aggregation,
                model_version=live.model_version,
                classified_at=live.analyzed_at,
                source="live_runtime",
            )
        elif db_row is not None:
            snapshot = MotorSnapshot(
                recorded_at=db_row["recorded_at"],
                status=db_row["status"],
                fundamental_hz=float(db_row["fundamental_hz"]),
                rms_a=float(db_row["rms_a"]),
                rms_b=float(db_row["rms_b"]),
                rms_c=float(db_row["rms_c"]),
                thd_percent_max=max(
                    float(db_row["thd_percent_a"]),
                    float(db_row["thd_percent_b"]),
                    float(db_row["thd_percent_c"]),
                ),
                current_unbalance_percent=float(db_row["unbalance_percent"]),
                crest_factor_max=max(
                    float(db_row["crest_factor_a"]),
                    float(db_row["crest_factor_b"]),
                    float(db_row["crest_factor_c"]),
                ),
                rotor_sideband_db_max=max(
                    float(db_row["rotor_sideband_db_a"]),
                    float(db_row["rotor_sideband_db_b"]),
                    float(db_row["rotor_sideband_db_c"]),
                ),
                zscore_max=float(db_row["zscore_max"]),
            )
            snapshot_source = "supabase"
            if db_row.get("predicted_class") is not None:
                fault = FaultAssessment(
                    dominant_fault=str(db_row["predicted_class"]),
                    confidence=float(db_row["model_confidence"] or 0.0),
                    is_fault=str(db_row["predicted_class"]) != "HEALTHY",
                    probabilities={},  # not persisted by cadence design
                    temporal_aggregation=None,
                    model_version=str(db_row["model_version"] or ""),
                    classified_at=db_row["recorded_at"],
                    source="supabase",
                )

        rul: RulAssessment | None = None
        rul_reason: str | None = None
        if db_ok:
            try:
                estimate = await assess_rul(self._repo, motor_id)
                rul = RulAssessment(
                    motor_id=motor_id,
                    rul_days_estimate=estimate.rul_days_estimate,
                    rul_status=(
                        estimate.rul_status.value
                        if isinstance(estimate.rul_status, RulStatus)
                        else str(estimate.rul_status)
                    ),
                    slope_sigma_per_day=estimate.slope_sigma_per_day,
                    current_health_index_sigma=estimate.current_health_index_sigma,
                    trend_window_days=estimate.trend_window_days,
                    rul_days_lower=estimate.rul_days_lower,
                    rul_days_upper=estimate.rul_days_upper,
                    trend_points=estimate.trend_points,
                    last_window_at=estimate.last_window_at,
                )
            except Exception as exc:
                logger.error("RUL assessment failed for %s: %s", motor_id, exc)
                rul_reason = f"RUL read path failed: {exc}"
        else:
            rul_reason = (
                "Supabase unreachable — the deterministic RUL protocol reads "
                "stored Health-Index history (SIGMO_RULES.md Section 14.2) "
                "and cannot be computed from the live buffer alone."
            )

        return MotorAssessment(
            motor_id=motor_id,
            model_version=self.model_version or "unavailable",
            generated_at=generated_at,
            database_connected=db_ok,
            snapshot=snapshot,
            snapshot_source=snapshot_source,
            fault_assessment=fault,
            rul=rul,
            rul_unavailable_reason=rul_reason,
        )

    # ------------------------------------------------------------------
    # Dual-View UX assemblies (Phase 5, SIGMO_RULES.md 6.a / 6.b)
    # ------------------------------------------------------------------
    def _latest_motor_state(
        self,
        live: _LiveVerdict | None,
        db_row: dict[str, object] | None,
    ) -> dict[str, object]:
        """Merge live-runtime and Supabase state for one motor, freshest
        wins (the live verdict supersedes the persisted row when this
        service analyzed a newer steady-state window)."""
        if live is None and db_row is None:
            raise ValueError("no telemetry")
        if live is not None and (
            db_row is None
            or live.analyzed_at >= db_row["recorded_at"]
        ):
            return {
                "recorded_at": live.analyzed_at,
                "status": live.status,
                "zscore_max": live.snapshot.zscore_max,
                "predicted_class": live.dominant_fault,
                "model_confidence": live.confidence,
                "thd_percent_max": live.snapshot.thd_percent_max,
                "unbalance_percent": (
                    live.snapshot.current_unbalance_percent
                ),
                "fundamental_hz": live.snapshot.fundamental_hz,
                "fft_peak_hz": live.fft_peak_hz,
                "sideband_50hz_db": live.sideband_50hz_db,
                "crest_factor_max": live.snapshot.crest_factor_max,
                "source": "live_runtime",
            }
        # persisted row (service restart or motor idle)
        thd_max = max(
            float(db_row["thd_percent_a"]),   # type: ignore[index]
            float(db_row["thd_percent_b"]),   # type: ignore[index]
            float(db_row["thd_percent_c"]),   # type: ignore[index]
        )
        return {
            "recorded_at": db_row["recorded_at"],
            "status": db_row["status"],
            "zscore_max": float(db_row["zscore_max"]),
            "predicted_class": db_row.get("predicted_class"),
            "model_confidence": db_row.get("model_confidence"),
            "thd_percent_max": thd_max,
            "unbalance_percent": float(db_row["unbalance_percent"]),
            "fundamental_hz": float(db_row["fundamental_hz"]),
            "fft_peak_hz": [
                float(db_row["envelope_peak_hz_a"]),  # type: ignore[index]
                float(db_row["envelope_peak_hz_b"]),  # type: ignore[index]
                float(db_row["envelope_peak_hz_c"]),  # type: ignore[index]
            ],
            "sideband_50hz_db": [
                float(db_row["rotor_sideband_db_a"]),  # type: ignore[index]
                float(db_row["rotor_sideband_db_b"]),  # type: ignore[index]
                float(db_row["rotor_sideband_db_c"]),  # type: ignore[index]
            ],
            "crest_factor_max": max(
                float(db_row["crest_factor_a"]),  # type: ignore[index]
                float(db_row["crest_factor_b"]),  # type: ignore[index]
                float(db_row["crest_factor_c"]),  # type: ignore[index]
            ),
            "source": "supabase",
        }

    async def _stage_and_rul(
        self, motor_id: str, state: dict[str, object]
    ) -> tuple[object, object]:
        """Deterministic Section 14 RUL + Stage 1-4 for one motor."""
        estimate = None
        try:
            estimate = await assess_rul(self._repo, motor_id)
        except Exception as exc:
            logger.error("RUL read failed for %s: %s", motor_id, exc)
        numeric = estimate.rul_days_estimate if (
            estimate is not None and estimate.is_numeric
        ) else None
        stage = classify_stage(
            zscore_max=float(state["zscore_max"]),
            gating_status=str(state["status"]),
            predicted_class=(
                str(state["predicted_class"])
                if state["predicted_class"] is not None else None
            ),
            rul_days_numeric=numeric,
            rul_status=(
                estimate.rul_status.value
                if estimate is not None else None
            ),
        )
        return stage, estimate

    async def executive_view(self) -> ExecutiveViewPayload:
        """Section 6.a — the complete Executive payload, one response.

        Aggregates every motor with telemetry (live runtime state
        supersedes the persisted row when fresher), classifies each via
        the deterministic stage engine, and computes the financial
        figures from customer-configured rates only.
        """
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        generated_at = _dt.now(_tz.utc)
        db_ok = True
        try:
            db_rows = {
                r["motor_id"]: r
                for r in await self._repo.fetch_all_latest_telemetry()
            }
            assets = {
                a["motor_id"]: a
                for a in await self._repo.list_motor_assets()
            }
            config = await self._repo.get_plant_config()
            self.database_connected = True
        except Exception as exc:
            db_ok = False
            self.database_connected = False
            db_rows, assets, config = {}, {}, {}
            logger.error("Executive read path failed: %s", exc)

        # Union of motors seen live or persisted
        motor_ids = set(db_rows) | set(self._live_verdicts)
        states: dict[str, dict[str, object]] = {}
        for mid in sorted(motor_ids):
            try:
                states[mid] = self._latest_motor_state(
                    self._live_verdicts.get(mid), db_rows.get(mid)
                )
            except ValueError:
                continue

        # Per-motor stage + RUL (RUL needs the DB; live-only outage mode
        # reports the reason honestly instead)
        stage_by_motor: dict[str, object] = {}
        rul_by_motor: dict[str, object] = {}
        if db_ok:
            for mid in states:
                stage_by_motor[mid], rul_by_motor[mid] = (
                    await self._stage_and_rul(mid, states[mid])
                )
        else:
            for mid in states:
                stage_by_motor[mid] = classify_stage(
                    zscore_max=float(states[mid]["zscore_max"]),
                    gating_status=str(states[mid]["status"]),
                    predicted_class=(
                        str(states[mid]["predicted_class"])
                        if states[mid]["predicted_class"] is not None
                        else None
                    ),
                    rul_days_numeric=None,
                    rul_status=None,
                )

        # plant_health_index — mean of per-motor OPERATOR scores: each
        # motor is scored with its stage/verdict context, so an active
        # Stage 2+ fault pulls the plant index down exactly as it pulls
        # the Machine Status Grid and Decision Engine down (one scoring
        # rule everywhere; the RUL engine keeps the pure z-anchored
        # series by design).
        motor_scores = [
            assessed_health_percent(
                float(st["zscore_max"]),
                stage=stage_by_motor[mid].stage,
                predicted_class=st["predicted_class"],
            )
            for mid, st in states.items()
        ]
        phi = plant_health_index_from_scores(motor_scores)
        phi_note = (
            None if phi is not None
            else "No motor telemetry yet — the plant index needs at "
                 "least one analyzed window."
        )

        # critical_asset_risk_summary
        healthy = sum(1 for st in stage_by_motor.values() if st.stage == 1)
        stage2 = sum(1 for st in stage_by_motor.values() if st.stage == 2)
        stage34 = sum(
            1 for st in stage_by_motor.values() if st.stage >= 3
        )
        summary = CriticalAssetRiskSummary(
            healthy=healthy,
            stage_2_moderate_risk=stage2,
            stage_3_4_critical_failure_risk=stage34,
            total_monitored=len(stage_by_motor),
        )

        # financial_risk_exposure_etb (customer-configured rates only)
        default_cost = config.get("default_downtime_cost_etb_per_hour")
        pairs: list[tuple[int, float]] = []
        unconfigured: list[str] = []
        for mid, st in stage_by_motor.items():
            if st.stage < 2:
                continue
            cost = assets.get(mid, {}).get("downtime_cost_etb_per_hour")
            if cost is None:
                cost = (
                    default_cost["value"] if default_cost else None
                )
            if cost is None:
                unconfigured.append(mid)
            else:
                pairs.append((st.stage, float(cost)))
        if not pairs and unconfigured:
            risk = FinancialAmount(
                value_etb=None, status="rate_not_configured",
                basis="exposure = sum(expected_outage_hours(stage) x "
                      "downtime cost/hour); no rate configured",
                unconfigured_items=unconfigured,
            )
        elif pairs and unconfigured:
            risk = FinancialAmount(
                value_etb=financial_risk_exposure_etb(pairs),
                status="partial_rates_configured",
                basis="exposure = sum(expected_outage_hours(stage) x "
                      "downtime cost/hour); rates present for "
                      f"{len(pairs)} of {len(pairs) + len(unconfigured)} "
                      "at-risk motors",
                unconfigured_items=unconfigured,
            )
        else:
            risk = FinancialAmount(
                value_etb=financial_risk_exposure_etb(pairs),
                status="computed" if pairs else "no_active_risk",
                basis="exposure = sum(expected_outage_hours(stage) x "
                      "downtime cost/hour) over Stage >= 2 motors",
            )

        # energy_efficiency_loss_penalty_etb (monthly, per motor summed)
        tariff = config.get("energy_tariff_etb_per_kwh")
        pf = config.get("plant_power_factor")
        op_hours = config.get("operating_hours_per_month")
        if tariff is None:
            energy = FinancialAmount(
                value_etb=None, status="rate_not_configured",
                basis="monthly loss = tariff x hours x nameplate_kw x "
                      "(THD^2 + unbalance derating + PF term); the ETB "
                      "energy tariff is not configured "
                      "(plant_config key energy_tariff_etb_per_kwh)",
            )
        else:
            no_nameplate: list[str] = []
            total_etb = 0.0
            for mid, st in states.items():
                kw = assets.get(mid, {}).get("nameplate_kw")
                if kw is None:
                    no_nameplate.append(mid)
                    continue
                total_etb += energy_penalty_monthly_etb(
                    nameplate_kw=float(kw),
                    thd_percent_max=float(st["thd_percent_max"]),
                    unbalance_percent=float(st["unbalance_percent"]),
                    tariff_etb_per_kwh=tariff["value"],
                    operating_hours_per_month=(
                        op_hours["value"] if op_hours else None
                    ),
                    plant_power_factor=(
                        pf["value"] if pf else None
                    ),
                )
            energy = FinancialAmount(
                value_etb=round(total_etb, 2),
                status="computed" if not no_nameplate
                else "partial_rates_configured",
                basis="monthly loss = tariff x hours x nameplate_kw x "
                      "(THD^2 + 0.01 x max(0, unbalance - 1%)"
                      + (" + (1/pf - 1)" if pf else "")
                      + "); nameplate basis",
                unconfigured_items=no_nameplate,
            )

        # rul_days_estimate — the binding (minimum numeric) window
        numeric_ruls = [
            (mid, est) for mid, est in rul_by_motor.items()
            if est is not None and est.is_numeric
        ]
        if numeric_ruls:
            binding_mid, binding = min(
                numeric_ruls,
                key=lambda me: int(me[1].rul_days_estimate),
            )
            rul_window = RulWindow(
                estimate=int(binding.rul_days_estimate),
                lower=binding.rul_days_lower,
                upper=binding.rul_days_upper,
                status=binding.rul_status.value,
                binding_motor_id=binding_mid,
            )
        else:
            precedence = [
                "CRITICAL_THRESHOLD_REACHED", "DEGRADING", "STABLE",
                "CALCULATING_BUILDING_TREND",
                "CALCULATING_INSUFFICIENT_DATA",
            ]
            worst_mid, worst_est = None, None
            for want in precedence:
                for mid, est in rul_by_motor.items():
                    if est is not None and est.rul_status.value == want:
                        worst_mid, worst_est = mid, est
                        break
                if worst_est is not None:
                    break
            if worst_est is not None:
                rul_window = RulWindow(
                    estimate=worst_est.rul_days_estimate,
                    lower=None, upper=None,
                    status=worst_est.rul_status.value,
                    binding_motor_id=worst_mid,
                )
            else:
                # No stored HI history at all (fresh deployment or DB
                # unreachable): the honest protocol status, never a
                # fabricated window.
                rul_window = RulWindow(
                    estimate="Calculating... (Insufficient Data)",
                    lower=None, upper=None,
                    status="CALCULATING_INSUFFICIENT_DATA",
                    binding_motor_id=None,
                )

        # executive_decision_directive
        worst_stage = max(
            (st.stage for st in stage_by_motor.values()), default=1
        )
        worst_mid = None
        worst_fault = None
        for mid, st in stage_by_motor.items():
            if st.stage == worst_stage:
                worst_mid = mid
                worst_fault = (
                    str(states[mid]["predicted_class"])
                    if states[mid]["predicted_class"] is not None
                    else None
                )
                break
        directive = executive_decision_directive(
            worst_stage=worst_stage,
            worst_motor_id=worst_mid,
            worst_fault_class=worst_fault,
            rul_days_numeric=(
                rul_window.estimate if isinstance(rul_window.estimate, int)
                else None
            ),
            rul_status=rul_window.status,
        )

        return ExecutiveViewPayload(
            generated_at=generated_at,
            model_version=self.model_version or "unavailable",
            database_connected=db_ok,
            motors_monitored=len(states),
            plant_health_index=phi,
            plant_health_index_note=phi_note,
            financial_risk_exposure_etb=risk,
            rul_days_estimate=rul_window,
            critical_asset_risk_summary=summary,
            energy_efficiency_loss_penalty_etb=energy,
            executive_decision_directive=directive,
        )

    async def motors_overview(self) -> MotorsOverviewPayload:
        """Per-motor live summary list for the dashboard status grid.

        Same state assembly as the executive view (live verdicts union
        persisted rows, freshest wins) but WITHOUT the per-motor RUL
        reads — the deterministic Stage engine alone drives the list,
        so this stays one snapshot query + pure computation.
        """
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        generated_at = _dt.now(_tz.utc)
        db_ok = True
        try:
            db_rows = {
                r["motor_id"]: r
                for r in await self._repo.fetch_all_latest_telemetry()
            }
            self.database_connected = True
        except Exception as exc:
            db_ok = False
            self.database_connected = False
            db_rows = {}
            logger.error("Motors overview read failed: %s", exc)

        motor_ids = set(db_rows) | set(self._live_verdicts)
        entries: list[MotorOverviewEntry] = []
        for mid in sorted(motor_ids):
            try:
                state = self._latest_motor_state(
                    self._live_verdicts.get(mid), db_rows.get(mid)
                )
            except ValueError:
                continue
            stage = classify_stage(
                zscore_max=float(state["zscore_max"]),
                gating_status=str(state["status"]),
                predicted_class=(
                    str(state["predicted_class"])
                    if state["predicted_class"] is not None
                    else None
                ),
                rul_days_numeric=None,
                rul_status=None,
            )
            predicted = state["predicted_class"]
            entries.append(
                MotorOverviewEntry(
                    motor_id=mid,
                    status=str(state["status"]),
                    predicted_class=(
                        str(predicted) if predicted is not None else None
                    ),
                    model_confidence=(
                        float(state["model_confidence"])
                        if state.get("model_confidence") is not None
                        else None
                    ),
                    health_percent=assessed_health_percent(
                        float(state["zscore_max"]),
                        stage=stage.stage,
                        predicted_class=predicted,
                    ),
                    stage=stage.stage,
                    stage_label=stage.label,
                    is_fault=(
                        predicted is not None and str(predicted) != "HEALTHY"
                    ),
                    thd_percent_max=(
                        float(state["thd_percent_max"])
                        if state.get("thd_percent_max") is not None
                        else None
                    ),
                    unbalance_percent=(
                        float(state["unbalance_percent"])
                        if state.get("unbalance_percent") is not None
                        else None
                    ),
                    fundamental_hz=(
                        float(state["fundamental_hz"])
                        if state.get("fundamental_hz") is not None
                        else None
                    ),
                    recorded_at=state["recorded_at"],
                    source=str(state["source"]),
                )
            )
        return MotorsOverviewPayload(
            generated_at=generated_at,
            model_version=self.model_version or "unavailable",
            database_connected=db_ok,
            motors=entries,
        )

    async def telemetry_history(
        self, motor_id: str, hours: int, limit: int
    ) -> TrendsPayload:
        """Steady-state window history for the Analytics trends charts.

        Persistence-only read (the live in-RAM verdicts are single-window
        state, not a series): the newest ``limit`` non-inrush windows of
        the last ``hours`` hours, oldest-first. Per-window health percent
        uses the same Section 14.5 mapping as the overview; worst-phase
        THD / crest / sideband mirror the snapshot derivation. Raises
        ValueError("No telemetry recorded ...") for unknown motors (the
        route maps it to 404); a motor with telemetry outside the window
        returns an honest empty points list.
        """
        rows = await self._repo.fetch_telemetry_history(
            motor_id, hours, limit
        )
        points: list[TrendPoint] = []
        for row in rows:
            zscore_max = float(row["zscore_max"])
            confidence = row.get("model_confidence")
            predicted = row.get("predicted_class")
            row_stage = classify_stage(
                zscore_max=zscore_max,
                gating_status=str(row["status"]),
                predicted_class=(
                    str(predicted) if predicted is not None else None
                ),
                rul_days_numeric=None,
                rul_status=None,
            )
            points.append(
                TrendPoint(
                    recorded_at=row["recorded_at"],
                    status=str(row["status"]),
                    health_percent=assessed_health_percent(
                        zscore_max,
                        stage=row_stage.stage,
                        predicted_class=(
                            str(predicted) if predicted is not None else None
                        ),
                    ),
                    zscore_max=zscore_max,
                    thd_percent_max=max(
                        float(row["thd_percent_a"]),
                        float(row["thd_percent_b"]),
                        float(row["thd_percent_c"]),
                    ),
                    unbalance_percent=float(row["unbalance_percent"]),
                    crest_factor_max=max(
                        float(row["crest_factor_a"]),
                        float(row["crest_factor_b"]),
                        float(row["crest_factor_c"]),
                    ),
                    rotor_sideband_db_max=max(
                        float(row["rotor_sideband_db_a"]),
                        float(row["rotor_sideband_db_b"]),
                        float(row["rotor_sideband_db_c"]),
                    ),
                    fundamental_hz=float(row["fundamental_hz"]),
                    rms_a=float(row["rms_a"]),
                    rms_b=float(row["rms_b"]),
                    rms_c=float(row["rms_c"]),
                    predicted_class=(
                        str(predicted) if predicted is not None else None
                    ),
                    model_confidence=(
                        float(confidence) if confidence is not None else None
                    ),
                )
            )
        return TrendsPayload(
            motor_id=motor_id,
            generated_at=datetime.now(timezone.utc),
            model_version=self.model_version or "unavailable",
            database_connected=True,
            window=TrendsWindow(
                hours=hours, limit=limit, returned=len(points)
            ),
            points=points,
        )

    async def reconcile_fault_episodes(self) -> int:
        """Open an episode row for every CURRENTLY-ACTIVE fault
        condition that predates the ingestion-path write (deploy
        bootstrap — e.g. a Stage 2 bearing fault detected while the old
        trigger required a gate ANOMALY that mechanical faults never
        produce).

        Honest by construction: the condition exists NOW (live verdict
        or freshest persisted row), the evidence columns come from that
        same measured window, and ``spectral_evidence.detected_via``
        records the reconciliation provenance. Idempotent per episode:
        the newest existing row is probed first, exactly like the
        ingestion path. Fail-soft; returns rows written.
        """
        written = 0
        try:
            overview = await self.motors_overview()
        except Exception as exc:
            logger.warning(
                "Fault-episode reconciliation skipped (overview read "
                "failed): %s", exc,
            )
            return 0
        for entry in overview.motors:
            if not entry.is_fault or entry.stage < 2:
                continue
            predicted = entry.predicted_class
            if not predicted or str(predicted) == "HEALTHY":
                continue
            motor_id = entry.motor_id
            try:
                taxonomy_code = exact_taxonomy_code(entry.stage, predicted)
                latest = await self._repo.get_latest_fault_episode(
                    motor_id
                )
                if latest is not None and latest["taxonomy_code"] == (
                    taxonomy_code
                ):
                    continue  # ongoing episode already on record
                snap = None
                try:
                    snap = await self._repo.fetch_latest_snapshot(motor_id)
                except Exception:
                    snap = None
                if snap is not None:
                    zscore = float(snap.get("zscore_max") or 0.0)
                    status = str(snap.get("status") or entry.status)
                    evidence = {
                        "fundamental_hz": snap.get("fundamental_hz"),
                        "thd_percent_max": max(
                            float(snap[k])
                            for k in (
                                "thd_percent_a", "thd_percent_b",
                                "thd_percent_c",
                            )
                        ),
                        "current_unbalance_percent": float(
                            snap.get("unbalance_percent") or 0.0
                        ),
                        "crest_factor_max": max(
                            float(snap[k])
                            for k in (
                                "crest_factor_a", "crest_factor_b",
                                "crest_factor_c",
                            )
                        ),
                        "rotor_sideband_db_max": max(
                            float(snap[k])
                            for k in (
                                "rotor_sideband_db_a",
                                "rotor_sideband_db_b",
                                "rotor_sideband_db_c",
                            )
                        ),
                        "envelope_peak_hz": [
                            snap.get("envelope_peak_hz_a"),
                            snap.get("envelope_peak_hz_b"),
                            snap.get("envelope_peak_hz_c"),
                        ],
                        "zscore_max": zscore,
                        "model_version": self.model_version,
                    }
                else:  # live-only motor (DB outage window): overview data
                    zscore = 0.0
                    status = entry.status
                    evidence = {
                        "fundamental_hz": entry.fundamental_hz,
                        "thd_percent_max": entry.thd_percent_max,
                        "current_unbalance_percent": (
                            entry.unbalance_percent
                        ),
                        "zscore_unavailable": True,
                        "model_version": self.model_version,
                    }
                evidence["detected_via"] = "startup_reconciliation"
                await self._repo.log_fault(
                    motor_id=motor_id,
                    taxonomy_code=taxonomy_code,
                    urgency_stage=entry.stage,
                    model_confidence=(
                        float(entry.model_confidence or 0.0)
                    ),
                    population_sigma=zscore,
                    absolute_threshold_breached=(status == "ANOMALY"),
                    trend_confirmed=False,
                    spectral_evidence=evidence,
                )
                written += 1
                logger.info(
                    "Fault episode reconciled motor=%s code=%s stage=%d",
                    motor_id, taxonomy_code, entry.stage,
                )
            except Exception as exc:
                logger.error(
                    "Fault-episode reconciliation failed for %s: %s",
                    motor_id, exc,
                )
        return written

    async def fault_logs(
        self, motor_id: str | None, limit: int
    ) -> FaultLogsPayload:
        """Newest fault episodes for the Fault Logs page (Section 10).

        All motors or one motor, newest first. Fault rows are written
        by the ingestion path when a NEW fault episode starts (episode
        dedup on the taxonomy code) and are permanently retained.
        """
        rows = await self._repo.fetch_fault_logs(motor_id, limit)
        entries = [
            FaultLogEntry(
                id=int(row["id"]),
                motor_id=str(row["motor_id"]),
                detected_at=row["detected_at"],
                taxonomy_code=str(row["taxonomy_code"]),
                urgency_stage=int(row["urgency_stage"]),
                model_confidence=float(row["model_confidence"]),
                population_sigma=float(row["population_sigma"]),
                absolute_threshold_breached=bool(
                    row["absolute_threshold_breached"]
                ),
                trend_confirmed=bool(row["trend_confirmed"]),
                four_signal_confirmed=bool(row["four_signal_confirmed"]),
                physically_verified=bool(row["physically_verified"]),
                verified_by=row.get("verified_by"),
                spectral_evidence=(
                    row["spectral_evidence"]
                    if isinstance(row["spectral_evidence"], dict)
                    else {}
                ),
                resolved_at=row.get("resolved_at"),
            )
            for row in rows
        ]
        return FaultLogsPayload(
            generated_at=datetime.now(timezone.utc),
            database_connected=True,
            motor_id=motor_id,
            limit=limit,
            returned=len(entries),
            faults=entries,
        )

    # ------------------------------------------------------------------
    # Live alerts (derived on read) + operator dismissals
    # ------------------------------------------------------------------

    def _derive_alerts(
        self, overview: MotorsOverviewPayload
    ) -> list[AlertEntry]:
        """Derive the live alert list from a motors overview snapshot.

        Alert rule (the same condition the dashboard badge uses):
        Stage >= 2 or an active fault verdict. The alert_key is the
        condition signature — the taxonomy verdict for faults,
        otherwise STAGE_<n> — so a stage escalation or a different
        fault re-alerts and a dismissal never outlives its condition.
        """
        entries: list[AlertEntry] = []
        for m in overview.motors:
            if not (m.stage >= 2 or m.is_fault):
                continue
            if m.is_fault and m.predicted_class:
                alert_key = str(m.predicted_class)
            else:
                alert_key = f"STAGE_{m.stage}"
            severity = (
                "critical"
                if m.stage >= 3
                else "high" if m.stage == 2 else "medium"
            )
            entries.append(
                AlertEntry(
                    alert_id=f"{m.motor_id}:{alert_key}",
                    motor_id=m.motor_id,
                    alert_key=alert_key,
                    severity=severity,
                    predicted_class=m.predicted_class,
                    stage=m.stage,
                    stage_label=m.stage_label,
                    is_fault=m.is_fault,
                    status=m.status,
                    model_confidence=m.model_confidence,
                    health_percent=m.health_percent,
                    thd_percent_max=m.thd_percent_max,
                    unbalance_percent=m.unbalance_percent,
                    recorded_at=m.recorded_at,
                    source=m.source,
                )
            )
        return entries

    async def alerts(self, include_dismissed: bool = False) -> AlertsPayload:
        """Live plant alerts minus dismissals (GET /api/v2/alerts).

        Dismissals auto-expire: any stored dismissal whose condition is
        no longer derived (fault cleared / stage normalised) is deleted
        on read, so a re-triggering fault alerts again. Fault episodes
        stay immutable in fault_logs (Section 10) — alerts are the
        operator's live inbox, not an audit trail.
        """
        overview = await self.motors_overview()
        derived = self._derive_alerts(overview)
        dismissed_at: dict[tuple[str, str], object] = {}
        try:
            rows = await self._repo.fetch_alert_dismissals()
            derived_pairs = {(a.motor_id, a.alert_key) for a in derived}
            stale: list[tuple[str, str]] = []
            for row in rows:
                pair = (str(row["motor_id"]), str(row["alert_key"]))
                if pair in derived_pairs:
                    dismissed_at[pair] = row["dismissed_at"]
                else:
                    stale.append(pair)
            if stale:
                await self._repo.delete_alert_dismissals(stale)
        except Exception as exc:
            logger.warning(
                "Alert dismissal store unavailable (fail-soft: every "
                "derived alert is shown active): %s", exc,
            )
        active = [
            a
            for a in derived
            if (a.motor_id, a.alert_key) not in dismissed_at
        ]
        alerts_out = active
        if include_dismissed:
            flagged = []
            for a in derived:
                stamp = dismissed_at.get((a.motor_id, a.alert_key))
                if stamp is not None:
                    flagged.append(
                        a.model_copy(
                            update={"dismissed": True, "dismissed_at": stamp}
                        )
                    )
            alerts_out = active + flagged
        return AlertsPayload(
            generated_at=overview.generated_at,
            database_connected=overview.database_connected,
            total_active=len(active),
            alerts=alerts_out,
        )

    async def _alert_pairs_for_ids(
        self, alert_ids: list[str]
    ) -> tuple[list[tuple[str, str]], list[str]]:
        """Map alert ids to (motor_id, alert_key) pairs against the
        CURRENT derivation. Unknown ids are reported, never guessed."""
        overview = await self.motors_overview()
        by_id = {
            a.alert_id: (a.motor_id, a.alert_key)
            for a in self._derive_alerts(overview)
        }
        pairs: list[tuple[str, str]] = []
        unknown: list[str] = []
        for alert_id in alert_ids:
            pair = by_id.get(alert_id)
            if pair is None:
                unknown.append(alert_id)
            else:
                pairs.append(pair)
        return pairs, unknown

    async def dismiss_alerts(
        self, alert_ids: list[str], dismissed_by: str = "dashboard"
    ) -> tuple[int, list[str]]:
        """Dismiss live alerts (single DELETE + bulk POST).

        Returns (affected_count, unknown_ids). Raises when the
        persistence store is unreachable — the caller answers 503.
        """
        pairs, unknown = await self._alert_pairs_for_ids(alert_ids)
        if pairs:
            await self._repo.upsert_alert_dismissals(pairs, dismissed_by)
        return len(pairs), unknown

    async def restore_alerts(
        self, alert_ids: list[str]
    ) -> tuple[int, list[str]]:
        """Restore previously dismissed alerts (POST .../restore).

        Idempotent: restoring a visible alert is a no-op success.
        """
        pairs, unknown = await self._alert_pairs_for_ids(alert_ids)
        if pairs:
            await self._repo.delete_alert_dismissals(pairs)
        return len(pairs), unknown

    async def technician_view(self, motor_id: str) -> TechnicianViewPayload:
        """Section 6.b — the complete Technician payload for one motor."""
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        generated_at = _dt.now(_tz.utc)
        live = self._live_verdicts.get(motor_id)
        db_ok = True
        db_row = None
        try:
            db_row = await self._repo.fetch_latest_snapshot(motor_id)
            self.database_connected = True
        except Exception as exc:
            db_ok = False
            self.database_connected = False
            logger.error("Snapshot fetch failed for %s: %s", motor_id, exc)
        if live is None and db_row is None:
            raise ValueError(
                f"No telemetry recorded for motor {motor_id!r} yet — send "
                "frames to POST /api/v2/telemetry first."
            )
        state = self._latest_motor_state(live, db_row)

        asset = None
        if db_ok:
            try:
                asset = await self._repo.get_motor_asset(motor_id)
            except Exception as exc:
                logger.error("Asset fetch failed for %s: %s", motor_id, exc)

        stage, estimate = await self._stage_and_rul(motor_id, state) \
            if db_ok else (
                classify_stage(
                    zscore_max=float(state["zscore_max"]),
                    gating_status=str(state["status"]),
                    predicted_class=(
                        str(state["predicted_class"])
                        if state["predicted_class"] is not None else None
                    ),
                    rul_days_numeric=None, rul_status=None,
                ),
                None,
            )

        predicted = (
            str(state["predicted_class"])
            if state["predicted_class"] is not None else None
        )
        if predicted is not None:
            taxonomy_code = exact_taxonomy_code(stage.stage, predicted)
            protocol = AMHARIC_PROTOCOLS[predicted]
            tools = list(TOOLS_AND_SPARES[predicted])
        else:
            taxonomy_code = (
                f"Stage {stage.stage} — Pending First Steady-State "
                "Classification (no classified window yet)"
            )
            protocol = (
                "የመጀመሪያው የ steady-state መስክ ገና አልተመዘገበም — ትክክለኛ የተዘረዘር "
                "ትንበያ ለመስጠት ሞተሩ ቢያንስ አንድ ሙሉ መስክ (32,768 ናሙና) "
                "እንዲሰበስብ ይጠበቃል። እስከዚያው የበርካታ አስፈላጊ አደጋ "
                "አልተለየም፤ የመሰረታዊ ደህንነት ምርመራዎችን ብቻ ይከታተሉ።"
            )
            tools = [
                "የተለመዱ የግሪስ እና የንጽህና መሣሪያዎች",
                "IR thermometer (የሙቀት መለኪያ)",
                "Clamp meter (የካረንት መለኪያ)",
            ]

        registered = asset is not None
        asset_bearing = (
            asset.get("bearing_part_number") if registered else None
        )
        asset_cap = (
            asset.get("spare_capacitor_spec") if registered else None
        )
        if registered and (asset_bearing or asset_cap):
            registry_note = (
                "Asset-specific part numbers from the motors registry."
            )
        elif registered:
            registry_note = (
                "Motor registered, but spare-part fields (bearing part "
                "number / capacitor spec) are not yet entered in the "
                "asset registry — commission with the physical nameplate "
                "data."
            )
        else:
            registry_note = (
                "Motor not yet registered in the asset registry — "
                "physical location and exact part numbers require "
                "commissioning data entry (POST "
                "/api/v2/admin/motor-assets)."
            )

        return TechnicianViewPayload(
            motor_id=motor_id,
            generated_at=generated_at,
            model_version=self.model_version or "unavailable",
            database_connected=db_ok,
            mcc_panel_location=MccPanelLocation(
                mcc_panel_id=asset.get("mcc_panel_id") if registered else None,
                cabinet_number=asset.get("cabinet_number") if registered else None,
                line_zone=asset.get("line_zone_id") if registered else None,
                registered=registered,
            ),
            motor_specifications=MotorSpecifications(
                asset_id=asset.get("asset_tag") if registered else None,
                motor_id=motor_id,
                nameplate_kw=(
                    float(asset["nameplate_kw"])
                    if registered and asset.get("nameplate_kw") is not None
                    else None
                ),
                rated_rpm=(
                    int(asset["rated_rpm"])
                    if registered and asset.get("rated_rpm") is not None
                    else None
                ),
                drive_type=(
                    asset.get("drive_type") if registered else None
                ),
                registered=registered,
            ),
            exact_fault_taxonomy_code=taxonomy_code,
            dominant_fault=predicted,
            model_confidence=(
                float(state["model_confidence"])
                if state["model_confidence"] is not None else None
            ),
            spectral_evidence_data=SpectralEvidenceData(
                fundamental_hz=state["fundamental_hz"],
                fft_peak_hz=state["fft_peak_hz"],
                sideband_50hz_db=state["sideband_50hz_db"],
                thd_percent_max=state["thd_percent_max"],
                current_unbalance_percent=state["unbalance_percent"],
                crest_factor_max=state["crest_factor_max"],
                zscore_baseline_deviation=float(state["zscore_max"]),
            ),
            amharic_repair_protocol=protocol,
            required_tools_and_spares=RequiredToolsAndSpares(
                items=tools,
                asset_bearing_part_number=asset_bearing,
                asset_capacitor_spec=asset_cap,
                registry_note=registry_note,
            ),
            fault_urgency_stage=FaultUrgencyStage(
                stage=stage.stage,
                label=stage.label,
                max_operating_hours_before_trip=stage.max_operating_hours,
                basis=stage.basis,
            ),
        )
