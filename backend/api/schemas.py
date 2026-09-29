"""Sigmo V2 — API contracts for the ESP32 ingestion layer.

The ESP32 is strictly a lightweight Raw Array Sender (SIGMO_RULES.md
Section 4.1): it POSTs synchronized three-phase raw ADC sample arrays of
2048 or 4096 points captured at 10.24 kHz. All DSP happens backend-side.
Raw arrays are validated here, processed in RAM, and never persisted
(Section 10).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ..config import DSP


class TelemetryFrame(BaseModel):
    """One synchronized three-phase acquisition frame from an ESP32 node.

    `sequence` and `captured_at` support firmware-side store-and-forward:
    a node that buffered frames during a network/power outage replays
    them with original capture timestamps and monotonically increasing
    sequence numbers (Section 9).
    """

    motor_id: str = Field(min_length=1, max_length=64)
    device_id: str = Field(min_length=1, max_length=64)
    sequence: int = Field(ge=0)
    captured_at: datetime
    sampling_rate_hz: float
    i_a: list[float]
    i_b: list[float]
    i_c: list[float]

    @field_validator("sampling_rate_hz")
    @classmethod
    def _validate_rate(cls, v: float) -> float:
        if abs(v - DSP.sampling_rate_hz) > 1.0:
            raise ValueError(
                f"sampling_rate_hz must be {DSP.sampling_rate_hz} Hz "
                f"(ESP32 I2S DMA contract), got {v}"
            )
        return v

    @model_validator(mode="after")
    def _validate_arrays(self) -> "TelemetryFrame":
        lengths = {len(self.i_a), len(self.i_b), len(self.i_c)}
        if len(lengths) != 1:
            raise ValueError(f"Phase arrays must be equal length, got {sorted(lengths)}")
        n = lengths.pop()
        if n not in DSP.frame_lengths:
            raise ValueError(
                f"Frame length must be one of {DSP.frame_lengths}, got {n}"
            )
        for name, arr in (("i_a", self.i_a), ("i_b", self.i_b), ("i_c", self.i_c)):
            for x in (arr[0], arr[-1]):
                if x != x or x in (float("inf"), float("-inf")):
                    raise ValueError(f"{name} contains non-finite samples")
        return self


class FrameAck(BaseModel):
    """Acknowledgement returned to the ESP32 for each accepted frame."""

    accepted: bool
    motor_id: str
    sequence: int
    window_fill_percent: float = Field(ge=0.0, le=100.0)
    window_analyzed: bool
    analysis: "WindowAnalysis | None" = None


class WindowAnalysis(BaseModel):
    """Result summary emitted when an accumulation window completes."""

    motor_id: str
    window_completed_at: datetime
    status: Literal["HEALTHY", "ANOMALY", "INRUSH_SUPPRESSED"]
    fundamental_hz: float
    rms_a: float
    rms_b: float
    rms_c: float
    thd_percent_max: float
    current_unbalance_percent: float
    crest_factor_max: float
    rotor_sideband_db_max: float
    zscore_max: float
    absolute_threshold_breaches: list[str]
    persisted: bool
    buffered_offline: bool
    # v6c dominant-fault verdict (Phase 4). None when the runtime could
    # not load a model bundle (degraded mode) or the window was
    # INRUSH_SUPPRESSED — the classifier is trained on steady-state
    # windows only, so startup transients are never classified.
    dominant_fault: str | None = None
    model_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    temporal_aggregation: dict[str, object] | None = None
    model_version: str | None = None


class HealthResponse(BaseModel):
    """Liveness/readiness for deployment monitoring."""

    service: str
    database_connected: bool
    offline_buffer_depth: int
    motors_streaming: int


class RulAssessment(BaseModel):
    """Deterministic RUL payload — SIGMO_RULES.md Section 14.6 contract.

    Embedded in the Executive View (Section 6.a `rul_days_estimate` and
    bounds) and the Technician View (audit fields). The estimate is
    computed EXCLUSIVELY by the Section 14 slope-extrapolation protocol
    over stored Supabase telemetry — never by an unverifiable model.
    """

    motor_id: str
    rul_days_estimate: int | str = Field(
        description=(
            "Integer days (numeric branch) or one of the exact status "
            "strings: 'Calculating... (Insufficient Data)', "
            "'Calculating... (Building Trend)', 'Stable (999+ Days)'."
        )
    )
    rul_status: Literal[
        "CALCULATING_INSUFFICIENT_DATA",
        "CALCULATING_BUILDING_TREND",
        "STABLE",
        "DEGRADING",
        "CRITICAL_THRESHOLD_REACHED",
    ]
    slope_sigma_per_day: float | None = Field(
        description="OLS d(HI)/dt over the trend window; null while calculating."
    )
    current_health_index_sigma: float | None = Field(
        description="HI (zscore_max) of the most recent non-inrush window."
    )
    trend_window_days: float | None = Field(
        description="Time span of the trend window actually used."
    )
    rul_days_lower: int | None = Field(
        description="Lower confidence bound (slope + SE); numeric estimates only."
    )
    rul_days_upper: int | None = Field(
        description="Upper confidence bound (slope − SE, capped 999); numeric estimates only."
    )
    trend_points: int = Field(
        ge=0, description="Non-inrush windows inside the trend window."
    )
    last_window_at: datetime | None = Field(
        description="Timestamp of the most recent non-inrush window (staleness audit)."
    )


class MotorSnapshot(BaseModel):
    """Latest persisted telemetry window (physical DSP measurements)."""

    recorded_at: datetime
    status: Literal["HEALTHY", "ANOMALY", "INRUSH_SUPPRESSED"]
    fundamental_hz: float
    rms_a: float
    rms_b: float
    rms_c: float
    thd_percent_max: float
    current_unbalance_percent: float
    crest_factor_max: float
    rotor_sideband_db_max: float
    zscore_max: float = Field(
        description="Health Index: largest baseline z-score of the window."
    )


class FaultAssessment(BaseModel):
    """Dominant-fault verdict of the ACTIVE v6c classifier.

    Single-label by design: the verdict names the DOMINANT fault of the
    most recent steady-state window (compound faults surface as the
    dominant component — OPTIMIZATION_REPORT.md 'stress-compound-eval').
    The temporal block carries the rolling-majority aggregation of the
    live stream (SIGMO_RULES.md Section 11.6); it never replaces the
    per-window verdict.
    """

    dominant_fault: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_fault: bool = Field(description="True when dominant_fault != 'HEALTHY'.")
    probabilities: dict[str, float] = Field(
        description="Full posterior over the 13-class taxonomy (sums to 1)."
    )
    temporal_aggregation: dict[str, object] | None = Field(
        default=None,
        description=(
            "Rolling-majority vote over the last 5 windows of this "
            "motor's live stream (windows_buffered, aggregated_class, "
            "vote_share, stable); null when the stream has no live "
            "history (e.g. after a service restart)."
        ),
    )
    model_version: str
    classified_at: datetime = Field(
        description="Timestamp of the window this verdict belongs to."
    )
    source: Literal["live_runtime", "supabase"] = Field(
        description="live_runtime: freshest analyzed window in RAM; "
        "supabase: persisted row (service restarted or motor idle)."
    )


class MotorAssessment(BaseModel):
    """Complete per-motor status payload (Phase 4 comprehensive contract).

    One structured JSON response carrying everything a dashboard needs
    right now — physical snapshot, dominant-fault verdict, health index
    and the deterministic Section 14 RUL — and the exact field
    foundation for the Dual-View UX (SIGMO_RULES.md Section 6) when it
    is built. No field in this payload is ever a placeholder estimate.
    """

    motor_id: str
    model_version: str = Field(
        description="ACTIVE classifier version serving the verdicts."
    )
    generated_at: datetime
    database_connected: bool
    snapshot: MotorSnapshot | None = Field(
        default=None,
        description="Latest physical measurements; null before the "
        "first window completes.",
    )
    snapshot_source: Literal["supabase", "live_runtime"] | None = None
    fault_assessment: FaultAssessment | None = Field(
        default=None,
        description="null until the first steady-state window is "
        "classified (or in degraded no-bundle mode)."
    )
    rul: RulAssessment | None = Field(
        default=None,
        description="Deterministic Section 14 estimate; null only when "
        "Supabase is unreachable (RUL reads stored HI history)."
    )
    rul_unavailable_reason: str | None = None


# ======================================================================
# Dual-View UX payloads (SIGMO_RULES.md Section 6, Phase 5)
# ======================================================================
class FinancialAmount(BaseModel):
    """A financial figure in ETB with its honest computation status.

    ``value_etb`` is null exactly when the required customer-entered
    rate is not configured — never an invented number (zero-fake-code
    directive). ``basis`` names the formula and inputs used.
    """

    value_etb: float | None
    status: Literal[
        "computed", "no_active_risk", "rate_not_configured",
        "partial_rates_configured",
    ]
    basis: str
    unconfigured_items: list[str] = Field(default_factory=list)


class RulWindow(BaseModel):
    """Section 6.a ``rul_days_estimate`` — the binding plant RUL window,
    computed EXCLUSIVELY by the deterministic Section 14 protocol."""

    estimate: int | str = Field(
        description="Whole days, or the exact Section 14.6 status string."
    )
    lower: int | None = Field(
        description="OLS slope-SE bound (14.6.5); only with numeric "
        "estimates."
    )
    upper: int | None = None
    status: str = Field(description="Section 14.6 rul_status of the "
                                    "binding motor.")
    binding_motor_id: str | None = Field(
        description="The motor whose RUL binds the plant window."
    )


class CriticalAssetRiskSummary(BaseModel):
    """Section 6.a ``critical_asset_risk_summary`` — motors by status."""

    healthy: int = Field(description="Stage 1 motors.")
    stage_2_moderate_risk: int = Field(description="Stage 2 motors.")
    stage_3_4_critical_failure_risk: int = Field(
        description="Stage 3 + Stage 4 motors."
    )
    total_monitored: int


class ExecutiveViewPayload(BaseModel):
    """Section 6.a — the strictly isolated Executive payload.

    Every field name is the exact rulebook contract; nothing generic,
    nothing estimated without a stated basis."""

    generated_at: datetime
    model_version: str
    database_connected: bool
    motors_monitored: int
    plant_health_index: float | None = Field(
        description="0-100% mean of per-motor health indices; null "
        "before any motor has telemetry."
    )
    plant_health_index_note: str | None = None
    financial_risk_exposure_etb: FinancialAmount
    rul_days_estimate: RulWindow
    critical_asset_risk_summary: CriticalAssetRiskSummary
    energy_efficiency_loss_penalty_etb: FinancialAmount
    executive_decision_directive: str = Field(
        description="Deterministic strategic action sentence."
    )


class MotorOverviewEntry(BaseModel):
    """One motor's live summary for the plant overview list.

    Freshness precedence mirrors the assessment contract: the live
    in-RAM verdict when this service analyzed the newest steady-state
    window, otherwise the latest persisted Supabase row.
    """

    motor_id: str
    status: str = Field(description="Gating status of the newest window.")
    predicted_class: str | None = None
    model_confidence: float | None = None
    health_percent: float | None = Field(
        description="Operator health score 0-100: z-anchored base "
        "(Section 14.5), capped into the fault-stage band while a "
        "fault condition is active (Stage 2 -> 60-80, Stage 3 -> "
        "40-60, Stage 4 -> 0-40)."
    )
    stage: int = Field(description="Stage 1-4 (deterministic engine).")
    stage_label: str
    is_fault: bool
    thd_percent_max: float | None = None
    unbalance_percent: float | None = None
    fundamental_hz: float | None = None
    recorded_at: datetime
    source: str = Field(description="'live_runtime' or 'supabase'.")


class MotorsOverviewPayload(BaseModel):
    """GET /api/v2/motors — every motor with telemetry, one response."""

    generated_at: datetime
    model_version: str
    database_connected: bool
    motors: list[MotorOverviewEntry]


class TrendPoint(BaseModel):
    """One steady-state window in the per-motor analytics time series."""

    recorded_at: datetime
    status: Literal["HEALTHY", "ANOMALY"]
    health_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Operator health score 0-100 (z-anchored base, "
        "capped into the fault-stage band while a fault condition was "
        "active on that window).",
    )
    zscore_max: float = Field(
        description="Largest baseline z-score of the window (sigma)."
    )
    thd_percent_max: float = Field(
        description="Worst-phase current THD (%)."
    )
    unbalance_percent: float = Field(
        description="Current unbalance (%)."
    )
    crest_factor_max: float = Field(
        description="Worst-phase crest factor (unitless)."
    )
    rotor_sideband_db_max: float = Field(
        description="Worst-phase rotor sideband level (dB)."
    )
    fundamental_hz: float
    rms_a: float
    rms_b: float
    rms_c: float
    predicted_class: str | None = Field(
        default=None,
        description=(
            "Persisted v6c dominant-fault verdict; null for windows "
            "written before the Phase-4 wiring, degraded-mode windows, "
            "or HEALTHY windows the classifier never labelled."
        ),
    )
    model_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0
    )


class TrendsWindow(BaseModel):
    """Echo of the request bounds plus honest row accounting."""

    hours: int
    limit: int
    returned: int


class TrendsPayload(BaseModel):
    """GET /api/v2/motors/{motor_id}/history — steady-state window series.

    Source for the dashboard Analytics trends charts. INRUSH_SUPPRESSED
    windows are excluded (startup transients are not a health state —
    same exclusion as Section 14.2.2 and the assessment read). An empty
    ``points`` list means the motor has telemetry, just none inside the
    requested window: honest absence, never fabricated rows.
    """

    motor_id: str
    generated_at: datetime
    model_version: str
    database_connected: bool
    window: TrendsWindow
    points: list[TrendPoint] = Field(
        description="Ascending by recorded_at (oldest first, chart-ready)."
    )


class FaultLogEntry(BaseModel):
    """One persisted fault episode (Section 10 permanent retention)."""

    id: int
    motor_id: str
    detected_at: datetime
    taxonomy_code: str = Field(
        description="Exact taxonomy code, e.g. 'Stage 2 — ... (CLASS)'."
    )
    urgency_stage: int = Field(ge=1, le=4)
    model_confidence: float = Field(ge=0.0, le=1.0)
    population_sigma: float = Field(
        description="Baseline deviation (zscore_max) at detection time."
    )
    absolute_threshold_breached: bool
    trend_confirmed: bool = Field(
        description=(
            "Section 11.6 temporal confirmation: the rolling-majority "
            "vote of the last 5 windows agrees with this verdict."
        )
    )
    four_signal_confirmed: bool = Field(
        description=(
            "All four Section 11.6 anti-false-positive signals held: "
            "sigma > 5, absolute threshold, confidence > 0.70, trend."
        )
    )
    physically_verified: bool = False
    verified_by: str | None = None
    spectral_evidence: dict[str, object] = Field(
        description="Quantitative DSP evidence of the detection window."
    )
    resolved_at: datetime | None = None


class FaultLogsPayload(BaseModel):
    """GET /api/v2/faults — newest fault episodes, newest first."""

    generated_at: datetime
    database_connected: bool
    motor_id: str | None = Field(
        default=None,
        description="Filter that was applied (null = all motors).",
    )
    limit: int
    returned: int
    faults: list[FaultLogEntry]


class AlertEntry(BaseModel):
    """One live plant alert, derived on read from a motor's condition.

    Alerts are never stored: an alert exists while a motor's newest
    window shows Stage >= 2 or an active fault verdict. A dismissal
    hides an alert while its condition persists and is auto-forgotten
    once the condition clears. The audit trail lives separately and
    immutably in fault_logs (Section 10).
    """

    alert_id: str = Field(
        description="Stable id '{motor_id}:{alert_key}' for dismiss/restore."
    )
    motor_id: str
    alert_key: str = Field(
        description="Condition signature: taxonomy verdict or STAGE_<n>."
    )
    severity: Literal["critical", "high", "medium"]
    predicted_class: str | None = None
    stage: int
    stage_label: str
    is_fault: bool
    status: str = Field(description="Gating status of the newest window.")
    model_confidence: float | None = None
    health_percent: float | None = None
    thd_percent_max: float | None = None
    unbalance_percent: float | None = None
    recorded_at: datetime
    source: str = Field(description="'live_runtime' or 'supabase'.")
    dismissed: bool = False
    dismissed_at: datetime | None = None


class AlertsPayload(BaseModel):
    """GET /api/v2/alerts — the operator's live alert inbox."""

    generated_at: datetime
    database_connected: bool
    total_active: int
    alerts: list[AlertEntry] = Field(
        description="Active alerts first; dismissed ones (only with "
        "include_dismissed=true) appended after them, flagged."
    )


class AlertDismissRequest(BaseModel):
    """POST /api/v2/alerts/dismiss | restore — bulk body."""

    alert_ids: list[str] = Field(
        min_length=1, max_length=100,
        description="Alert ids as returned by GET /api/v2/alerts.",
    )


class AlertDismissalResult(BaseModel):
    """Per-item honest outcome of a dismiss or restore call."""

    requested: int
    affected: int = Field(
        description="Items dismissed or restored (idempotent repeats "
        "count as affected)."
    )
    unknown: list[str] = Field(
        description="Ids matching no currently-derived alert."
    )


class AIExplainRequest(BaseModel):
    """POST /api/v2/ai/explain — request a real-LLM technician analysis."""

    motor_id: str = Field(min_length=1, max_length=64)


class InviteTechnicianRequest(BaseModel):
    """POST /api/v2/invitations — the admin invite modal payload.

    Email-only by design: the technician sets their own password from
    the Supabase invitation email; nothing else is collected here.
    """

    email: str = Field(min_length=3, max_length=254)


class AIExplainResponse(BaseModel):
    """POST /api/v2/ai/explain — bilingual (Amharic + English) result.

    The analysis is generated by a REAL LLM (openai / deepseek /
    gemini — whichever key is configured) from the actual V2 telemetry
    evidence echoed in ``data_context``. Never a canned or fabricated
    answer: with no provider key the route fails honestly with 503.
    """

    motor_id: str
    provider: str = Field(description="openai | deepseek | gemini")
    model: str
    generated_at: datetime
    analysis_am: str = Field(description="Amharic analysis for field technicians.")
    analysis_en: str = Field(description="English analysis.")
    elapsed_ms: int
    data_context: dict[str, object] = Field(
        description="The exact measured evidence that was fed to the model."
    )


class MccPanelLocation(BaseModel):
    """Section 6.b ``mcc_panel_location`` — physical hardware location."""

    mcc_panel_id: str | None
    cabinet_number: str | None
    line_zone: str | None
    registered: bool = Field(
        description="False until the asset is registered in the motors "
        "registry (commissioning data entry)."
    )


class MotorSpecifications(BaseModel):
    """Section 6.b ``motor_specifications`` — nameplate identity."""

    asset_id: str | None = Field(description="Motor Asset ID / tag.")
    motor_id: str
    nameplate_kw: float | None
    rated_rpm: int | None
    drive_type: Literal["DOL", "SOFT_STARTER", "VFD"] | None
    registered: bool


class SpectralEvidenceData(BaseModel):
    """Section 6.b ``spectral_evidence_data`` — quantitative DSP metrics
    of the latest steady-state window (measured, never estimated)."""

    fundamental_hz: float | None
    fft_peak_hz: list[float] = Field(
        description="Envelope peak frequencies per phase (Hz)."
    )
    sideband_50hz_db: list[float] = Field(
        description="Rotor sideband ratios around line frequency per "
        "phase (dB)."
    )
    thd_percent_max: float | None
    current_unbalance_percent: float | None
    crest_factor_max: float | None
    zscore_baseline_deviation: float = Field(
        description="Z-score baseline deviation magnitude (Health "
        "Index, sigma)."
    )


class RequiredToolsAndSpares(BaseModel):
    """Section 6.b ``required_tools_and_spares``."""

    items: list[str]
    items_en: list[str] = Field(
        default_factory=list,
        description="English display edition of the tools list "
        "(same order as ``items``; empty only when unavailable)."
    )
    asset_bearing_part_number: str | None = Field(
        description="Exact bearing part number from the asset registry; "
        "null when not registered (never invented)."
    )
    asset_capacitor_spec: str | None = Field(
        description="Capacitor specification from the asset registry."
    )
    registry_note: str | None = None


class FaultUrgencyStage(BaseModel):
    """Section 6.b ``fault_urgency_stage`` — Stage 1-4 severity."""

    stage: int = Field(ge=1, le=4)
    label: str
    max_operating_hours_before_trip: int | None = Field(
        description="Maximum allowable running hours before mandatory "
        "trip/shutdown; None = unlimited (Stage 1)."
    )
    basis: str = Field(description="The exact evidence that set the "
                                    "stage.")


class TechnicianViewPayload(BaseModel):
    """Section 6.b — the strictly isolated Technician payload."""

    motor_id: str
    generated_at: datetime
    model_version: str
    database_connected: bool
    mcc_panel_location: MccPanelLocation
    motor_specifications: MotorSpecifications
    exact_fault_taxonomy_code: str
    dominant_fault: str | None = Field(
        description="Raw 13-class verdict backing the taxonomy code "
        "(null before first steady-state classification)."
    )
    model_confidence: float | None = None
    spectral_evidence_data: SpectralEvidenceData
    amharic_repair_protocol: str = Field(
        description="Canonical Amharic rulebook repair protocol "
        "(SIGMO_RULES.md Section 6.b). Field name is part of the "
        "public API contract and must not change."
    )
    repair_protocol_en: str = Field(
        description="English display edition of the repair protocol — "
        "identical steps/limits, translated for the EN UI toggle."
    )
    required_tools_and_spares: RequiredToolsAndSpares
    fault_urgency_stage: FaultUrgencyStage


class MotorAssetRegistration(BaseModel):
    """Commissioning data entry for one motor (asset registry write).

    Physical identity is mandatory (it IS the Section 6.b payload
    source); the spares/cost extension fields are optional and default
    to honestly-unregistered."""

    mcc_panel_id: str = Field(min_length=1)
    cabinet_number: str = Field(min_length=1)
    line_zone_id: str = Field(min_length=1)
    nameplate_kw: float = Field(gt=0)
    rated_rpm: int = Field(gt=0)
    drive_type: Literal["DOL", "SOFT_STARTER", "VFD"]
    asset_tag: str | None = None
    bearing_part_number: str | None = None
    spare_capacitor_spec: str | None = None
    downtime_cost_etb_per_hour: float | None = Field(default=None, gt=0)


class PlantConfigEntry(BaseModel):
    """One business-rate configuration write (audit trail in
    updated_at). Values are customer-entered — never seeded."""

    value: float
    unit: str = Field(min_length=1)
    description: str = Field(min_length=1)


FrameAck.model_rebuild()
