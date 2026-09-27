"""Sigmo V2 — Deterministic Stage & Plant-Aggregation Engine
(SIGMO_RULES.md Section 6.a, backed by Sections 11.6 / 14).

Determinism statement: every function here is a pure function of its
arguments. No randomness, no model inference, no hidden state. Health
status, stage, and every financial figure are the exact arithmetic
result of (a) measured telemetry, (b) the ACTIVE classifier verdict,
and (c) the deterministic Section 14 RUL — combined with documented
commissioning-policy defaults (``backend.config.ViewConfig``) that the
customer can override per plant via the ``plant_config`` Supabase table.

Stage model (Section 6.b ``fault_urgency_stage``), anchored to the
rulebook's existing boundaries:
  Stage 1  Healthy               z < 2.0 sigma, HEALTHY verdict, no
                                absolute-threshold breach.
  Stage 2  Moderate / early     fault verdict, OR gating ANOMALY
                                (IEEE/ISO absolute breach), OR
                                2.0 <= z < 3.5.
  Stage 3  Severe               3.5 <= z < 5.0 with fault evidence.
  Stage 4  Critical             z >= 5.0 (== CRITICAL_FAILURE_THRESHOLD,
                                == 4-signal population-deviation
                                threshold), OR Section-14 RUL exhausted
                                (numeric 0 / CRITICAL_THRESHOLD_REACHED).

Financial model (Section 6.a):
  * financial_risk_exposure_etb = sum over Stage >= 2 motors of
    expected_outage_hours(stage) x downtime_cost_etb_per_hour(motor).
    Rates are customer-entered (asset registry / plant_config); with no
    rate configured the value is honestly null with a reason — never an
    invented number.
  * energy_efficiency_loss_penalty_etb (per month) = tariff
    x operating_hours x nameplate_kw x penalty_fraction, where
    penalty_fraction = THD_frac^2  (harmonic copper-loss scaling)
                      + 0.01 x max(0, unbalance_pct - 1.0)
                        (NEMA MG-1 style derating)
                      + max(0, 1/pf - 1)  (only when a measured plant
                        power factor is registered; current-only MCSA
                        cannot measure true PF, so it is never assumed).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import VIEW


# ----------------------------------------------------------------------
# Stage classification (Section 6.b fault_urgency_stage)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class StageAssessment:
    """One motor's urgency stage with the exact evidence that produced
    it — the basis is reported, never hidden."""

    stage: int                      # 1..4
    label: str
    max_operating_hours: int | None  # None = unlimited (Stage 1)
    basis: str


STAGE_LABELS: dict[int, str] = {
    1: "Stage 1 — Healthy",
    2: "Stage 2 — Moderate Risk (Early-Stage Fault)",
    3: "Stage 3 — Severe Risk",
    4: "Stage 4 — Critical Failure Risk",
}


def classify_stage(
    *,
    zscore_max: float,
    gating_status: str,
    predicted_class: str | None,
    rul_days_numeric: int | None,
    rul_status: str | None,
    max_operating_hours: dict[int, int | None] | None = None,
) -> StageAssessment:
    """Deterministic Stage 1-4 verdict for one motor.

    ``zscore_max`` is the Health Index (Section 14.1); ``gating_status``
    is the IEEE/ISO absolute-threshold gate result; ``predicted_class``
    the ACTIVE classifier verdict (None = not classified); the RUL pair
    is the deterministic Section 14 estimate.
    """
    policy = max_operating_hours or VIEW.max_operating_hours
    fault_verdict = predicted_class is not None and predicted_class != "HEALTHY"

    # Stage 4 — critical boundary (14.5 threshold == 11.6 signal-1)
    if (
        zscore_max >= VIEW.stage4_z
        or rul_status == "CRITICAL_THRESHOLD_REACHED"
        or (rul_days_numeric is not None and rul_days_numeric <= 0)
    ):
        basis = (
            f"Health Index {zscore_max:.2f} sigma >= "
            f"{VIEW.stage4_z:.1f} sigma critical threshold"
            if zscore_max >= VIEW.stage4_z
            else "Section 14 RUL exhausted (critical threshold reached)"
        )
        return StageAssessment(4, STAGE_LABELS[4], policy[4], basis)

    # Stage 3 — severe degradation band
    if zscore_max >= VIEW.stage3_z:
        return StageAssessment(
            3, STAGE_LABELS[3], policy[3],
            f"Health Index {zscore_max:.2f} sigma in severe band "
            f"[{VIEW.stage3_z:.1f}, {VIEW.stage4_z:.1f})",
        )

    # Stage 2 — any honest fault evidence
    if fault_verdict:
        return StageAssessment(
            2, STAGE_LABELS[2], policy[2],
            f"Classifier verdict {predicted_class} (early-stage fault)",
        )
    if gating_status == "ANOMALY":
        return StageAssessment(
            2, STAGE_LABELS[2], policy[2],
            "IEEE/ISO absolute-threshold breach (gating ANOMALY)",
        )
    if zscore_max >= VIEW.stage2_z:
        return StageAssessment(
            2, STAGE_LABELS[2], policy[2],
            f"Health Index {zscore_max:.2f} sigma above early-fault "
            f"boundary {VIEW.stage2_z:.1f} sigma",
        )

    # Stage 1 — healthy
    return StageAssessment(
        1, STAGE_LABELS[1], policy[1],
        f"Health Index {zscore_max:.2f} sigma within tolerance, "
        "no fault verdict, no absolute-threshold breach",
    )


def motor_health_percent(zscore_max: float) -> float:
    """Per-motor normalized health index, 0-100 %.

    Linear mapping anchored to the rulebook: z = 0 -> 100 %,
    z >= CRITICAL_FAILURE_THRESHOLD (5.0 sigma) -> 0 %. Deterministic
    and monotone; the same 5-sigma anchor as Stage 4 and Section 14.5.
    """
    critical = VIEW.stage4_z
    return round(100.0 * max(0.0, min(1.0, 1.0 - zscore_max / critical)), 1)


def plant_health_index(zscores: list[float]) -> float | None:
    """Plant-wide 0-100 % health index: mean of per-motor health
    percentages over all motors with telemetry. None when no motor has
    telemetry yet (honest absence, never a fabricated 100 %)."""
    if not zscores:
        return None
    return round(
        sum(motor_health_percent(z) for z in zscores) / len(zscores), 1
    )


# ----------------------------------------------------------------------
# Financial model (Section 6.a)
# ----------------------------------------------------------------------
def financial_risk_exposure_etb(
    stage_hours: list[tuple[int, float]],
) -> float:
    """Downtime risk exposure: sum(stage, cost_etb_per_hour) of
    expected_outage_hours(stage) x cost. Only Stage >= 2 motors carry
    exposure. Caller supplies configured rates; None rates must be
    excluded upstream and reported as unconfigured."""
    exposure = 0.0
    for stage, cost in stage_hours:
        if stage >= 2 and cost > 0:
            exposure += VIEW.expected_outage_hours[stage] * cost
    return round(exposure, 2)


def energy_penalty_fraction(
    thd_percent_max: float,
    unbalance_percent: float,
    plant_power_factor: float | None = None,
) -> float:
    """Fraction of active power lost to power-quality defects.

    THD^2 copper-loss scaling + NEMA MG-1 style unbalance derating
    (first 1 % is the free band) + PF term only when a measured plant
    power factor is registered. Pure arithmetic on measured values.
    """
    thd_frac = max(0.0, thd_percent_max) / 100.0
    unb_excess = max(
        0.0, unbalance_percent - VIEW.unbalance_free_band_percent
    )
    fraction = thd_frac * thd_frac + (
        VIEW.unbalance_loss_per_percent * unb_excess
    )
    if plant_power_factor is not None and 0.1 < plant_power_factor < 1.0:
        fraction += max(0.0, 1.0 / plant_power_factor - 1.0)
    return fraction


def energy_penalty_monthly_etb(
    *,
    nameplate_kw: float,
    thd_percent_max: float,
    unbalance_percent: float,
    tariff_etb_per_kwh: float,
    operating_hours_per_month: float | None = None,
    plant_power_factor: float | None = None,
) -> float:
    """Estimated monthly ETB loss from power-quality defects for one
    motor, on a nameplate-kW basis (documented conservative basis; the
    registry carries the nameplate rating)."""
    hours = (
        operating_hours_per_month
        if operating_hours_per_month is not None
        else VIEW.default_operating_hours_per_month
    )
    fraction = energy_penalty_fraction(
        thd_percent_max, unbalance_percent, plant_power_factor
    )
    return round(
        nameplate_kw * fraction * hours * tariff_etb_per_kwh, 2
    )


# ----------------------------------------------------------------------
# Executive decision directive (Section 6.a)
# ----------------------------------------------------------------------
def executive_decision_directive(
    *,
    worst_stage: int,
    worst_motor_id: str | None,
    worst_fault_class: str | None,
    rul_days_numeric: int | None,
    rul_status: str | None,
) -> str:
    """One concise strategic action sentence for plant leadership.

    Deterministic template selection — no LLM, no speculation. When the
    plant is fully healthy the directive says exactly that.
    """
    if worst_stage <= 1:
        return (
            "Plant healthy — no intervention required. Continue "
            "condition monitoring and scheduled maintenance."
        )
    where = f"motor {worst_motor_id}" if worst_motor_id else "the flagged motor"
    fault = (
        worst_fault_class.replace("_", " ").title()
        if worst_fault_class
        else "degradation"
    )
    if worst_stage >= 4:
        return (
            f"CRITICAL: trip or shut down {where} immediately "
            f"(Stage 4, {fault}); mobilize repair resources now and "
            "reallocate load to maintain production."
        )
    if rul_days_numeric is not None:
        return (
            f"Plan {fault} repair on {where} (Stage {worst_stage}) "
            f"during the next scheduled stop within {rul_days_numeric} "
            "days — the deterministic RUL window."
        )
    if rul_status == "STABLE":
        return (
            f"Schedule {fault} inspection on {where} (Stage "
            f"{worst_stage}) at the next maintenance window; condition "
            "is currently stable but fault evidence is confirmed."
        )
    return (
        f"Schedule {fault} repair on {where} (Stage {worst_stage}) at "
        "the next maintenance window; RUL is still being established "
        "from stored Health-Index history."
    )
