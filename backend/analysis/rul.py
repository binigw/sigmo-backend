"""Sigmo V2 — Deterministic RUL via Slope Extrapolation
(SIGMO_RULES.md Section 14, "Distance-to-Healthy Core").

Classification statement (Section 14 preamble): this module is a
DETERMINISTIC, mathematics-based time-series extrapolation over live
telemetry — standard industrial condition-monitoring practice. It is
NOT a statistical/ML run-to-failure prognosis model and never claims
failure-time knowledge that does not exist in the data. Every output
is either (a) the exact arithmetic result of stored Supabase telemetry
or (b) an explicit non-numeric status string.

Health Index (Section 14.1): the per-window Population/Personalized
Baseline deviation magnitude ``zscore_max`` in sigma units, already
computed by the ingestion gate and persisted in
``telemetry_features``. The Isolation-Forest novelty gate keeps its
Section 11.6 role (trip flag) and is NOT the HI.

Determinism guarantee: ``estimate_rul`` is a pure function of its
arguments — identical rows in, identical estimate out. The database is
read exactly once per assessment (``assess_rul``); there is no hidden
state, no randomness, no model inference.

Interpretation notes pinned by the rulebook:
  * Cold-start clock (14.3): wall-clock span from the motor's EARLIEST
    non-inrush row to NOW. A motor with ≥ 24 h of history that simply
    has no rows in the current trend window reports
    CALCULATING_BUILDING_TREND — never a stale numeric estimate
    (fail-safe by design).
  * Trend window (14.4): the last ``RUL.lookback_days`` days of
    wall-clock time, INRUSH_SUPPRESSED rows excluded at the SQL level.
  * Alerting: an RUL estimate NEVER triggers an alert by itself
    (14.7.2); alerting stays exclusively with the 4-signal
    confirmation rule (Section 11.6).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from ..config import RUL

if TYPE_CHECKING:  # runtime import would be circular (repository → rul)
    from ..db.repository import SigmoRepository

_SECONDS_PER_DAY = 86_400.0


class RulStatus(str, Enum):
    """Section 14.6 audit statuses (verbatim from the rulebook)."""

    CALCULATING_INSUFFICIENT_DATA = "CALCULATING_INSUFFICIENT_DATA"
    CALCULATING_BUILDING_TREND = "CALCULATING_BUILDING_TREND"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    CRITICAL_THRESHOLD_REACHED = "CRITICAL_THRESHOLD_REACHED"


# Section 14.6 exact output strings — part of the API contract.
MSG_INSUFFICIENT_DATA = "Calculating... (Insufficient Data)"
MSG_BUILDING_TREND = "Calculating... (Building Trend)"
MSG_STABLE = "Stable (999+ Days)"


@dataclass(frozen=True)
class HealthIndexSample:
    """One non-inrush telemetry row: timestamp + HI in sigma."""

    recorded_at: datetime
    health_index_sigma: float


@dataclass(frozen=True)
class HealthIndexHistory:
    """One motor's fetched HI history for a single RUL assessment.

    Produced by ``SigmoRepository.fetch_health_index_history``
    (Section 14.2): INRUSH_SUPPRESSED rows are already excluded at the
    SQL level. Pure data — no behaviour, no I/O.
    """

    first_recorded_at: datetime | None
    current_sample: HealthIndexSample | None
    window_samples: list[HealthIndexSample]


@dataclass(frozen=True)
class RulEstimate:
    """Full Section 14.6 output contract + audit fields.

    ``rul_days_estimate`` is either an integer day count (numeric
    branch of the contract) or one of the three exact status strings.
    ``rul_days_lower`` / ``rul_days_upper`` are the confidence bounds
    of Section 14.5.5, populated only alongside a numeric estimate.
    """

    rul_status: RulStatus
    rul_days_estimate: int | str
    slope_sigma_per_day: float | None
    current_health_index_sigma: float | None
    trend_window_days: float | None
    rul_days_lower: int | None
    rul_days_upper: int | None
    trend_points: int
    last_window_at: datetime | None

    @property
    def is_numeric(self) -> bool:
        return isinstance(self.rul_days_estimate, int)


# ----------------------------------------------------------------------
# Pure deterministic mathematics (Section 14.4 / 14.5)
# ----------------------------------------------------------------------
def ols_slope_sigma_per_day(
    samples: list[HealthIndexSample],
) -> tuple[float, float]:
    """Ordinary-least-squares regression of HI on time.

    Returns ``(slope_sigma_per_day, slope_standard_error)``. Raises
    ValueError on degenerate input (fewer than 2 samples or zero time
    span) — callers must satisfy the Section 14.4.2 minimums first, so
    a degenerate call is a programming error, not a data state.
    """
    if len(samples) < 2:
        raise ValueError("OLS requires at least 2 samples")
    t_days = np.asarray(
        [s.recorded_at.timestamp() for s in samples], dtype=np.float64
    ) / _SECONDS_PER_DAY
    hi = np.asarray([s.health_index_sigma for s in samples], dtype=np.float64)
    t_centered = t_days - t_days.mean()
    sxx = float(np.sum(t_centered ** 2))
    if sxx <= 0.0:
        raise ValueError("zero time span in trend window (degenerate OLS)")
    slope = float(np.sum(t_centered * (hi - hi.mean())) / sxx)
    intercept = float(hi.mean() - slope * t_days.mean())
    residuals = hi - (intercept + slope * t_days)
    n = len(samples)
    sse = float(np.sum(residuals ** 2))
    # With n >= 2 the divisor is >= 0; n == 2 fits exactly (SE = 0).
    dof = n - 2
    se_slope = 0.0 if dof == 0 else float(np.sqrt(sse / dof / sxx))
    return slope, se_slope


def _clamp_bound(value: float) -> int:
    """Section 14.5.5: bounds clamped to [0, 999], rounded to whole days
    (float noise must never shave a day off a bound)."""
    return int(round(min(999.0, max(0.0, value))))


def estimate_rul(
    first_recorded_at: datetime | None,
    current_sample: HealthIndexSample | None,
    window_samples: list[HealthIndexSample],
    now: datetime | None = None,
) -> RulEstimate:
    """Pure Section 14 evaluation over one motor's HI history.

    Arguments:
      first_recorded_at: earliest non-inrush row timestamp (cold-start
          clock origin). ``None`` means the motor has no usable rows.
      current_sample: the most recent non-inrush row (Section 14.4.4
          current HI). ``None`` means no usable rows.
      window_samples: the non-inrush rows inside the trend window
          (last ``RUL.lookback_days`` days), ascending by time.
      now: evaluation instant; defaults to current UTC. Injected
          explicitly by tests for reproducibility.

    Returns the complete Section 14.6 estimate. No I/O, no state.
    """
    now = now or datetime.now(timezone.utc)

    # --- 14.3 cold start: < 24 h accumulated telemetry ---------------
    if first_recorded_at is None or current_sample is None:
        return RulEstimate(
            rul_status=RulStatus.CALCULATING_INSUFFICIENT_DATA,
            rul_days_estimate=MSG_INSUFFICIENT_DATA,
            slope_sigma_per_day=None,
            current_health_index_sigma=None,
            trend_window_days=None,
            rul_days_lower=None,
            rul_days_upper=None,
            trend_points=0,
            last_window_at=None,
        )
    accumulated_hours = (now - first_recorded_at).total_seconds() / 3600.0
    if accumulated_hours < RUL.cold_start_min_hours:
        return RulEstimate(
            rul_status=RulStatus.CALCULATING_INSUFFICIENT_DATA,
            rul_days_estimate=MSG_INSUFFICIENT_DATA,
            slope_sigma_per_day=None,
            current_health_index_sigma=current_sample.health_index_sigma,
            trend_window_days=None,
            rul_days_lower=None,
            rul_days_upper=None,
            trend_points=len(window_samples),
            last_window_at=current_sample.recorded_at,
        )

    # --- 14.4.2 trend-validity minimums ------------------------------
    span_days = (
        (window_samples[-1].recorded_at - window_samples[0].recorded_at).total_seconds()
        / _SECONDS_PER_DAY
        if len(window_samples) >= 2
        else 0.0
    )
    if (
        len(window_samples) < RUL.min_trend_points
        or span_days < RUL.min_trend_span_days
    ):
        return RulEstimate(
            rul_status=RulStatus.CALCULATING_BUILDING_TREND,
            rul_days_estimate=MSG_BUILDING_TREND,
            slope_sigma_per_day=None,
            current_health_index_sigma=current_sample.health_index_sigma,
            trend_window_days=span_days if span_days > 0 else None,
            rul_days_lower=None,
            rul_days_upper=None,
            trend_points=len(window_samples),
            last_window_at=current_sample.recorded_at,
        )

    # --- 14.4.3 deterministic OLS slope -------------------------------
    slope, se_slope = ols_slope_sigma_per_day(window_samples)
    current_hi = current_sample.health_index_sigma

    # --- 14.5.3 current HI already at/beyond the critical boundary ----
    if current_hi >= RUL.critical_failure_threshold_sigma:
        return RulEstimate(
            rul_status=RulStatus.CRITICAL_THRESHOLD_REACHED,
            rul_days_estimate=0,
            slope_sigma_per_day=round(slope, 6),
            current_health_index_sigma=current_hi,
            trend_window_days=round(span_days, 3),
            rul_days_lower=0,
            rul_days_upper=0,
            trend_points=len(window_samples),
            last_window_at=current_sample.recorded_at,
        )

    # --- 14.5.1 stable or improving -----------------------------------
    if slope <= 0.0:
        return RulEstimate(
            rul_status=RulStatus.STABLE,
            rul_days_estimate=MSG_STABLE,
            slope_sigma_per_day=round(slope, 6),
            current_health_index_sigma=current_hi,
            trend_window_days=round(span_days, 3),
            rul_days_lower=None,
            rul_days_upper=None,
            trend_points=len(window_samples),
            last_window_at=current_sample.recorded_at,
        )

    # --- 14.5.2 degrading: the deterministic formula -------------------
    remaining_sigma = RUL.critical_failure_threshold_sigma - current_hi
    rul_days = remaining_sigma / slope

    # --- 14.5.4 presentation cap ---------------------------------------
    if rul_days > RUL.presentation_cap_days:
        return RulEstimate(
            rul_status=RulStatus.STABLE,
            rul_days_estimate=MSG_STABLE,
            slope_sigma_per_day=round(slope, 6),
            current_health_index_sigma=current_hi,
            trend_window_days=round(span_days, 3),
            rul_days_lower=None,
            rul_days_upper=None,
            trend_points=len(window_samples),
            last_window_at=current_sample.recorded_at,
        )

    # --- 14.5.5 confidence bounds from the OLS slope standard error ----
    # SE == 0 (perfectly linear history) collapses both bounds onto the
    # estimate; the 999 fallback applies only when slope <= SE (slope
    # not resolvable from zero trend noise).
    lower = _clamp_bound(remaining_sigma / (slope + se_slope))
    upper = (
        _clamp_bound(remaining_sigma / (slope - se_slope))
        if slope > se_slope
        else RUL.presentation_cap_days
    )

    return RulEstimate(
        rul_status=RulStatus.DEGRADING,
        rul_days_estimate=int(round(rul_days)),
        slope_sigma_per_day=round(slope, 6),
        current_health_index_sigma=current_hi,
        trend_window_days=round(span_days, 3),
        rul_days_lower=lower,
        rul_days_upper=upper,
        trend_points=len(window_samples),
        last_window_at=current_sample.recorded_at,
    )


# ----------------------------------------------------------------------
# Service layer: one Supabase read + the pure evaluation (Section 14.2)
# ----------------------------------------------------------------------
async def assess_rul(
    repository: "SigmoRepository",
    motor_id: str,
    now: datetime | None = None,
) -> RulEstimate:
    """Assess one motor's RUL from its stored Supabase HI series.

    Reads the motor's non-inrush ``telemetry_features`` history exactly
    once (earliest row, most recent row, and the trend-window rows via
    ``SigmoRepository.fetch_health_index_history``) and evaluates the
    deterministic Section 14 protocol. Pure read path — no writes, no
    alerting side effects (14.7.2).
    """
    history = await repository.fetch_health_index_history(
        motor_id, lookback_days=RUL.lookback_days
    )
    return estimate_rul(
        first_recorded_at=history.first_recorded_at,
        current_sample=history.current_sample,
        window_samples=history.window_samples,
        now=now,
    )
