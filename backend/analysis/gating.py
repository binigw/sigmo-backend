"""Sigmo V2 — Stage-2 anomaly gating for the ingestion path.

Implements the deterministic gating signals available at Step 2 of the
build (SIGMO_RULES.md Section 11.3 absolute thresholds + Section 11
z-score deviation vs stored baselines). The trained Isolation Forest /
XGBoost gating (Section 5) is a later step and plugs in behind the same
interface; nothing here fakes model output.

A window is marked:
  INRUSH_SUPPRESSED  — startup transient detected; never alarms (Section 9)
  ANOMALY            — any absolute IEEE/ISO threshold breached, OR
                       z-score vs personalized baseline exceeds threshold
  HEALTHY            — otherwise
"""
from __future__ import annotations

from dataclasses import dataclass

from ..dsp.pipeline import ThreePhaseFeatures

# Absolute thresholds mandated by SIGMO_RULES.md Section 11.3
# (IEEE 1159 power quality, ISO/NEMA practice for current signatures).
THD_MAX_PERCENT = 5.0
UNBALANCE_MAX_PERCENT = 2.0
CREST_FACTOR_MIN = 1.30
CREST_FACTOR_MAX = 1.55  # 1.40-1.45 nominal; +/- band avoids hair-trigger
ROTOR_SIDEBAND_ALERT_DB = -45.0  # 2sf sidebands above this = developing fault


@dataclass(frozen=True)
class GatingResult:
    status: str  # HEALTHY | ANOMALY | INRUSH_SUPPRESSED
    breaches: tuple[str, ...]
    zscore_max: float


def _zscore_max(
    features: ThreePhaseFeatures,
    baseline: dict[str, tuple[float, float, float]],
) -> float:
    """Largest |z| across baselined features; 0.0 when no baseline exists
    (motor still in COMMISSIONING / POPULATION_TRACKED state)."""
    observed = {
        "rms_a": features.phase_a.rms_a,
        "rms_b": features.phase_b.rms_a,
        "rms_c": features.phase_c.rms_a,
        "thd_percent_a": features.phase_a.thd_percent,
        "thd_percent_b": features.phase_b.thd_percent,
        "thd_percent_c": features.phase_c.thd_percent,
        "unbalance_percent": features.current_unbalance_percent,
        "rotor_sideband_db_a": features.phase_a.rotor_sideband_ratio_db,
        "rotor_sideband_db_b": features.phase_b.rotor_sideband_ratio_db,
        "rotor_sideband_db_c": features.phase_c.rotor_sideband_ratio_db,
    }
    worst = 0.0
    for name, value in observed.items():
        stats = baseline.get(name)
        if stats is None:
            continue
        mean, std, _threshold = stats
        if std <= 0.0:
            continue
        z = abs(value - mean) / std
        if z > worst:
            worst = z
    return worst


def evaluate_window(
    features: ThreePhaseFeatures,
    baseline: dict[str, tuple[float, float, float]],
) -> GatingResult:
    """Gate one completed analysis window."""
    if features.inrush_detected:
        return GatingResult("INRUSH_SUPPRESSED", tuple(), 0.0)

    breaches: list[str] = []

    thd_max = max(
        features.phase_a.thd_percent,
        features.phase_b.thd_percent,
        features.phase_c.thd_percent,
    )
    if thd_max > THD_MAX_PERCENT:
        breaches.append(f"THD {thd_max:.2f}% > {THD_MAX_PERCENT}% (IEEE 1159)")

    if features.current_unbalance_percent > UNBALANCE_MAX_PERCENT:
        breaches.append(
            f"Current unbalance {features.current_unbalance_percent:.2f}% "
            f"> {UNBALANCE_MAX_PERCENT}% (NEMA MG-1)"
        )

    for label, phase in (
        ("A", features.phase_a),
        ("B", features.phase_b),
        ("C", features.phase_c),
    ):
        if not CREST_FACTOR_MIN <= phase.crest_factor <= CREST_FACTOR_MAX:
            breaches.append(
                f"Crest factor phase {label} {phase.crest_factor:.3f} outside "
                f"[{CREST_FACTOR_MIN}, {CREST_FACTOR_MAX}]"
            )
        if phase.rotor_sideband_ratio_db > ROTOR_SIDEBAND_ALERT_DB:
            breaches.append(
                f"Rotor 2sf sideband phase {label} "
                f"{phase.rotor_sideband_ratio_db:.1f} dB > "
                f"{ROTOR_SIDEBAND_ALERT_DB} dB"
            )

    z_max = _zscore_max(features, baseline)
    z_breached = False
    for name, (_mean, std, threshold) in baseline.items():
        if std <= 0.0:
            continue
        # threshold defaults to 5.0 sigma per schema (Section 11.6 signal 1)
        if z_max > threshold:
            z_breached = True
            break
    if z_breached:
        breaches.append(f"Baseline deviation {z_max:.1f} sigma exceeds threshold")

    status = "ANOMALY" if breaches else "HEALTHY"
    return GatingResult(status, tuple(breaches), z_max)
