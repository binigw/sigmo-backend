"""Sigmo V2 — ML feature contracts and unified feature extraction.

This module is the single source of truth for:

  * the 13-class fault taxonomy (``CLASS_NAMES``)
  * the ordered model input vector (``FEATURE_NAMES``, 117 features)
  * unified DSP feature extraction shared by the offline trainers
    (ZZU-MCC5 motor windows, SEED/IEEE PQD grid signals)

v6 (bearing-ceiling campaign) changes vs v5 — see models/OPTIMIZATION_REPORT.md
"V6 ADDENDUM A1" for the measured evidence:
  * the fixed 1.5-3.5 kHz HF demodulation band is replaced by a 9-band
    kurtogram filterbank (750 Hz - 6 kHz: 7 adjacent 750-Hz narrow bands
    + 2 wide half-range cross-bands); the per-window band is the argmax
    of Hilbert-envelope kurtosis, chosen AFTER excluding any band that
    overlaps the PWM switching-carrier cluster (strongest 1.8-6.2 kHz
    spectral line >= 8x the band median floor, +/- 300 Hz).  On
    VFD-instrumented rigs the raw kurtogram is trapped by the periodic
    carrier pulses; the exclusion is what makes selection trustworthy
    (v6a vs v6b ablation).
  * five new features: hfenv_band_hz (selected band centre) and the
    defect-comb structure set hfenv_opk1_sb1x_eng / hfenv_pk2_order /
    hfenv_pk2_rel / hfenv_pk2_sb1x_eng (energy sideband ratios
    mean(A(c-1X)^2, A(c+1X)^2)/A(c)^2 clipped to [0, 25], plus the
    guarded 2nd envelope-spectrum peak outside +/-0.5 orders of pk1 and
    its +/-1x sidebands).  Inner-race defect combs rotate with the shaft
    (sidebands present); outer-race combs are stationary (absent).

v4 (mechanical-fault optimization) changes vs v3:
  * envelope analysis is computed in the SHAFT-ORDER domain
    (per-window shaft-speed estimate with slip-aware fallback),
    making features invariant to the VFD speed ramps present in
    ZZU-MCC5 "speed_circulation" recordings (fundamental 8-50 Hz);
  * the 11 coarse envelope bands are replaced by a high-resolution
    fingerprint: 24 narrow bands (0.5x .. 12.0x shaft order, 0.5x
    step) plus 24 matched local-SNR bands (amplitude vs. local noise
    floor) targeted at bearing BPFO/BPFI/BSF energy, which lives at
    non-integer shaft orders and was invisible to the coarse grid;
  * explicit stator-current sideband combs at f1 +/- k*f_shaft and
    2*f1 +/- k*f_shaft (k = 1, 2) for unbalance / misalignment
    discrimination;
  * THD is bounded to harmonics 2..40 (v3 leaked switching-noise
    energy into the ratio, e.g. THD 87% on clean VFD windows).

Feature availability matrix (NaN means "not applicable to source"):
  * ZZU-MCC5 3-phase windows : all 117 features
  * IEEE_PQD / SEED_PQD      : single-channel features only
    (fundamental, crest, kurtosis, THD, harmonic ratios, interharmonic
    ratio, HF energy, cycle-RMS stats when >= 3 supply cycles,
    rms_pu).  Motor-only features (envelope, sidebands, sequence
    components, rotor sidebands) are NaN by construction.
XGBoost handles NaN natively, and the trainer never imputes.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as _sig
from scipy import stats

# ---------------------------------------------------------------------------
# Taxonomy (order is the model's class index — DO NOT REORDER)
# v6c-13 (2026-09-25): SHAFT_MISALIGNMENT restored from the vetted KAIST
# varying-load corpus (ztmf3m7h5x).  Its recording campaigns were not
# speed-matched (f1 operating-point confound, d'~1.8) — see
# scripts/datasets.manifest.json and OPTIMIZATION_REPORT.md section I;
# Phase 3 includes an f1-sensitivity audit for this class.
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "HEALTHY",
    "BEARING_OUTER_RACE",
    "BEARING_INNER_RACE",
    "BEARING_BALL",
    "SHAFT_MISALIGNMENT",
    "MECH_UNBALANCE_ECCENTRICITY",
    "BROKEN_ROTOR_BAR",
    "STATOR_WINDING_INTERTURN",
    "PHASE_CURRENT_UNBALANCE",
    "VOLTAGE_SAG_SWELL",
    "CAPACITOR_BANK_FAILURE",
    "CAPACITOR_SWITCHING_TRANSIENT",
    "HARMONICS_HIGH_THD",
]

# ---------------------------------------------------------------------------
# Feature grid constants
# ---------------------------------------------------------------------------
HARMONIC_ORDERS: tuple[int, ...] = (2, 3, 4, 5, 7)
THD_MAX_HARMONIC: int = 40
FUNDAMENTAL_MIN_HZ: float = 5.0
FUNDAMENTAL_MAX_HZ: float = 130.0
ROTOR_SIDEBAND_HZ: float = 8.0          # search half-width around f1
ENVELOPE_SNR_CLIP_DB: tuple[float, float] = (-20.0, 80.0)

# High-resolution envelope order grid (shaft orders).
ENVELOPE_ORDER_GRID: tuple[float, ...] = tuple(0.5 * i for i in range(1, 25))  # 0.5 .. 12.0
_ENVELOPE_BAND_HALF: float = 0.25        # +/- half-width of each narrow band
_ENVELOPE_FLOOR_HALF: float = 1.5        # local floor window (orders)
_ENVELOPE_FLOOR_GUARD: float = 0.375     # exclusion around the band centre

# Shaft-speed estimation: envelope peak search window relative to f1, with
# a VFD/DOL slip fallback when no 1x shaft component is measurable.
_SHAFT_SEARCH: tuple[float, float] = (0.90, 0.995)
_SHAFT_FALLBACK_RATIO: float = 0.97      # typical slip ~3% (documented estimate)
_SHAFT_LOCK_MIN_RATIO: float = 2.5       # peak must exceed local floor x2.5

# --- v6 PWM-safe kurtogram filterbank (bearing demodulation) -------------
# 9 candidate bands spanning 750 Hz - 6 kHz: 7 adjacent 750-Hz narrow
# bands plus 2 wide half-range cross-bands (kurtogram practice: wide
# bands catch impulsive content spread across neighbouring narrow
# bands).  The v6b corpus selected-band median was 2,625 Hz (narrow band
# 3) with a healthy spread of 750 Hz - 5.6 kHz — selection is
# data-driven per window, never pinned.
_KURTOGRAM_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (750.0, 1500.0), (1500.0, 2250.0), (2250.0, 3000.0), (3000.0, 3750.0),
    (3750.0, 4500.0), (4500.0, 5250.0), (5250.0, 6000.0),
    (750.0, 3000.0), (3000.0, 6000.0),
)
# PWM switching-carrier trap (VFD rigs): periodic kHz carrier pulses look
# maximally "impulsive" to a raw kurtogram, so an unguarded argmax would
# select the carrier band every time.  The carrier is detected as the
# strongest spectral line in 1.8-6.2 kHz that (a) towers >= 8x over that
# band's median floor AND (b) dominates >= 3x every rival line outside a
# +/-100 Hz adjacency guard (which only absorbs the carrier's own f1/2f1
# sidebands).  Condition (b) is what keeps a strong bearing-resonance
# comb from masquerading as a carrier on DOL/soft-starter rigs — its
# sibling defect lines (defect-rate spacing) sit within 2x of the max.
# Candidate bands overlapping carrier +/- 300 Hz are excluded BEFORE
# maximisation (V6 ADDENDUM A1).
_CARRIER_SEARCH_HZ: tuple[float, float] = (1800.0, 6200.0)
_CARRIER_MEDIAN_RATIO: float = 8.0
_CARRIER_DOMINANCE_RATIO: float = 3.0
_CARRIER_RIVAL_GUARD_HZ: float = 100.0
_CARRIER_EXCLUDE_HZ: float = 300.0
_SB_ENG_CLIP: tuple[float, float] = (0.0, 25.0)   # sideband energy-ratio clip
_PK_GUARD_ORDERS: float = 0.5                     # 2nd-peak exclusion guard

ISO_MOTOR_FEATURES: tuple[str, ...] = (
    "fundamental_hz", "crest_factor_max", "thd_percent_max",
    "negative_seq_ratio", "unbalance_percent", "hf_energy_ratio_max",
    "env_kurt", "env_crest", "env_opk1_rel", "slip_ratio",
    "sb1x_db_max", "eband_1p0x", "eband_2p0x",
)


def _grid_names(prefix: str) -> list[str]:
    out = []
    for g in ENVELOPE_ORDER_GRID:
        out.append(f"{prefix}_{str(g).replace('.', 'p')}x")
    return out


_BASE_FEATURES: list[str] = [
    "fundamental_hz",
    "crest_factor_max",
    "kurtosis_max",
    "thd_percent_max",
    "h2_ratio_max",
    "h3_ratio_max",
    "h4_ratio_max",
    "h5_ratio_max",
    "h7_ratio_max",
    "interharmonic_ratio_max",
    "hf_energy_ratio_max",
    "cycle_rms_min_ratio",
    "cycle_rms_max_ratio",
    "cycle_rms_cv",
    "rotor_sideband_db_max",
    "negative_seq_ratio",
    "unbalance_percent",
    "rms_cv",
    "rms_pu",
]

_MOTOR_FEATURES: list[str] = [
    "slip_ratio",
    "rpm_est",
    "env_kurt",
    "env_crest",
    "env_opk1_order",
    "env_opk1_rel",
    "env_beat_hz",
    "env_beat_rel",
    "hfenv_kurt",
    "hfenv_crest",
    "hfenv_opk1_order",
    "hfenv_opk1_rel",
    "hfenv_opk1_sb1x_rel",
    "hfenv_opk1_sb1x_eng",
    "hfenv_pk2_order",
    "hfenv_pk2_rel",
    "hfenv_pk2_sb1x_eng",
    "hfenv_band_hz",
    "hfenv_opk2_rel",
    "hfenv_opk3_rel",
    "hfenv_1x_rel",
    "hfenv_2x_rel",
    "sb1x_db_max",
    "sb2x_db_max",
    "sbh2_1x_db_max",
    "sbh2_2x_db_max",
]

FEATURE_NAMES: list[str] = (
    _BASE_FEATURES
    + _MOTOR_FEATURES
    + _grid_names("eband")
    + _grid_names("esnr")
    + _grid_names("eratio")
)

assert len(FEATURE_NAMES) == 19 + 26 + 24 + 48, len(FEATURE_NAMES)  # 117

# Metadata columns written by the builders (NOT model inputs).
META_COLUMNS: list[str] = [
    "source", "file", "label", "severity", "rpm", "torque_nm", "window_idx",
]

PQD_NOMINAL_RMS_PU: float = 1.0 / np.sqrt(2.0)   # unit-magnitude sine RMS


# ---------------------------------------------------------------------------
# Low-level spectral helpers
# ---------------------------------------------------------------------------
def _amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-sided, Hann-windowed, amplitude-scaled FFT of a real signal."""
    n = x.shape[0]
    win = np.hanning(n)
    xw = (x - float(np.mean(x))) * win
    spec = np.abs(np.fft.rfft(xw)) * (2.0 / float(np.sum(win)))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec[0] = 0.0  # DC removed analytically as well
    return freqs, spec


def _parabolic(freqs: np.ndarray, amps: np.ndarray, idx: int) -> tuple[float, float]:
    """Sub-bin peak refinement around local maximum ``idx``."""
    if idx <= 0 or idx >= amps.size - 1:
        return float(freqs[idx]), float(amps[idx])
    y0, y1, y2 = amps[idx - 1], amps[idx], amps[idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-18:
        return float(freqs[idx]), float(amps[idx])
    shift = 0.5 * (y0 - y2) / denom
    shift = float(np.clip(shift, -1.0, 1.0))
    df = float(freqs[1] - freqs[0])
    amp = y1 - 0.25 * (y0 - y2) * shift
    return float(freqs[idx] + shift * df), float(max(amp, 0.0))


def _peak_in_range(freqs: np.ndarray, amps: np.ndarray, f_lo: float, f_hi: float):
    """Strongest parabolic-refined peak in [f_lo, f_hi], or None."""
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(mask):
        return None
    idx_local = int(np.argmax(amps[mask]))
    idx = int(np.flatnonzero(mask)[idx_local])
    return _parabolic(freqs, amps, idx)


def _amp_near(freqs: np.ndarray, amps: np.ndarray, f: float, half_width: float) -> float:
    """Peak amplitude within ``f +/- half_width`` (0 when out of band)."""
    if f - half_width < 0.0 or f + half_width > freqs[-1]:
        return 0.0
    pk = _peak_in_range(freqs, amps, f - half_width, f + half_width)
    return 0.0 if pk is None else float(pk[1])


def _dominant_hf_carrier(x: np.ndarray, fs: float) -> float | None:
    """Dominant PWM switching-carrier frequency, or None.

    The carrier is the strongest spectral line inside the 1.8-6.2 kHz
    search band that (a) towers at least ``_CARRIER_MEDIAN_RATIO`` (8x)
    over that band's median amplitude floor and (b) dominates at least
    ``_CARRIER_DOMINANCE_RATIO`` (3x) every rival line outside a
    +/-100 Hz adjacency guard.  Condition (b) prevents a strong bearing
    resonance comb from masquerading as a carrier: its sibling defect
    lines at defect-rate spacing sit within 2x of the comb maximum,
    whereas a switching carrier is the dominant HF line by >9 dB.
    DOL / soft-starter recordings carry no such line and return None
    (no exclusion is applied).
    """
    freqs, amps = _amplitude_spectrum(x, fs)
    mask = (freqs >= _CARRIER_SEARCH_HZ[0]) & (freqs <= _CARRIER_SEARCH_HZ[1])
    if not np.any(mask):
        return None
    sel = np.flatnonzero(mask)
    floor = float(np.median(amps[sel]))
    if floor <= 1e-12:
        return None
    top = sel[int(np.argmax(amps[sel]))]
    if float(amps[top]) < _CARRIER_MEDIAN_RATIO * floor:
        return None
    rival_mask = mask & (np.abs(freqs - freqs[top]) > _CARRIER_RIVAL_GUARD_HZ)
    rival = float(np.max(amps[rival_mask])) if np.any(rival_mask) else 0.0
    if float(amps[top]) < _CARRIER_DOMINANCE_RATIO * rival:
        return None
    f_c, _ = _parabolic(freqs, amps, top)
    if not _CARRIER_SEARCH_HZ[0] <= f_c <= _CARRIER_SEARCH_HZ[1]:
        return None
    return float(f_c)


def _envelope_spectrum_kurtosis(env: np.ndarray, fs: float) -> float:
    """Spectral kurtosis of a Hilbert envelope: the excess kurtosis of the
    envelope's (Hann-windowed, amplitude-scaled) spectrum.

    A band whose envelope carries a strong line comb — a bearing defect
    train at BPFO/BPFI order — has a spiky envelope spectrum and scores
    high; a band of demodulated noise has a flat spectrum and scores
    near zero.  This is the statistic the v6 kurtogram maximises, and it
    is exactly what the downstream defect-comb features
    (hfenv_opk1_*, hfenv_pk2_*) consume.
    """
    _freqs, amps = _amplitude_spectrum(env, fs)
    if amps.size < 4:
        return -np.inf
    return float(stats.kurtosis(amps[1:], fisher=True))


def _select_resonance_band(x: np.ndarray, fs: float) -> tuple[float, float] | None:
    """Per-window kurtogram selection of the bearing demodulation band.

    Scores every valid candidate band by the spectral kurtosis of its
    Hilbert envelope (``_envelope_spectrum_kurtosis`` — spikiness of the
    envelope spectrum, i.e. defect-comb prominence over the demodulated
    floor) and returns the (lo, hi) of the argmax.  PWM safety runs
    FIRST: any band that overlaps the detected carrier cluster (carrier
    +/- 300 Hz) is excluded before maximisation — an unguarded kurtogram
    is trapped by the carrier sideband cluster (its periodic envelope AM
    is a strong spectral line, so the carrier band out-scores every
    defect band; V6 ADDENDUM A1: raw-kurtogram v6a vs PWM-safe v6b
    ablation).

    Returns None when no candidate band fits under Nyquist, or when the
    carrier cluster excludes every candidate (geometrically unreachable
    with 9 bands spanning 5.25 kHz against a 600-Hz cluster).
    """
    nyq_cap = fs / 2.0 * 0.98
    candidates = [b for b in _KURTOGRAM_BANDS_HZ if b[1] <= nyq_cap]
    if not candidates:
        return None
    carrier = _dominant_hf_carrier(x, fs)
    if carrier is not None:
        clo, chi = carrier - _CARRIER_EXCLUDE_HZ, carrier + _CARRIER_EXCLUDE_HZ
        candidates = [b for b in candidates if b[1] < clo or b[0] > chi]
        if not candidates:
            return None
    best: tuple[float, float] | None = None
    best_k = -np.inf
    for lo, hi in candidates:
        try:
            sos = _sig.butter(4, [lo, hi], btype="band", fs=fs, output="sos")
            xf = _sig.sosfiltfilt(sos, x)
            if not np.all(np.isfinite(xf)):
                continue
            env = np.abs(_sig.hilbert(xf))
            env = env - float(np.mean(env))
            k = _envelope_spectrum_kurtosis(env, fs)
        except (ValueError, FloatingPointError):
            continue
        if k > best_k:
            best_k, best = k, (float(lo), float(hi))
    return best


def _harmonics_block(freqs: np.ndarray, amps: np.ndarray, f1: float, a1: float):
    """Bounded THD (h=2..40) and per-order ratios for HARMONIC_ORDERS."""
    df = float(freqs[1] - freqs[0])
    half = max(2.0 * df, 0.015 * f1)
    sq_sum = 0.0
    h_amps: dict[int, float] = {}
    for h in range(2, THD_MAX_HARMONIC + 1):
        fh = h * f1
        if fh + half >= freqs[-1]:
            break
        ah = _amp_near(freqs, amps, fh, half)
        sq_sum += ah * ah
        if h in HARMONIC_ORDERS:
            h_amps[h] = ah
    thd = 100.0 * float(np.sqrt(sq_sum)) / a1
    ratios = {h: h_amps.get(h, 0.0) / a1 for h in HARMONIC_ORDERS}
    return min(thd, 500.0), ratios


def _masked_interharmonic(freqs: np.ndarray, amps: np.ndarray, f1: float, a1: float):
    """Largest spectral line outside all harmonic neighbourhoods, rel. to f1."""
    df = float(freqs[1] - freqs[0])
    guard = int(np.ceil(max(2.0 * df, 0.02 * f1) / df))
    residual = amps.copy()
    h = 1
    while h * f1 <= freqs[-1]:
        c = int(round(h * f1 / df))
        lo = max(0, c - guard)
        hi = min(residual.size, c + guard + 1)
        residual[lo:hi] = 0.0
        h += 1
    residual[freqs < 3.0] = 0.0
    return float(np.max(residual)) / a1


def _cycle_rms_stats(x: np.ndarray, fs: float, f1: float):
    """Per-supply-cycle RMS statistics (needs >= 3 full cycles)."""
    spc = int(round(fs / f1))
    if spc <= 0:
        return None
    n_cycles = int(x.shape[0] / spc)
    if n_cycles < 3:
        return None
    n_cycles = min(n_cycles, 64)
    cycles = x[: n_cycles * spc].reshape(n_cycles, spc)
    rms = np.sqrt(np.mean(cycles * cycles, axis=1))
    mean_rms = float(np.mean(rms))
    if mean_rms <= 0.0:
        return None
    return (
        float(np.min(rms)) / mean_rms,
        float(np.max(rms)) / mean_rms,
        float(np.std(rms)) / mean_rms,
    )


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------
def nan_feature_dict() -> dict[str, float]:
    return {name: float("nan") for name in FEATURE_NAMES}


def extract_signal_features(
    x: np.ndarray, fs: float, nominal_rms: float | None = None
) -> dict[str, float]:
    """Single-channel feature block (ZZU phase legs AND PQD grid signals).

    Three-phase-only features (sequence components, envelope order
    fingerprint, mechanical sidebands) are left NaN; callers with 3-phase
    data must use :func:`extract_window_features` instead.
    """
    row = nan_feature_dict()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.shape[0] < 64 or not np.all(np.isfinite(x)):
        return row
    rms = float(np.sqrt(np.mean(x * x)))
    if rms <= 0.0:
        return row

    freqs, amps = _amplitude_spectrum(x, fs)
    pk = _peak_in_range(freqs, amps, FUNDAMENTAL_MIN_HZ, FUNDAMENTAL_MAX_HZ)
    if pk is None or pk[1] <= 0.0:
        return row
    f1, a1 = pk

    row["fundamental_hz"] = float(np.clip(f1, 0.0, 200.0))
    row["crest_factor_max"] = float(np.max(np.abs(x)) / rms)
    row["kurtosis_max"] = float(stats.kurtosis(x, fisher=True))
    thd, ratios = _harmonics_block(freqs, amps, f1, a1)
    row["thd_percent_max"] = thd
    row["h2_ratio_max"] = ratios[2]
    row["h3_ratio_max"] = ratios[3]
    row["h4_ratio_max"] = ratios[4]
    row["h5_ratio_max"] = ratios[5]
    row["h7_ratio_max"] = ratios[7]
    row["interharmonic_ratio_max"] = _masked_interharmonic(freqs, amps, f1, a1)
    hf_lo = max(10.0 * f1, 1000.0)
    hi_mask = freqs >= hf_lo
    f1_band = _amp_near(freqs, amps, f1, max(2.0 * (freqs[1] - freqs[0]), 0.015 * f1))
    denom = f1_band * f1_band
    row["hf_energy_ratio_max"] = (
        float(np.sum(amps[hi_mask] * amps[hi_mask])) / denom if denom > 0 else 0.0
    )
    cyc = _cycle_rms_stats(x - np.mean(x), fs, f1)
    if cyc is not None:
        row["cycle_rms_min_ratio"], row["cycle_rms_max_ratio"], row["cycle_rms_cv"] = cyc
    if nominal_rms is not None:
        row["rms_pu"] = rms / float(nominal_rms)
    return row


def _estimate_shaft_freq(
    efreqs: np.ndarray, eamps: np.ndarray, f1: float
) -> tuple[float, bool]:
    """Shaft frequency from the Park-modulus envelope spectrum.

    Locks onto the 1x shaft component expected inside
    ``_SHAFT_SEARCH`` x f1.  When no component clears the local-noise
    criterion (perfectly balanced windows), falls back to the
    documented slip assumption ``_SHAFT_FALLBACK_RATIO`` x f1.
    """
    lo, hi = _SHAFT_SEARCH[0] * f1, _SHAFT_SEARCH[1] * f1
    pk = _peak_in_range(efreqs, eamps, lo, hi)
    if pk is not None:
        f_sh, a_sh = pk
        span = (efreqs >= 0.85 * lo) & (efreqs <= 1.05 * hi)
        floor = float(np.median(eamps[span])) if np.any(span) else 0.0
        if floor > 0.0 and a_sh >= _SHAFT_LOCK_MIN_RATIO * floor:
            return float(np.clip(f_sh, 2.0, f1)), True
    return float(_SHAFT_FALLBACK_RATIO * f1), False


def extract_window_features(
    currents: np.ndarray, fs: float, nominal_rms: float | None = None
) -> dict[str, float]:
    """Full 77-feature vector from one 3-phase current window.

    ``currents`` shape (n, 3) = [Ia, Ib, Ic].  All heavy DSP (Park-modulus
    envelope, shaft-order normalization, sideband combs) runs here.
    """
    row = nan_feature_dict()
    cur = np.asarray(currents, dtype=np.float64)
    if cur.ndim != 2 or cur.shape[1] != 3 or cur.shape[0] < 256:
        return row
    if not np.all(np.isfinite(cur)):
        return row

    # --- per-phase base block; "*_max" columns aggregate the 3 legs -------
    per_phase = [extract_signal_features(cur[:, i], fs, nominal_rms) for i in range(3)]
    if all(np.isnan(p["fundamental_hz"]) for p in per_phase):
        return row
    agg_max = (
        "crest_factor_max", "kurtosis_max", "thd_percent_max",
        "h2_ratio_max", "h3_ratio_max", "h4_ratio_max", "h5_ratio_max",
        "h7_ratio_max", "interharmonic_ratio_max", "hf_energy_ratio_max",
    )
    for name in agg_max:
        vals = [p[name] for p in per_phase]
        row[name] = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else float("nan")
    for name in ("cycle_rms_min_ratio", "cycle_rms_max_ratio", "cycle_rms_cv"):
        vals = [p[name] for p in per_phase if not np.isnan(p[name])]
        if vals:
            row[name] = float(np.max(vals))
    if nominal_rms is not None:
        rms_ph = [float(np.sqrt(np.mean(cur[:, i] * cur[:, i]))) for i in range(3)]
        row["rms_pu"] = float(np.mean(rms_ph)) / float(nominal_rms)

    # Reference phase spectrum for fundamental + sidebands -----------------
    freqs, amps = _amplitude_spectrum(cur[:, 0], fs)
    pk = _peak_in_range(freqs, amps, FUNDAMENTAL_MIN_HZ, FUNDAMENTAL_MAX_HZ)
    if pk is None or pk[1] <= 0.0:
        return row
    f1, a1 = pk
    row["fundamental_hz"] = float(np.clip(f1, 0.0, 200.0))

    # --- sequence components at f1 ---------------------------------------
    n = cur.shape[0]
    t = np.arange(n, dtype=np.float64) / fs
    ref = np.exp(-2j * np.pi * f1 * t)
    phasors = np.array([2.0 * np.mean((cur[:, i] - cur[:, i].mean()) * ref) for i in range(3)])
    alpha = np.exp(2j * np.pi / 3.0)
    i_pos = (phasors[0] + alpha * phasors[1] + alpha * alpha * phasors[2]) / 3.0
    i_neg = (phasors[0] + alpha * alpha * phasors[1] + alpha * phasors[2]) / 3.0
    i_mag = sorted((abs(i_pos), abs(i_neg)))
    if i_mag[1] > 1e-12:
        # Smaller-sequence / larger-sequence: unbalance magnitude is the
        # physically relevant quantity, and this form is immune to phase-leg
        # wiring swaps at the MCC panel (field-hardening).
        nsr = i_mag[0] / i_mag[1]
        row["negative_seq_ratio"] = float(min(nsr, 1.0))
        row["unbalance_percent"] = float(min(100.0 * nsr, 100.0))
    rms_ph = np.sqrt(np.mean(cur * cur, axis=0))
    if float(np.mean(rms_ph)) > 0.0:
        row["rms_cv"] = float(np.std(rms_ph) / np.mean(rms_ph))

    # --- Park-modulus envelope -------------------------------------------
    ia, ib, ic = cur[:, 0], cur[:, 1], cur[:, 2]
    x_alpha = ia - 0.5 * (ib + ic)
    x_beta = (np.sqrt(3.0) / 2.0) * (ib - ic)
    pm = np.sqrt(x_alpha * x_alpha + x_beta * x_beta)
    pm_c = pm - float(np.mean(pm))
    env_rms_raw = float(np.sqrt(np.mean(pm_c * pm_c)))
    if env_rms_raw <= 1e-12:
        return row

    # Supply-ripple removal: the Park envelope of VFD-fed currents is
    # dominated by ripple harmonics at k*f1 (2x, 6x...).  Under speed
    # variation these components alias across shaft-order bins and wreck
    # speed-invariance, so they are least-squares notched first.  What
    # remains (pm_c) carries the mechanical order content.
    win_t = np.arange(cur.shape[0], dtype=np.float64) / fs
    ripple_cols: list[np.ndarray] = []
    k = 2
    while k * f1 < 0.5 * fs and k <= 12:
        ripple_cols.append(np.cos(2.0 * np.pi * k * f1 * win_t))
        ripple_cols.append(np.sin(2.0 * np.pi * k * f1 * win_t))
        k += 1
    ripple_A = np.stack(ripple_cols, axis=1)
    coef, _, _, _ = np.linalg.lstsq(ripple_A, pm, rcond=None)
    pm_c = pm - ripple_A @ coef
    pm_c = pm_c - float(np.mean(pm_c))
    env_rms = float(np.sqrt(np.mean(pm_c * pm_c)))
    if env_rms <= 1e-12:
        return row

    efreqs, eamps = _amplitude_spectrum(pm_c, fs)
    f_shaft, locked = _estimate_shaft_freq(efreqs, eamps, f1)
    row["slip_ratio"] = float(np.clip(f_shaft / f1, 0.0, 1.0))
    row["rpm_est"] = float(np.clip(60.0 * f_shaft, 0.0, 6000.0))
    row["env_kurt"] = float(stats.kurtosis(pm_c, fisher=True))
    row["env_crest"] = float(np.max(np.abs(pm_c)) / env_rms)

    orders = efreqs / f_shaft
    order_mask = (orders >= 0.4) & (orders <= 12.5)
    if np.any(order_mask):
        sel = np.flatnonzero(order_mask)
        top = sel[int(np.argmax(eamps[sel]))]
        o_ref, a_ref = _parabolic(efreqs / f_shaft, eamps, top)
        row["env_opk1_order"] = float(np.clip(o_ref, 0.0, 15.0))
        row["env_opk1_rel"] = float(min(a_ref / env_rms, 60.0))

    # Slip-frequency beat (broken-rotor-bar AM signature): at low slip the
    # f1 +/- 2s*f1 sidebands hide inside the 1 s resolution skirt of the
    # fundamental, but their beat survives in the envelope at ~s*f1 Hz.
    beat_hi = min(0.35 * f_shaft, 3.0)
    beat_mask = (efreqs >= 0.3) & (efreqs <= beat_hi)
    if np.any(beat_mask):
        bsel = np.flatnonzero(beat_mask)
        btop = bsel[int(np.argmax(eamps[bsel]))]
        bf, ba = _parabolic(efreqs, eamps, btop)
        row["env_beat_hz"] = float(np.clip(bf, 0.0, 5.0))
        row["env_beat_rel"] = float(min(ba / env_rms, 60.0))

    # --- high-resolution envelope order fingerprint -----------------------
    grid = np.asarray(ENVELOPE_ORDER_GRID, dtype=np.float64)
    for g, eb, es in zip(grid, _grid_names("eband"), _grid_names("esnr")):
        band = (orders >= g - _ENVELOPE_BAND_HALF) & (orders <= g + _ENVELOPE_BAND_HALF)
        if not np.any(band):
            continue
        a_band = float(np.max(eamps[band]))
        floor_span = (
            (orders >= g - _ENVELOPE_FLOOR_HALF) & (orders <= g - _ENVELOPE_FLOOR_GUARD)
        ) | (
            (orders >= g + _ENVELOPE_FLOOR_GUARD) & (orders <= g + _ENVELOPE_FLOOR_HALF)
        )
        floor = float(np.median(eamps[floor_span])) if np.any(floor_span) else 0.0
        row[eb] = float(min(a_band / env_rms, 60.0))
        if floor > 0.0:
            snr_db = 20.0 * np.log10((a_band + 1e-12) / floor)
            row[es] = float(np.clip(snr_db, *ENVELOPE_SNR_CLIP_DB))

    # Local shape-contrast ratios: band amplitude relative to its two
    # neighbouring band pairs.  Envelope absolute levels shrink strongly
    # with shaft speed on this VFD rig; shape ratios are speed-robust.
    eb_vals = np.asarray(
        [row.get(k, float("nan")) for k in _grid_names("eband")], dtype=np.float64
    )
    for gi, g in enumerate(grid):
        neigh = []
        for off in (-2, -1, 1, 2):
            j = gi + off
            if 0 <= j < len(grid) and np.isfinite(eb_vals[j]):
                neigh.append(eb_vals[j])
        ern = _grid_names("eratio")[gi]
        if len(neigh) >= 2 and np.isfinite(eb_vals[gi]):
            med = float(np.median(neigh))
            if med > 1e-12:
                row[ern] = float(np.clip(eb_vals[gi] / med, 0.0, 20.0))

    # --- HF-resonance demodulated envelope (bearing-impulse targeted) ------
    # Bearing impacts ring structural/cabling resonances in the kHz range;
    # demodulating that band exposes BPFO/BPFI-order content that the smooth
    # 6x supply ripple buries in the baseband Park envelope.
    # v6: the demodulation band is selected per window by the PWM-safe
    # kurtogram (_select_resonance_band) instead of the fixed 1.5-3.5 kHz
    # band, and the selection + defect-comb structure contribute five new
    # features (hfenv_band_hz, hfenv_opk1_sb1x_eng, hfenv_pk2_*).
    if fs / 2.0 * 0.98 >= _KURTOGRAM_BANDS_HZ[0][1]:
        try:
            x_sum = ia + ib + ic
            band_sel = _select_resonance_band(x_sum, fs)
            if band_sel is None:
                raise FloatingPointError("no valid kurtogram band")
            band_lo, band_hi = band_sel
            row["hfenv_band_hz"] = float(0.5 * (band_lo + band_hi))
            sos = _sig.butter(
                4, [band_lo, band_hi], btype="band", fs=fs, output="sos"
            )
            x_filt = _sig.sosfiltfilt(sos, x_sum)
            if not np.all(np.isfinite(x_filt)):
                raise FloatingPointError("non-finite HF envelope input")
            hf_env = np.abs(_sig.hilbert(x_filt))
            hf_env = hf_env - float(np.mean(hf_env))
            hf_rms = float(np.sqrt(np.mean(hf_env * hf_env)))
            if hf_rms > 1e-12:
                hfreqs, hamps = _amplitude_spectrum(hf_env, fs)
                horders = hfreqs / f_shaft
                row["hfenv_kurt"] = float(stats.kurtosis(hf_env, fisher=True))
                row["hfenv_crest"] = float(np.max(np.abs(hf_env)) / hf_rms)
                hmask = (horders >= 0.4) & (horders <= 12.5)
                if np.any(hmask):
                    hsel = np.flatnonzero(hmask)
                    htop = hsel[int(np.argmax(hamps[hsel]))]
                    ho_ref, ha_ref = _parabolic(hfreqs / f_shaft, hamps, htop)
                    row["hfenv_opk1_order"] = float(np.clip(ho_ref, 0.0, 15.0))
                    row["hfenv_opk1_rel"] = float(min(ha_ref / hf_rms, 60.0))
                    # Defect-comb structure: inner-race defects rotate with the
                    # shaft, so their defect order carries +/-1x shaft AM
                    # sidebands; outer-race defects are stationary (clean
                    # comb, multiple harmonics, no 1x sidebands).  These three
                    # features encode exactly that asymmetry.
                    def _hf_at(order_target: float) -> float:
                        band = (horders >= order_target - 0.25) & (
                            horders <= order_target + 0.25
                        )
                        return float(min(np.max(hamps[band]) / hf_rms, 60.0)) if np.any(band) else float("nan")

                    side_a = _hf_at(ho_ref - 1.0) if ho_ref - 1.0 >= 0.4 else float("nan")
                    side_b = _hf_at(ho_ref + 1.0) if ho_ref + 1.0 <= 12.5 else float("nan")
                    sides = [s for s in (side_a, side_b) if s == s]
                    if sides:
                        row["hfenv_opk1_sb1x_rel"] = float(np.mean(sides))
                    # v6 energy sideband ratio: mean(A(c-1X)^2, A(c+1X)^2)
                    # / A(c)^2, clipped to [0, 25].  Inner-race defect combs
                    # rotate with the shaft (sidebands present); outer-race
                    # combs are stationary (absent).  The hf_rms
                    # normalisation of _hf_at cancels inside the ratio.
                    def _sb_eng(order_target: float) -> float:
                        a0 = _hf_at(order_target)
                        if a0 != a0 or a0 <= 1e-12:
                            return float("nan")
                        pair: list[float] = []
                        for off in (-1.0, 1.0):
                            t = order_target + off
                            if 0.4 <= t <= 12.5:
                                v = _hf_at(t)
                                if v == v:
                                    pair.append(v * v)
                        if not pair:
                            return float("nan")
                        return float(
                            np.clip(np.mean(pair) / (a0 * a0), *_SB_ENG_CLIP)
                        )

                    eng = _sb_eng(ho_ref)
                    if eng == eng:
                        row["hfenv_opk1_sb1x_eng"] = eng

                    # v6 guarded 2nd defect peak: the strongest envelope-
                    # spectrum peak OUTSIDE +/-0.5 orders of pk1 and its
                    # +/-1x sidebands, so the comb's own structure cannot
                    # shadow an independent second mechanism.
                    guard = (
                        (np.abs(horders - ho_ref) <= _PK_GUARD_ORDERS)
                        | (np.abs(horders - (ho_ref - 1.0)) <= _PK_GUARD_ORDERS)
                        | (np.abs(horders - (ho_ref + 1.0)) <= _PK_GUARD_ORDERS)
                    )
                    gmask = hmask & ~guard
                    if np.any(gmask):
                        gsel = np.flatnonzero(gmask)
                        gtop = gsel[int(np.argmax(hamps[gsel]))]
                        go_ref, ga_ref = _parabolic(hfreqs / f_shaft, hamps, gtop)
                        row["hfenv_pk2_order"] = float(np.clip(go_ref, 0.0, 15.0))
                        row["hfenv_pk2_rel"] = float(min(ga_ref / hf_rms, 60.0))
                        geng = _sb_eng(go_ref)
                        if geng == geng:
                            row["hfenv_pk2_sb1x_eng"] = geng
                    if 2.0 * ho_ref <= 12.5:
                        row["hfenv_opk2_rel"] = _hf_at(2.0 * ho_ref)
                    if 3.0 * ho_ref <= 12.5:
                        row["hfenv_opk3_rel"] = _hf_at(3.0 * ho_ref)
                for g_ord, col in ((1.0, "hfenv_1x_rel"), (2.0, "hfenv_2x_rel")):
                    gband = (horders >= g_ord - 0.25) & (horders <= g_ord + 0.25)
                    if np.any(gband):
                        row[col] = float(min(np.max(hamps[gband]) / hf_rms, 60.0))
        except (ValueError, FloatingPointError):
            pass
        # Any failure leaves the hfenv_* features NaN (XGBoost-native).

    # --- mechanical sideband combs on the phase currents -------------------
    def _sideband_db(base: float, delta: float) -> float:
        # Each AM sideband (base-delta, base+delta) is evaluated on its own;
        # under VFD speed-lock the lower f1-f_shaft sideband sits at ~0 Hz and
        # is skipped, while the measurable upper one still reports.  Centres
        # aliasing onto supply harmonics (f_shaft ~= f1) are rejected — an
        # honest NaN beats an aliased 0 dB "detection".
        alias_guard = max(3.0, 0.12 * f1)
        centres = []
        for c in (base - delta, base + delta):
            if not (3.0 <= c <= freqs[-1]):
                continue
            if any(abs(c - k * f1) < alias_guard for k in range(1, 7)):
                continue
            centres.append(c)
        if not centres:
            return float("nan")
        best = 0.0
        hw = max(1.0, 0.08 * delta)
        for i in range(3):
            _, a_ph = _amplitude_spectrum(cur[:, i], fs)
            for f_c in centres:
                best = max(best, _amp_near(freqs, a_ph, f_c, hw))
        return float(np.clip(20.0 * np.log10((best + 1e-12) / a1), -120.0, 0.0))

    row["sb1x_db_max"] = _sideband_db(f1, f_shaft)
    row["sb2x_db_max"] = _sideband_db(f1, 2.0 * f_shaft)
    row["sbh2_1x_db_max"] = _sideband_db(2.0 * f1, f_shaft)
    row["sbh2_2x_db_max"] = _sideband_db(2.0 * f1, 2.0 * f_shaft)

    # --- rotor-bar sidebands around f1 (search +/-ROTOR_SIDEBAND_HZ) -------
    best_sb = 0.0
    for i in range(3):
        _, a_ph = _amplitude_spectrum(cur[:, i], fs)
        for side in (-1.0, 1.0):
            lo = f1 + (0.75 if side > 0 else -ROTOR_SIDEBAND_HZ)
            hi = f1 + (ROTOR_SIDEBAND_HZ if side > 0 else -0.75)
            if lo < 3.0 or hi > freqs[-1] or lo >= hi:
                continue
            pk_sb = _peak_in_range(freqs, a_ph, lo, hi)
            if pk_sb is not None:
                best_sb = max(best_sb, pk_sb[1])
    if best_sb > 0.0:
        row["rotor_sideband_db_max"] = float(
            np.clip(20.0 * np.log10(best_sb / a1), -120.0, 0.0)
        )
    return row
