"""Sigmo V2 — MCSA DSP Pipeline.

Implements the full backend signal chain mandated by SIGMO_RULES.md
Section 4.1 for three-phase current waveforms (I_a, I_b, I_c):

    1. PyWavelets (db4) wavelet denoising
    2. Dynamic fundamental-frequency tracking (DOL / Soft-Starter / VFD)
    3. 16,384-point Zoom FFT via complex demodulation
    4. Hilbert envelope demodulation (bearing-band analysis)
    5. Feature extraction: RMS, THD, Crest Factor, Kurtosis,
       rotor-fault sideband ratio, phase-current unbalance
    6. Startup inrush suppression gate (Section 9)

Raw sample arrays live ONLY in RAM for the duration of processing
(Section 10) — nothing in this module performs persistence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from scipy.signal import butter, get_window, hilbert, sosfiltfilt

from ..config import DSP


# --------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PhaseFeatures:
    """Feature vector extracted from a single phase current waveform."""

    rms_a: float
    peak_a: float
    crest_factor: float
    kurtosis: float
    thd_percent: float
    fundamental_hz: float
    fundamental_amplitude: float
    rotor_sideband_ratio_db: float
    envelope_peak_hz: float
    envelope_peak_amplitude: float
    zoom_fft_peak_hz: tuple[float, ...]
    zoom_fft_peak_amplitude: tuple[float, ...]


@dataclass(frozen=True)
class ThreePhaseFeatures:
    """Aggregated three-phase analysis result for one acquisition window."""

    phase_a: PhaseFeatures
    phase_b: PhaseFeatures
    phase_c: PhaseFeatures
    current_unbalance_percent: float
    mean_fundamental_hz: float
    inrush_detected: bool


# --------------------------------------------------------------------------
# Stage 1 — Wavelet denoising
# --------------------------------------------------------------------------
def wavelet_denoise(signal: np.ndarray) -> np.ndarray:
    """Denoise a raw current waveform with db4 wavelet soft-thresholding.

    Uses the universal threshold (VisuShrink) with a robust MAD noise
    estimate taken from the finest detail band, applied to detail
    coefficients only so the 50/60 Hz fundamental and its sidebands are
    preserved.
    """
    signal = np.asarray(signal, dtype=np.float64)
    max_level = pywt.dwt_max_level(signal.size, pywt.Wavelet(DSP.wavelet).dec_len)
    level = min(DSP.wavelet_level, max_level)
    coeffs = pywt.wavedec(signal, DSP.wavelet, level=level)
    detail_finest = coeffs[-1]
    sigma = np.median(np.abs(detail_finest - np.median(detail_finest))) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(signal.size))
    denoised = [coeffs[0]] + [
        pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
    ]
    out = pywt.waverec(denoised, DSP.wavelet)
    return out[: signal.size]


# --------------------------------------------------------------------------
# Stage 2 — Dynamic fundamental tracking (DOL / Soft-Starter / VFD)
# --------------------------------------------------------------------------
def track_fundamental(signal: np.ndarray, fs: float) -> tuple[float, float]:
    """Locate the supply fundamental within the configured search band.

    Returns (frequency_hz, amplitude_a) using a Hann-windowed FFT with
    parabolic interpolation on the log-magnitude peak for sub-bin
    accuracy. The wide 20-90 Hz search band supports VFD operation.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    win = get_window("hann", x.size)
    spectrum = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    mag = np.abs(spectrum) * 2.0 / np.sum(win)

    lo, hi = DSP.fundamental_search_hz
    band = (freqs >= lo) & (freqs <= hi)
    band_idx = np.flatnonzero(band)
    k = band_idx[np.argmax(mag[band_idx])]

    # Parabolic interpolation for sub-bin frequency estimation
    if 0 < k < mag.size - 1 and mag[k - 1] > 0 and mag[k + 1] > 0:
        a, b, c = np.log(mag[k - 1]), np.log(mag[k]), np.log(mag[k + 1])
        denom = a - 2.0 * b + c
        delta = 0.0 if denom == 0.0 else 0.5 * (a - c) / denom
    else:
        delta = 0.0
    bin_width = fs / x.size
    return float(freqs[k] + delta * bin_width), float(mag[k])


# --------------------------------------------------------------------------
# Stage 3 — 16,384-point Zoom FFT via complex demodulation
# --------------------------------------------------------------------------
def zoom_fft(
    signal: np.ndarray,
    fs: float,
    center_hz: float,
    span_hz: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """High-resolution spectrum around `center_hz` (complex demodulation).

    The signal is mixed down to baseband, low-pass filtered, decimated,
    and transformed with a 16,384-point FFT, yielding millihertz-class
    resolution needed to resolve 2sf rotor-bar sidebands around the
    fundamental.

    Returns (frequencies_hz, amplitude_a) covering center_hz +/- span/2.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    t = np.arange(n) / fs

    # Mix to baseband
    analytic = x * np.exp(-2j * np.pi * center_hz * t)

    # Anti-alias low-pass at span/2, then decimate
    decimation = max(1, int(fs / (2.5 * span_hz)))
    cutoff = (span_hz / 2.0) / (fs / 2.0)
    sos = butter(6, cutoff, btype="low", output="sos")
    filtered = sosfiltfilt(sos, analytic)
    decimated = filtered[::decimation]
    fs_dec = fs / decimation

    # Zero-pad / truncate to the mandated 16,384-point transform.
    # Blackman-Harris: -92 dB sidelobes prevent fundamental leakage from
    # masking -40..-60 dB rotor-fault sidebands (Hann's -31 dB sidelobes
    # are insufficient at 2sf offsets with 3.2 s records).
    nfft = DSP.zoom_fft_points
    win = get_window("blackmanharris", decimated.size)
    spectrum = np.fft.fftshift(np.fft.fft(decimated * win, n=nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs_dec)) + center_hz
    mag = np.abs(spectrum) * 2.0 / np.sum(win)

    keep = (freqs >= center_hz - span_hz / 2.0) & (freqs <= center_hz + span_hz / 2.0)
    return freqs[keep], mag[keep]


def rotor_sideband_ratio_db(
    zoom_freqs: np.ndarray,
    zoom_mag: np.ndarray,
    fundamental_hz: float,
) -> float:
    """Ratio (dB) of the strongest 2sf sideband pair to the fundamental.

    Broken rotor bars produce sidebands at f1 +/- 2*s*f1. We search the
    configured slip-frequency offset band on both sides of the
    fundamental and report the larger sideband relative to the carrier.
    Healthy motors typically show < -50 dB; > -45 dB indicates
    developing rotor faults.
    """
    lo, hi = DSP.rotor_sideband_search_hz
    fund_idx = int(np.argmin(np.abs(zoom_freqs - fundamental_hz)))
    fund_amp = float(zoom_mag[fund_idx])
    if fund_amp <= 0.0:
        return -120.0

    lower = (zoom_freqs >= fundamental_hz - hi) & (zoom_freqs <= fundamental_hz - lo)
    upper = (zoom_freqs >= fundamental_hz + lo) & (zoom_freqs <= fundamental_hz + hi)
    candidates = []
    for band in (lower, upper):
        idx = np.flatnonzero(band)
        if idx.size:
            candidates.append(float(np.max(zoom_mag[idx])))
    if not candidates:
        return -120.0
    sideband = max(candidates)
    if sideband <= 0.0:
        return -120.0
    return float(20.0 * np.log10(sideband / fund_amp))


# --------------------------------------------------------------------------
# Stage 4 — Hilbert envelope demodulation (bearing signatures)
# --------------------------------------------------------------------------
def hilbert_envelope_spectrum(
    signal: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope spectrum of the bearing-fault band.

    Band-pass the current in the configured high-frequency band where
    bearing impacts modulate the current, take the Hilbert analytic
    envelope, remove its DC, and return the envelope's spectrum where
    BPFO/BPFI/FTF/BSF characteristic frequencies appear.
    """
    x = np.asarray(signal, dtype=np.float64)
    lo, hi = DSP.envelope_band_hz
    nyq = fs / 2.0
    sos = butter(4, [lo / nyq, min(hi / nyq, 0.99)], btype="band", output="sos")
    band = sosfiltfilt(sos, x - np.mean(x))
    envelope = np.abs(hilbert(band))
    envelope -= np.mean(envelope)
    win = get_window("hann", envelope.size)
    spectrum = np.fft.rfft(envelope * win)
    freqs = np.fft.rfftfreq(envelope.size, d=1.0 / fs)
    mag = np.abs(spectrum) * 2.0 / np.sum(win)
    keep = freqs <= 500.0  # bearing defect frequencies live below 500 Hz
    return freqs[keep], mag[keep]


# --------------------------------------------------------------------------
# Stage 5 — Scalar feature extraction
# --------------------------------------------------------------------------
def compute_thd_percent(
    signal: np.ndarray, fs: float, fundamental_hz: float
) -> float:
    """Total Harmonic Distortion (%) up to the configured harmonic count."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    win = get_window("hann", x.size)
    spectrum = np.fft.rfft(x * win)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    mag = np.abs(spectrum) * 2.0 / np.sum(win)
    bin_width = fs / x.size
    half_window = max(2, int(round(1.0 / bin_width)))  # +/-1 Hz search

    def peak_near(f: float) -> float:
        k = int(round(f / bin_width))
        if k >= mag.size:
            return 0.0
        s = slice(max(0, k - half_window), min(mag.size, k + half_window + 1))
        return float(np.max(mag[s]))

    v1 = peak_near(fundamental_hz)
    if v1 <= 0.0:
        return 0.0
    harmonics = [
        peak_near(h * fundamental_hz)
        for h in range(2, DSP.harmonics_for_thd + 1)
        if h * fundamental_hz < fs / 2.0
    ]
    return float(100.0 * np.sqrt(np.sum(np.square(harmonics))) / v1)


def kurtosis_excess(signal: np.ndarray) -> float:
    """Excess kurtosis (Fisher). Healthy sinusoidal current ~= -1.5."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    var = np.mean(x**2)
    if var == 0.0:
        return 0.0
    return float(np.mean(x**4) / var**2 - 3.0)


def phase_current_unbalance_percent(rms_values: tuple[float, float, float]) -> float:
    """NEMA-style unbalance: max deviation from mean / mean * 100."""
    arr = np.asarray(rms_values, dtype=np.float64)
    mean = float(np.mean(arr))
    if mean == 0.0:
        return 0.0
    return float(np.max(np.abs(arr - mean)) / mean * 100.0)


def detect_inrush(signal: np.ndarray) -> bool:
    """Startup inrush gate (SIGMO_RULES.md Section 9).

    Splits the window into eight segments; if any early segment RMS
    exceeds the final segment RMS by the configured ratio, the window is
    flagged as an inrush transient and must not raise fault alarms.
    """
    x = np.asarray(signal, dtype=np.float64)
    segments = np.array_split(x - np.mean(x), 8)
    rms = np.array([np.sqrt(np.mean(s**2)) for s in segments])
    steady = rms[-1]
    if steady == 0.0:
        return True
    return bool(np.max(rms[:4]) / steady > DSP.inrush_rms_ratio_threshold)


def _top_spectral_peaks(
    freqs: np.ndarray, mag: np.ndarray, count: int = 5
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the `count` strongest local maxima of a spectrum."""
    if mag.size < 3:
        return tuple(), tuple()
    interior = np.flatnonzero((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])) + 1
    if interior.size == 0:
        return tuple(), tuple()
    order = interior[np.argsort(mag[interior])[::-1][:count]]
    return (
        tuple(float(freqs[i]) for i in order),
        tuple(float(mag[i]) for i in order),
    )


# --------------------------------------------------------------------------
# Full per-phase and three-phase pipeline
# --------------------------------------------------------------------------
def analyze_phase(signal: np.ndarray, fs: float) -> PhaseFeatures:
    """Run the complete single-phase MCSA chain and return its features."""
    denoised = wavelet_denoise(signal)
    ac = denoised - np.mean(denoised)

    rms = float(np.sqrt(np.mean(ac**2)))
    peak = float(np.max(np.abs(ac)))
    crest = peak / rms if rms > 0.0 else 0.0

    fund_hz, fund_amp = track_fundamental(ac, fs)
    thd = compute_thd_percent(ac, fs, fund_hz)
    kurt = kurtosis_excess(ac)

    zf, zm = zoom_fft(ac, fs, center_hz=fund_hz)
    sideband_db = rotor_sideband_ratio_db(zf, zm, fund_hz)
    peak_hz, peak_amp = _top_spectral_peaks(zf, zm)

    ef, em = hilbert_envelope_spectrum(ac, fs)
    valid = ef > 2.0  # ignore near-DC residue
    if np.any(valid):
        k = np.flatnonzero(valid)[np.argmax(em[valid])]
        env_hz, env_amp = float(ef[k]), float(em[k])
    else:
        env_hz, env_amp = 0.0, 0.0

    return PhaseFeatures(
        rms_a=rms,
        peak_a=peak,
        crest_factor=float(crest),
        kurtosis=kurt,
        thd_percent=thd,
        fundamental_hz=fund_hz,
        fundamental_amplitude=fund_amp,
        rotor_sideband_ratio_db=sideband_db,
        envelope_peak_hz=env_hz,
        envelope_peak_amplitude=env_amp,
        zoom_fft_peak_hz=peak_hz,
        zoom_fft_peak_amplitude=peak_amp,
    )


def analyze_three_phase(
    i_a: np.ndarray,
    i_b: np.ndarray,
    i_c: np.ndarray,
    fs: float = DSP.sampling_rate_hz,
) -> ThreePhaseFeatures:
    """Analyze one synchronized three-phase acquisition window.

    Raw arrays are consumed from RAM and never persisted (Section 10).
    """
    lengths = {np.asarray(i_a).size, np.asarray(i_b).size, np.asarray(i_c).size}
    if len(lengths) != 1:
        raise ValueError(
            f"Phase arrays must be equal length, got {sorted(lengths)}"
        )
    inrush = detect_inrush(np.asarray(i_a, dtype=np.float64))

    pa = analyze_phase(np.asarray(i_a, dtype=np.float64), fs)
    pb = analyze_phase(np.asarray(i_b, dtype=np.float64), fs)
    pc = analyze_phase(np.asarray(i_c, dtype=np.float64), fs)

    unbalance = phase_current_unbalance_percent((pa.rms_a, pb.rms_a, pc.rms_a))
    mean_fund = float(np.mean([pa.fundamental_hz, pb.fundamental_hz, pc.fundamental_hz]))

    return ThreePhaseFeatures(
        phase_a=pa,
        phase_b=pb,
        phase_c=pc,
        current_unbalance_percent=unbalance,
        mean_fundamental_hz=mean_fund,
        inrush_detected=inrush,
    )
