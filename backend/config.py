"""Sigmo V2 — Central configuration.

All credentials are read exclusively from environment variables (Secrets),
per SIGMO_RULES.md Section 4.2. No credentials are ever hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DSPConfig:
    """Hard DSP constants mandated by SIGMO_RULES.md Section 4.1."""

    sampling_rate_hz: float = 10_240.0          # ESP32 I2S DMA rate
    frame_lengths: tuple[int, ...] = (2048, 4096)  # accepted raw frames
    analysis_window_samples: int = 32_768        # accumulated per phase (3.2 s)
    zoom_fft_points: int = 16_384                # mandated Zoom FFT length
    wavelet: str = "db4"                         # Daubechies-4 denoising
    wavelet_level: int = 5
    # DOL/Soft-Starter/VFD: must cover deep VFD turndown — ZZU-MCC5
    # recordings at 1000 rpm run a 16.7 Hz supply fundamental.
    fundamental_search_hz: tuple[float, float] = (12.0, 95.0)
    harmonics_for_thd: int = 12
    # 2sf sideband search band. Lower bound must clear the Blackman-
    # Harris main lobe half-width (4/T = 1.25 Hz at 3.2 s windows) to
    # avoid fundamental leakage reading as a false sideband.
    rotor_sideband_search_hz: tuple[float, float] = (1.5, 6.0)
    envelope_band_hz: tuple[float, float] = (500.0, 4000.0)     # bearing band
    inrush_rms_ratio_threshold: float = 1.8      # startup transient gate


@dataclass(frozen=True)
class RULConfig:
    """Deterministic RUL via Slope Extrapolation constants (SIGMO_RULES.md
    Section 14). Every value is a hard rulebook constant, not a tuning
    knob: the 5.0-sigma critical threshold is deliberately identical to
    the Section 11.6 signal-1 population-deviation threshold and the
    motor_baselines.zscore_alert_threshold default, so "RUL exhausted"
    and "4-signal alert" reference one and the same boundary."""

    cold_start_min_hours: float = 24.0        # Section 14.3: < 24 h -> insufficient data
    lookback_days: int = 7                    # Section 14.4.1: historical window (3-7 allowed)
    min_trend_span_days: float = 3.0          # Section 14.4.2: window time span minimum
    min_trend_points: int = 12                # Section 14.4.2: non-inrush window count minimum
    critical_failure_threshold_sigma: float = 5.0  # Section 14.5
    presentation_cap_days: int = 999          # Section 14.5.4: > 999 reported as Stable


@dataclass(frozen=True)
class DatabaseConfig:
    """Supabase PostgreSQL — the ONLY permitted persistence layer."""

    dsn: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", ""))
    supabase_url: str = field(default_factory=lambda: os.environ.get("SUPABASE_URL", ""))
    min_pool_size: int = 1
    max_pool_size: int = 5
    healthy_log_interval_s: int = 1800           # 15-30 min healthy cadence
    anomaly_log_interval_s: int = 60             # 1 min escalated cadence
    retention_days: int = 90                     # healthy telemetry cleanup

    def require_dsn(self) -> str:
        if not self.dsn:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. Sigmo V2 "
                "persists exclusively to external Supabase PostgreSQL "
                "(SIGMO_RULES.md Section 4.2). Set DATABASE_URL to your "
                "Supabase connection string in Secrets."
            )
        return self.dsn


@dataclass(frozen=True)
class APIConfig:
    """HTTP surface settings. CORS origins come from the environment
    (Secrets) as a comma-separated list. The MVP default ``*`` allows the
    external Replit frontend to fetch immediately; production locks this
    down by setting CORS_ALLOW_ORIGINS (e.g.
    ``https://sigmo-dashboard.replit.app``). allow_credentials is False
    by design: the wildcard origin and credentials are mutually
    exclusive under the CORS specification."""

    cors_allow_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")
        if o.strip()
    ))
    # API keys (Section 7 hardening): comma-separated "key:scope" pairs
    # from the SIGMO_API_KEYS secret. Parsed by backend.api.auth.load_keys
    # at startup; raw values never leave this field.
    api_keys_raw: str = field(default_factory=lambda: (
        os.environ.get("SIGMO_API_KEYS", "")
    ))
    cors_allow_methods: tuple[str, ...] = ("GET", "POST", "PUT", "OPTIONS")
    cors_allow_headers: tuple[str, ...] = ("*",)


@dataclass(frozen=True)
class ViewConfig:
    """Dual-View policy defaults (Phase 5, SIGMO_RULES.md 6 / 11 / 14).

    Every value here is a documented commissioning default that the
    customer can override per plant via the ``plant_config`` Supabase
    table (admin endpoint); the tables store ONLY explicit overrides.
    Nothing here is a measured quantity — measured values always come
    from telemetry or the motor_assets registry.

    Stage thresholds anchor to the rulebook: Stage 4 boundary == the
    5.0-sigma CRITICAL_FAILURE_THRESHOLD (Section 14.5) == the 4-signal
    population-deviation threshold (Section 11.6); Stage 2 is the
    early-fault boundary (Section 10 adaptive logging escalates at
    "early Stage 2+").
    """

    stage2_z: float = 2.0            # early-fault boundary (sigma)
    stage3_z: float = 3.5            # severe boundary (sigma)
    stage4_z: float = 5.0            # == RUL.critical_failure_threshold_sigma
    # Maximum allowable operating hours before mandatory trip/shutdown.
    max_operating_hours: dict[int, int | None] = field(default_factory=lambda: {
        1: None,   # healthy — unlimited
        2: 720,    # 30 days: plan replacement at next scheduled stop
        3: 72,     # 3 days: schedule immediate intervention
        4: 0,      # trip/shutdown now
    })
    # Expected unplanned-outage hours used by the financial risk exposure
    # (Section 6.a): exposure = sum(expected hours x hourly downtime cost).
    expected_outage_hours: dict[int, float] = field(default_factory=lambda: {
        1: 0.0, 2: 8.0, 3: 24.0, 4: 48.0,
    })
    # Energy-loss model basis: NEMA MG-1 style unbalance derating
    # (1% loss per 1% current unbalance above the 1% free band) and
    # THD^2 copper-loss scaling; PF term applies only when the customer
    # registers a measured plant power factor.
    unbalance_free_band_percent: float = 1.0
    unbalance_loss_per_percent: float = 0.01
    default_operating_hours_per_month: float = 720.0  # 24 h x 30 d


DSP = DSPConfig()
RUL = RULConfig()
DB = DatabaseConfig()
API = APIConfig()
VIEW = ViewConfig()
