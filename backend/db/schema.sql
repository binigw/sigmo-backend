-- ====================================================================
-- Sigmo V2 — Supabase PostgreSQL Schema (SIGMO_RULES.md 4.2, 10, 11)
-- Run once against the external Supabase project (SQL Editor or CI).
-- NOTE: raw waveform arrays are NEVER stored (Section 10).
-- ====================================================================

CREATE TABLE IF NOT EXISTS motors (
    motor_id            TEXT PRIMARY KEY,
    mcc_panel_id        TEXT        NOT NULL,
    cabinet_number      TEXT        NOT NULL,
    line_zone_id        TEXT        NOT NULL,
    nameplate_kw        NUMERIC(8,2) NOT NULL CHECK (nameplate_kw > 0),
    rated_rpm           INTEGER     NOT NULL CHECK (rated_rpm > 0),
    drive_type          TEXT        NOT NULL CHECK (drive_type IN ('DOL','SOFT_STARTER','VFD')),
    commissioned_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    baseline_state      TEXT        NOT NULL DEFAULT 'COMMISSIONING'
                        CHECK (baseline_state IN
                          ('COMMISSIONING','HEALTHY_CONFIRMED',
                           'PRE_EXISTING_FAULT','POPULATION_TRACKED'))
);

-- Immutable population baseline derived from ZZU-MCC5 + IEEE PQD
-- healthy distributions (Cold Start Protocol step 1). Row updates are
-- blocked by trigger; superseding requires a new version row.
CREATE TABLE IF NOT EXISTS population_baseline (
    baseline_version    INTEGER     PRIMARY KEY,
    feature_name        TEXT        NOT NULL,
    healthy_mean        DOUBLE PRECISION NOT NULL,
    healthy_std         DOUBLE PRECISION NOT NULL CHECK (healthy_std >= 0),
    p05                 DOUBLE PRECISION NOT NULL,
    p95                 DOUBLE PRECISION NOT NULL,
    source_dataset      TEXT        NOT NULL CHECK (source_dataset IN ('ZZU_MCC5','IEEE_PQD','COMBINED')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (baseline_version, feature_name)
);

CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'population_baseline is immutable (SIGMO_RULES.md 11.1)';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS population_baseline_immutable ON population_baseline;
CREATE TRIGGER population_baseline_immutable
    BEFORE UPDATE OR DELETE ON population_baseline
    FOR EACH ROW EXECUTE FUNCTION forbid_mutation();

-- Phase-5 Dual-View asset extension (SIGMO_RULES.md 6.b): spare-part and
-- cost data required by the Technician payload. Nullable on purpose —
-- motors commissioned before Phase 5 keep working; unregistered values
-- surface as explicit "not registered" statuses, never fake numbers.
ALTER TABLE motors
    ADD COLUMN IF NOT EXISTS asset_tag TEXT,
    ADD COLUMN IF NOT EXISTS bearing_part_number TEXT,
    ADD COLUMN IF NOT EXISTS spare_capacitor_spec TEXT,
    ADD COLUMN IF NOT EXISTS downtime_cost_etb_per_hour NUMERIC(12,2)
        CHECK (downtime_cost_etb_per_hour IS NULL
               OR downtime_cost_etb_per_hour > 0);

-- Plant-wide business configuration (SIGMO_RULES.md 6.a financial
-- fields): customer-entered rates ONLY. The application never seeds
-- values — an unset key means the corresponding output field returns
-- an explicit "rate not configured" status instead of an invented
-- number (zero-fake-code directive).
CREATE TABLE IF NOT EXISTS plant_config (
    key         TEXT PRIMARY KEY,
    value       DOUBLE PRECISION NOT NULL,
    unit        TEXT NOT NULL,
    description TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Per-motor personalized baseline — only written after healthy
-- confirmation (Cold Start Protocol step 4). Permanently retained.
CREATE TABLE IF NOT EXISTS motor_baselines (
    motor_id            TEXT        NOT NULL REFERENCES motors(motor_id),
    feature_name        TEXT        NOT NULL,
    baseline_mean       DOUBLE PRECISION NOT NULL,
    baseline_std        DOUBLE PRECISION NOT NULL CHECK (baseline_std >= 0),
    zscore_alert_threshold DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    established_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (motor_id, feature_name)
);

-- Feature-extracted telemetry ONLY (no raw arrays). Subject to 90-day
-- cleanup when status = 'HEALTHY' (Section 10).
CREATE TABLE IF NOT EXISTS telemetry_features (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    motor_id            TEXT        NOT NULL REFERENCES motors(motor_id),
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    status              TEXT        NOT NULL CHECK (status IN ('HEALTHY','ANOMALY','INRUSH_SUPPRESSED')),
    fundamental_hz      DOUBLE PRECISION NOT NULL,
    rms_a               DOUBLE PRECISION NOT NULL,
    rms_b               DOUBLE PRECISION NOT NULL,
    rms_c               DOUBLE PRECISION NOT NULL,
    thd_percent_a       DOUBLE PRECISION NOT NULL,
    thd_percent_b       DOUBLE PRECISION NOT NULL,
    thd_percent_c       DOUBLE PRECISION NOT NULL,
    crest_factor_a      DOUBLE PRECISION NOT NULL,
    crest_factor_b      DOUBLE PRECISION NOT NULL,
    crest_factor_c      DOUBLE PRECISION NOT NULL,
    kurtosis_a          DOUBLE PRECISION NOT NULL,
    kurtosis_b          DOUBLE PRECISION NOT NULL,
    kurtosis_c          DOUBLE PRECISION NOT NULL,
    rotor_sideband_db_a DOUBLE PRECISION NOT NULL,
    rotor_sideband_db_b DOUBLE PRECISION NOT NULL,
    rotor_sideband_db_c DOUBLE PRECISION NOT NULL,
    envelope_peak_hz_a  DOUBLE PRECISION NOT NULL,
    envelope_peak_hz_b  DOUBLE PRECISION NOT NULL,
    envelope_peak_hz_c  DOUBLE PRECISION NOT NULL,
    unbalance_percent   DOUBLE PRECISION NOT NULL,
    zscore_max          DOUBLE PRECISION NOT NULL DEFAULT 0,
    -- v6c production wiring (Phase 4, 2026-09-25): the dominant-fault
    -- verdict of the ACTIVE classifier for the analyzed window. NULL
    -- for rows written before the wiring, when no model bundle could be
    -- loaded (degraded mode), or for INRUSH_SUPPRESSED windows (the
    -- classifier is trained on steady-state windows only).
    predicted_class     TEXT,
    model_confidence    DOUBLE PRECISION,
    model_version       TEXT
);
-- Idempotent migration for databases provisioned before the Phase-4
-- wiring (apply_schema runs at every boot; ADD COLUMN IF NOT EXISTS is
-- a no-op once present).
ALTER TABLE telemetry_features
    ADD COLUMN IF NOT EXISTS predicted_class TEXT,
    ADD COLUMN IF NOT EXISTS model_confidence DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS model_version TEXT;

CREATE INDEX IF NOT EXISTS idx_telemetry_motor_time
    ON telemetry_features (motor_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_status_time
    ON telemetry_features (status, recorded_at);

-- Fault / maintenance history — PERMANENTLY retained (Section 10).
CREATE TABLE IF NOT EXISTS fault_logs (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    motor_id            TEXT        NOT NULL REFERENCES motors(motor_id),
    detected_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    taxonomy_code       TEXT        NOT NULL,
    urgency_stage       SMALLINT    NOT NULL CHECK (urgency_stage BETWEEN 1 AND 4),
    model_confidence    DOUBLE PRECISION NOT NULL CHECK (model_confidence BETWEEN 0 AND 1),
    population_sigma    DOUBLE PRECISION NOT NULL,
    absolute_threshold_breached BOOLEAN NOT NULL,
    trend_confirmed     BOOLEAN     NOT NULL,
    four_signal_confirmed BOOLEAN   GENERATED ALWAYS AS
        (population_sigma > 5.0 AND absolute_threshold_breached
         AND model_confidence > 0.70 AND trend_confirmed) STORED,
    physically_verified BOOLEAN     NOT NULL DEFAULT FALSE,
    verified_by         TEXT,
    spectral_evidence   JSONB       NOT NULL,
    resolved_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_fault_logs_motor ON fault_logs (motor_id, detected_at DESC);

-- Operator alert dismissals — live alert hygiene for the dashboard
-- (the mutable sibling of the immutable fault_logs above). Alerts are
-- DERIVED on read (stage >= 2 or an active fault verdict) and never
-- stored; a dismissal hides one live alert while its condition
-- persists and is auto-forgotten once that condition clears, so a
-- re-triggering or escalating fault alerts again.
CREATE TABLE IF NOT EXISTS alert_dismissals (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    motor_id            TEXT        NOT NULL,
    alert_key           TEXT        NOT NULL,
    dismissed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    dismissed_by        TEXT        NOT NULL DEFAULT 'dashboard',
    UNIQUE (motor_id, alert_key)
);
CREATE INDEX IF NOT EXISTS idx_alert_dismissals_motor
    ON alert_dismissals (motor_id, dismissed_at DESC);

-- 90-day retention for routine HEALTHY telemetry only. Baselines and
-- fault logs are never touched by this routine (Section 10).
CREATE OR REPLACE FUNCTION cleanup_healthy_telemetry() RETURNS INTEGER AS $$
DECLARE
    removed INTEGER;
BEGIN
    DELETE FROM telemetry_features
    WHERE status = 'HEALTHY'
      AND recorded_at < now() - INTERVAL '90 days';
    GET DIAGNOSTICS removed = ROW_COUNT;
    RETURN removed;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------------
-- ROW-LEVEL SECURITY (Supabase Security Advisor; applied 2026-09-30)
--
-- Architecture: the FastAPI backend connects as `postgres` (BYPASSRLS via
-- DATABASE_URL), so ALL backend reads/writes are unaffected. The dashboard
-- browser uses supabase-js ONLY for auth, the personnel directory and its
-- own user_roles row (policies below / created at project setup). Every
-- telemetry/motor/fault/alert table is therefore DENY-BY-DEFAULT for
-- anon/authenticated: no policies + default grants revoked. Data flows
-- exclusively through the Sigmo V2 API (X-API-Key scopes).
-- Idempotent: safe on every boot.
-- ---------------------------------------------------------------------------
ALTER TABLE public.motors               ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.motor_baselines      ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.telemetry_features   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.fault_logs           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.population_baseline  ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.plant_config         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alert_dismissals     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.personnel            ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_roles           ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON public.motors              FROM anon, authenticated;
REVOKE ALL ON public.motor_baselines     FROM anon, authenticated;
REVOKE ALL ON public.telemetry_features  FROM anon, authenticated;
REVOKE ALL ON public.fault_logs          FROM anon, authenticated;
REVOKE ALL ON public.population_baseline FROM anon, authenticated;
REVOKE ALL ON public.plant_config        FROM anon, authenticated;
REVOKE ALL ON public.alert_dismissals    FROM anon, authenticated;
