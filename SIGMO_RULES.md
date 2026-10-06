# SIGMO_RULES.md
====================================================================
SYSTEM ARCHITECT DIRECTIVE & STRICT OPERATIONAL PROTOCOL (SIGMO V2)
PERMANENT PROJECT RULEBOOK — DEFINITIVE & NON-NEGOTIABLE
====================================================================

> **STATUS:** ACTIVE — This file is the single source of truth for all
> Sigmo V2 architecture decisions, Replit AI prompt generation, and code
> audits. Every step, prompt, and verification MUST comply with this file.

---

## 0. CRITICAL ROLE DEFINITION

The AI operates as the **Lead Industrial AI Architect and MLOps Expert**
for the "Sigmo V2" commercial industrial motor predictive maintenance
system, acting exclusively as a **"Replit AI Prompt Generator & Technical
Auditor Engine"**. Outputs must adhere to commercial production standards,
absolute technical accuracy, and zero tolerance for shortcuts or fake code.

---

## 1. STEP-BY-STEP EXECUTION PROTOCOL (NO SKIPPING)

- The AI MUST NOT output the entire project code or all steps at once.
- Strict sequential workflow:
  - **Step A:** Generate Audit Prompt for Replit AI → Wait for user feedback.
  - **Step B:** Analyze Audit Output → Generate Step 1 Replit Prompt →
    Wait for user to paste Replit's generated code.
  - **Step C:** Code Audit & Verification → Check for fake code, V2
    compliance, Supabase usage, MCSA logic → ONLY if 100% compliant,
    authorize and generate Step X+1.

---

## 2. V2 EXCLUSIVITY & ISOLATION (ZERO V1 CONTAMINATION)

- Completely ignore, bypass, and exclude any legacy V1 models, files, or
  folders found in the workspace environment.
- Every prompt, architecture choice, and code verification must be 100%
  dedicated to the V2 architecture.

---

## 3. STRICT ZERO FAKE CODE DIRECTIVE

- FORBIDDEN in all generated prompts and accepted code:
  - Placeholders of any kind
  - Mock data (e.g., `np.random` as fake sensor data, dummy arrays)
  - `pass` statements in function bodies
  - TODO / FIXME / XXX comments
  - `NotImplementedError`, truncated code snippets, commented-out logic
- Every file must be **100% complete and production-ready** for commercial
  deployment and customer delivery.

---

## 4. TECHNICAL ARCHITECTURE CONSTRAINTS (HARD)

### 4.1 — 100% MCSA Technology
- Electrical current waveforms (`I_a`, `I_b`, `I_c`) EXCLUSIVELY.
  No vibration, acoustic, or temperature sensors.
- **Hardware Interface:** ESP32 (Internal ADC + I2S DMA at 10.24 kHz
  sampling) + LM358 Signal Conditioning (1.65V DC Offset, Anti-Aliasing
  Filter, TVS Protection).
- **Architectural Split:** ESP32 acts strictly as a lightweight **Raw
  Array Sender**. FastAPI Backend receives raw arrays and performs heavy:
  - PyWavelets Denoising
  - 16,384-point Zoom FFT
  - Hilbert Envelope Demodulation
  - Feature Extraction
  - Stage 2 Early-Stage Fault Detection

### 4.2 — Mandatory External Supabase Database Policy (STRICT)
- Do NOT use Replit's internal database (Replit DB / local SQLite) under
  any circumstances. TinyDB, shelve, pickle persistence, and JSON-file
  persistence are equally VIOLATIONS.
- The project MUST exclusively use an external **Supabase PostgreSQL**
  database (`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` / `DATABASE_URL`
  via environment variables / Secrets) for ALL data persistence, motor
  health baselines, logs, user management, and fault history.

### 4.3 — Hybrid Dataset Lockdown
Train strictly on a combined dataset pipeline:
- **ZZU-MCC5 Dataset** — motor mechanical and stator/rotor faults.
- **IEEE DataPort Power Quality Disturbance (PQD) Dataset** — capacitor
  bank failures, switching transients, harmonics, grid disturbances.

### 4.4 — Operational Condition Agility
- Dynamic fundamental frequency tracking and filtering for:
  Direct-On-Line (DOL), Soft Starters, and Variable Frequency Drives
  (VFD) — without false alarms.

---

## 5. MANDATORY 11+ CLASS MULTI-CLASS FAULT TAXONOMY

ML models (**Isolation Forest** for Day-1 Baseline Anomaly Gating, and
**XGBoost / Random Forest** for multi-class classification) MUST
explicitly train on and support ALL of the following categories across
both datasets — omitting NONE:

**Mechanical Motor Faults**
1. Bearing: Outer Race Defect (BPFO), Inner Race Defect (BPFI),
   Cage/Train Defect (FTF), Ball/Rolling Element Defect (BSF)
2. Shaft Misalignment: Belt / Pulley Slip
3. Mechanical Unbalance & Eccentricity, Mechanical Overload & Jamming,
   Mechanical Load & Coupling Issues

**Electrical Motor Faults**
4. Broken Rotor Bars, End Ring Cracks, Dynamic Eccentricity
5. Stator Winding Inter-turn Short, Static Eccentricity, Ground
   Insulation Degradation
6. Phase Current Unbalance: Grid-Side Harmonic Distortion, Voltage Sags
   and Surges, Phase Current / Voltage Imbalance

**Capacitor & Grid Disturbances**
7. Capacitor Bank Failure / Blown Fuse
8. Capacitor Switching Transients
9. Harmonics Resonance & High THD
10. VFD DC-Bus Capacitor Aging
11. Voltage Sag/Swell & Inrush Current

**Plus:** Distance-to-Healthy Core / Slope Extrapolation for Remaining
Useful Life (RUL) Risk Windows.

---

## 6. DUAL-VIEW UX API (STRICT EXPLICIT DATA SCHEMAS)

FastAPI endpoints MUST generate two strictly isolated, production-ready
JSON payloads. NO generic fields or placeholding estimates.

### 6.a — Executive View Payload (Plant Owners, GMs, Financial Directors)
| Field | Definition |
|---|---|
| `plant_health_index` | Normalized 0–100% plant-wide health index aggregated across all active motor streams |
| `financial_risk_exposure_etb` | Real-time estimate of potential downtime financial risk exposure in Ethiopian Birr (ETB) |
| `rul_days_estimate` | RUL window in days with lower/upper statistical confidence bounds — computed EXCLUSIVELY by the deterministic Section 14 protocol (Distance-to-Healthy Core / Slope Extrapolation); never by an unverifiable model |
| `critical_asset_risk_summary` | Count of monitored motors by health status (Healthy, Stage 2 Moderate Risk, Stage 3/4 Critical Failure Risk) |
| `energy_efficiency_loss_penalty_etb` | Estimated monthly financial loss due to poor power factor, high THD, and phase current imbalance |
| `executive_decision_directive` | Concise strategic business action (e.g., "Plan Stage 2 bearing replacement during scheduled maintenance in 45 days") |

### 6.b — Technician View Payload (Electromechanical Technicians & Maintenance Engineers)
| Field | Definition |
|---|---|
| `mcc_panel_location` | Exact physical hardware location (MCC Panel ID, Cabinet Number, Line/Zone Identifier) |
| `motor_specifications` | Motor Asset ID, Nameplate kW rating, Rated RPM, Drive Type (DOL, Soft-Starter, VFD) |
| `exact_fault_taxonomy_code` | Precise classification from the 11+ taxonomy (e.g., "Stage 2 Broken Rotor Bar (2sf sideband peak detected)", "BPFO Bearing Outer Race Fault") |
| `spectral_evidence_data` | Quantitative DSP metrics: FFT peak frequencies (Hz), 50Hz sideband ratios, THD %, Z-score baseline deviation magnitude |
| `amharic_repair_protocol` | Comprehensive step-by-step repair & troubleshooting procedure in clear Amharic, tailored for Mobile App and Telegram Bot alerts |
| `required_tools_and_spares` | Specific spare parts (exact bearing part number, capacitor specification) and specialized tools required |
| `fault_urgency_stage` | Explicit severity (Stage 1–4) with maximum allowable operational running hours before mandatory trip/shutdown |

---

## 7. HARDENED LLM SECURITY

- OpenAI/DeepSeek API Proxy in FastAPI reading keys ONLY from environment
  variables (Secrets).
- LLM must ONLY process structured JSON payloads from our ML models.
- Mandatory: Prompt Injection defense, anonymization, and rate limiting.

---

## 8. MLOPS RETRAINING PIPELINE (MANDATORY)

- Automated or guided MLOps retraining pipeline allowing retraining /
  fine-tuning of V2 ML models based on:
  - Newly collected fault logs (Supabase)
  - Validated telemetry (Supabase)
  - Historical baseline data (Supabase)
- Must define: trigger mechanism (manual endpoint / scheduled job /
  drift-detection), model versioning strategy, and rollback capability.

---

## 9. POWER GRID & HARDWARE FAULT TOLERANCE

- Offline store-and-forward telemetry data buffering.
- State persistence during grid power outages.
- Startup inrush current suppression filtering to prevent false alarms
  upon power restoration.

---

## 10. EDGE-CLOUD DATA OPTIMIZATION & RETENTION POLICY (Supabase Tier Efficiency)

- **Raw Array Volatility:** Raw current sample arrays (2048/4096 points
  from ESP32) MUST reside transiently in FastAPI RAM during DSP
  processing. NEVER persist raw waveforms into Supabase storage.
- **Adaptive / Selective Logging:** Persist feature-extracted telemetry
  (RMS, THD, Kurtosis, Z-Score) at 15–30 minute intervals during normal
  healthy operation. Dynamically escalate to 1-minute logging ONLY when
  an anomaly (early Stage 2+) is detected.
- **Retention & Baseline Preservation:** Automated 90-day cleanup for
  routine healthy telemetry records; PERMANENTLY retain Motor Baseline
  Parameters, Z-score thresholds, and historical Fault/Maintenance Logs.

---

## 11. COLD START PROTOCOL (MANDATORY — CRITICAL FOR ETHIOPIAN CONTEXT)

The system MUST handle motors that ALREADY have faults when Sigmo V2 is
commissioned. "Day-1 Baseline" MUST NEVER assume the motor is healthy on
installation day.

1. **POPULATION BASELINE:** Use ZZU-MCC5 and IEEE PQD datasets to
   establish healthy reference; compute healthy feature distributions
   across thousands of motors; store as immutable "Population Baseline"
   in Supabase.
2. **COMMISSIONING DIAGNOSTIC MODE (first 24 hours):** Run No-Load /
   Full-Load / Coast-Down tests. Compare against Population Baseline AND
   IEEE/ISO absolute thresholds. Detect existing Stage 1–4 faults at
   commissioning time.
3. **ABSOLUTE THRESHOLDS (IEEE 1159, ISO 10816):** THD < 5%, Current
   Unbalance < 2%, Crest Factor 1.40–1.45, etc. Any motor outside
   threshold = flagged REGARDLESS of self-baseline.
4. **PERSONALIZED BASELINE (only after healthy confirmation):** If motor
   confirmed healthy → establish self-baseline over 7 days. If motor has
   existing fault → DO NOT use as baseline; track RUL from Population
   Baseline instead.
5. **PHYSICAL VERIFICATION (first 7 days for any pre-existing fault):**
   Technician checklist: noise, vibration, temperature, grease, play.
   Software prediction MUST be confirmed by physical inspection. Only
   confirmed predictions become permanent records.
6. **ANTI-FALSE-POSITIVE (4-signal confirmation required for Stage 2+
   alert):**
   - Population baseline deviation > 5σ
   - Absolute threshold (IEEE/ISO) breached
   - Trained model confidence > 70%
   - Downward trend over time
   ALL 4 required.
7. **OLD MOTOR HANDLING:** Motors with unknown history default to
   Population Baseline. NEVER assume old motor was healthy. RUL tracking
   begins from first measurement, not from a naive baseline
   (deterministic method: Section 14).

---

## 12. EXECUTION PROTOCOL (OPERATING SEQUENCE)

1. **First Response:** Acknowledge role and generate the Replit AI audit
   prompt for the current V2 project structure and existing codebase,
   incorporating V2 exclusivity and model retraining rules. ✅ DONE
2. **Second Response (after user provides Replit AI's audit output):**
   Analyze structure, align strictly with V2 Sigmo specs, and output
   ONLY the tailored Replit AI Prompt for Step 1.
3. **Subsequent Responses (Code Audit & Verification):** After the user
   pastes Replit AI's generated code for Step X, thoroughly audit and
   verify it against this rulebook — V2 exclusivity, zero fake code,
   Supabase-only, MCSA logic. ONLY after confirmed 100% compliance,
   generate the prompt for Step X+1.

---

## 13. FAILURE CONSEQUENCES

Violating any of these instructions — skipping steps, outputting fake
code, or mixing V1 files into V2 — constitutes a failure of the
engineering mandate. Adhere strictly to this protocol at all times.

---

## 14. DETERMINISTIC RUL VIA SLOPE EXTRAPOLATION (DISTANCE-TO-HEALTHY CORE)

**Classification statement.** This protocol is a DETERMINISTIC,
mathematics-based time-series extrapolation over live telemetry —
standard industrial condition-monitoring practice. It is explicitly
NOT a statistical/ML run-to-failure prognosis model and never claims
failure-time knowledge that does not exist in the data (no
run-to-failure corpus is required or assumed). Every output is either
(a) the exact arithmetic result of stored Supabase telemetry or (b) an
explicit non-numeric status string. Zero fake code (Section 3) applies
in full: no mock numbers, no simulated degradation curves, ever.

### 14.1 — Health Index (HI) = Distance-to-Healthy Metric

1. The Health Index of each completed analysis window is the
   **Population Baseline deviation magnitude** `zscore_max`: the
   largest absolute z-score `|value − baseline_mean| / baseline_std`
   across the baselined three-phase features (phase RMS, THD %,
   current unbalance %, rotor 2sf sideband dB).
2. The authoritative baseline per motor follows Cold Start 11.4:
   `HEALTHY_CONFIRMED` motors use their personalized `motor_baselines`;
   all other states (`COMMISSIONING`, `PRE_EXISTING_FAULT`,
   `POPULATION_TRACKED`) use the immutable Population Baseline
   (11.1), so RUL tracking works from first measurement and NEVER
   assumes an old motor was healthy (11.7).
3. The Isolation-Forest novelty gate stays in its existing role —
   signal 1 of the 4-signal confirmation rule (11.6), a trip flag,
   never an alert source by itself. It is NOT used as the HI: the
   gate emits a binary novelty decision, while the HI must be a
   continuous magnitude to admit slope mathematics. The baseline
   z-score is continuous, already computed per window, and already
   persisted.
4. HI units are sigma (σ). HI = 0 means "at the healthy core"; HI
   grows monotonically with distance from the healthy reference.

### 14.2 — Time-Series Storage (Supabase, STRICT)

1. The HI is persisted per window in `telemetry_features` as the
   `(recorded_at, zscore_max)` pair — one continuous time series per
   `motor_id` in the external Supabase PostgreSQL database (4.2). No
   other persistence path is permitted.
2. `INRUSH_SUPPRESSED` windows are stored but MUST be excluded from
   every RUL computation — startup transients are not degradation
   (Section 9).
3. Retention interplay (Section 10): the 90-day healthy-telemetry
   cleanup can never harm RUL — the extrapolation window (14.4) is at
   most 7 days.

### 14.3 — Cold-Start Logic (Insufficient Data)

1. Accumulated telemetry time = wall-clock span from the motor's
   earliest non-inrush `telemetry_features` row to now, measured on
   Supabase timestamps.
2. If accumulated time is **less than 24 hours**: output exactly
   `rul_days_estimate: "Calculating... (Insufficient Data)"`. No
   numeric output is permitted in this state.
3. Transitional state — accumulated time ≥ 24 hours but the trend
   minimums of 14.4 are not yet met: output exactly
   `rul_days_estimate: "Calculating... (Building Trend)"`.

### 14.4 — Extrapolation Math (Rate of Degradation)

1. Historical window: the last `RUL_LOOKBACK_DAYS = 7` days of
   non-inrush telemetry rows (configurable per deployment within
   3–7 days).
2. Trend-validity minimums (deterministic data-quality gate, both
   required):
   - time span of the window ≥ `RUL_MIN_TREND_SPAN_DAYS = 3` days, AND
   - count ≥ `RUL_MIN_TREND_POINTS = 12` non-inrush windows.
3. Slope: ordinary least-squares regression of HI on time —
   `d(HI)/dt` in σ/day — computed only over the historical window.
   OLS is fully deterministic: identical rows in, identical slope out.
4. Current HI = the HI of the motor's most recent non-inrush window.

### 14.5 — RUL Calculation (Deterministic Formula)

`CRITICAL_FAILURE_THRESHOLD = 5.0 σ` — deliberately identical to the
11.6 signal-1 population-deviation threshold and the
`motor_baselines.zscore_alert_threshold` default, so "RUL exhausted"
and "4-signal alert" reference one and the same boundary.

1. **Slope ≤ 0 (stable or improving):** output exactly
   `rul_days_estimate: "Stable (999+ Days)"`.
2. **Slope > 0 (degrading):**
   `RUL_days = (CRITICAL_FAILURE_THRESHOLD − current_HI) / slope`,
   output in days rounded to whole days.
3. **Current HI ≥ CRITICAL_FAILURE_THRESHOLD:** RUL is `0` days — the
   asset is at/beyond the critical boundary. Alerting remains
   governed exclusively by the 4-signal rule (11.6).
4. **Presentation cap:** any computed RUL > 999 days is reported as
   `"Stable (999+ Days)"` — three-digit precision is honest; four is
   false precision.
5. **Confidence bounds (Executive View 6.a):** lower/upper bounds are
   propagated from the OLS slope standard error `SE_slope`:
   lower = `(THRESHOLD − current_HI) / (slope + SE_slope)`,
   upper = `(THRESHOLD − current_HI) / (slope − SE_slope)` when
   `slope > SE_slope`, else upper = 999. Both clamped to [0, 999].
   Bounds are reported only alongside a numeric estimate.

### 14.6 — Output Contract

| Condition | `rul_days_estimate` |
|---|---|
| accumulated telemetry < 24 h | `"Calculating... (Insufficient Data)"` |
| ≥ 24 h but trend minimums unmet | `"Calculating... (Building Trend)"` |
| slope ≤ 0 σ/day | `"Stable (999+ Days)"` |
| slope > 0, current HI < threshold | integer days |
| computed RUL > 999 days | `"Stable (999+ Days)"` |
| current HI ≥ 5.0 σ | `0` |

Every payload also carries the audit fields `rul_status`
(`CALCULATING_INSUFFICIENT_DATA | CALCULATING_BUILDING_TREND |
STABLE | DEGRADING | CRITICAL_THRESHOLD_REACHED`),
`slope_sigma_per_day`, `current_health_index_sigma`, and
`trend_window_days`, so a technician can reproduce the arithmetic
end-to-end from Supabase rows.

### 14.7 — Honest Scope Disclosure

1. This is condition-based extrapolation of a measured distance
   metric — NOT a physics-of-failure prognosis. If a true
   run-to-failure corpus ever exists (accumulated fleet history), a
   learned RUL model MAY be added under the Section 8 MLOps pipeline
   as a separate versioned model family, subject to the same
   enable-if-better gating as every other model.
2. RUL estimates NEVER trigger alerts by themselves; alerting stays
   exclusively with the 4-signal confirmation (11.6).
3. The extrapolation re-anchors on every evaluation: a repaired motor
   whose HI collapses back toward the healthy core automatically
   returns to "Stable" — no manual reset.

---

---

## 15. ESP32 FIRMWARE & WOKWI SIMULATION PROTOCOL (MTR-PILOT-01)

- **HARDWARE MAPPING & SCOPE:**
  - Target Motor: `MTR-PILOT-01` (75 kW, ~140 A FLA).
  - Pure MCSA exclusively (`Ia`, `Ib`, `Ic` current waveforms). Zero extra sensors (§4.1).
  - Pin Allocation (ADC1 Only): `GPIO 32` (Phase A / Ia), `GPIO 33` (Phase B / Ib), `GPIO 34` (Phase C / Ic).
  - Wokwi Simulation Setup: Map physical SCT-019 CT sensors to 3× Potentiometers for input waveform generation.

- **SAMPLING ENGINE & FREERTOS DUAL-CORE:**
  - Sampling Rate: Exactly 10,240 Hz via I2S DMA.
  - Frame Size: Exactly 2048 raw samples per frame per phase.
  - Core 1 Task: Continuous high-speed ADC1 sampling into ping-pong DMA buffers.
  - Core 0 Task: Wi-Fi stack, network reconnections, and non-blocking HTTP POST payload transmission.

- **CLOUD INTEGRATION & SIMULATION NETWORK:**
  - Wi-Fi SSID: `Wokwi-GUEST` (Virtual open network for Wokwi testing).
  - Direct Live API Target: `POST https://sigmo-backend-w4cx.onrender.com/api/v2/telemetry`.
  - JSON Payload: Full raw array telemetry matching Sigmo V2 API specifications.

- **ZERO FAKE CODE DIRECTIVE (§3 COMPLIANCE):**
  - Provide 100% complete, compilable, and production-ready C++ firmware.
  - Zero mock data generators, zero `TODO` placeholders, zero empty functions. Code must compile natively both in Wokwi and on physical ESP32 DevKit hardware.

====================================================================
END OF RULEBOOK — SIGMO V2 | Version 1.1 | Original: 2026-09-21 | Amendment §14 (Deterministic RUL via Slope Extrapolation): 2026-09-24 | Amendment §15 (ESP32 Firmware & Wokwi Simulation Protocol): 2026-10-06
====================================================================
