"""Sigmo V2 — FastAPI application entrypoint.

Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
(from the sigmo_v2/ directory, with DATABASE_URL set in Secrets)

Endpoints:
  POST /api/v2/telemetry   — ESP32 raw-array ingestion (the ONLY place
                             raw waveforms enter the system; RAM-only);
                             each completed steady-state window now also
                             carries the ACTIVE v6c dominant-fault verdict
  GET  /api/v2/health      — liveness/readiness + offline buffer depth
  GET  /api/v2/motors/{motor_id}/assessment
                           — comprehensive per-motor payload in ONE
                             structured JSON response: latest physical
                             snapshot, v6c dominant-fault verdict with
                             probabilities + temporal aggregation, and
                             the deterministic Section 14 RUL estimate
  GET  /api/v2/motors/{motor_id}/history
                           — steady-state window series for the Analytics
                             trends charts (Section 14.5 health index,
                             worst-phase power quality, mechanical
                             signature and the persisted verdict per
                             window, oldest-first)
  GET  /api/v2/faults    — fault episode history (Section 10 permanent
                           retention; Section 11.6 four-signal evidence
                           per episode, newest first)
  GET  /api/v2/models/active — ACTIVE classifier version metadata
  GET  /api/v2/views/executive
                           — Section 6.a Executive payload (one JSON:
                             plant health index, ETB risk exposure,
                             binding RUL window, stage counts, energy
                             penalty, decision directive)
  GET  /api/v2/views/technician/{motor_id}
                           — Section 6.b Technician payload (one JSON:
                             MCC location, nameplate specs, exact fault
                             taxonomy code, spectral evidence, Amharic
                             repair protocol, tools & spares, urgency
                             stage)
  PUT  /api/v2/admin/motor-assets/{motor_id}
                           — commissioning asset-registry entry
  GET  /api/v2/admin/motor-assets — registered assets
  GET  /api/v2/admin/plant-config  — configured business rates
  PUT  /api/v2/admin/plant-config/{key} — set one business rate
  POST /api/v2/maintenance/retention-cleanup
                           — triggers the 90-day healthy-telemetry cleanup

CORS: configured for the external Replit Agent frontend (Phase 4).
Origins are env-driven via CORS_ALLOW_ORIGINS (comma-separated,
default ``*`` for MVP); credentials are never required by the API.

Startup behaviour honours Section 9 outage tolerance: if Supabase is
unreachable at boot, the service still starts, ingests, analyzes, and
buffers feature rows in RAM, then flushes when the database returns.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.auth import enforce_api_key, load_keys
from .api.ingestion import IngestionService
from .api.schemas import (
    ExecutiveViewPayload,
    FrameAck,
    HealthResponse,
    MotorAssetRegistration,
    MotorAssessment,
    PlantConfigEntry,
    TechnicianViewPayload,
    TelemetryFrame,
    MotorsOverviewPayload,
    TrendsPayload,
    FaultLogsPayload,
    AIExplainRequest,
    AIExplainResponse,
)
from .ai import client as ai_client
from .ai.client import AIExplanationError, AIProviderNotConfigured
from .analysis.repair_protocols_am import AMHARIC_PROTOCOLS
from .analysis.staging import motor_health_percent
from .ai.client import AIExplanationError, AIProviderNotConfigured
from .config import API
from .db.repository import SigmoRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sigmo.main")

SCHEMA_PATH = Path(__file__).parent / "db" / "schema.sql"

repository = SigmoRepository()
service = IngestionService(repository)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Section 7 hardening: parse API keys from the environment BEFORE the
    # service accepts traffic (fail-closed when unset — see auth module).
    load_keys(API.api_keys_raw)
    try:
        await repository.connect()
        await repository.apply_schema(SCHEMA_PATH.read_text(encoding="utf-8"))
        service.database_connected = True
        logger.info("Connected to Supabase PostgreSQL; schema verified.")
    except Exception as exc:
        # Section 9: never refuse to start because the grid/network is down.
        service.database_connected = False
        logger.error(
            "Supabase unavailable at startup (%s). Running in degraded "
            "store-and-forward mode; features will buffer in RAM.", exc,
        )
    yield
    await repository.close()


app = FastAPI(
    title="Sigmo V2 — MCSA Predictive Maintenance Backend",
    version="2.0.0",
    lifespan=lifespan,
)

# API-key authentication (Section 7 hardening). Registered BEFORE CORS
# (Starlette: last added = outermost) so CORS wraps auth and even
# 401/403/503 responses carry the CORS headers browsers need.
app.add_middleware(BaseHTTPMiddleware, dispatch=enforce_api_key)

# CORS (Phase 4): the Replit Agent frontend is served from a different
# origin and must be able to fetch the assessment payload and POST
# telemetry. Wildcard by MVP default; locked down per deployment via
# CORS_ALLOW_ORIGINS. allow_credentials is False because the API is
# key/token-free and the CORS spec forbids ``*`` + credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(API.cors_allow_origins),
    allow_methods=list(API.cors_allow_methods),
    allow_headers=list(API.cors_allow_headers),
    allow_credentials=False,
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.post("/api/v2/telemetry", response_model=FrameAck)
async def ingest_telemetry(frame: TelemetryFrame) -> FrameAck:
    """Accept one raw three-phase frame from an ESP32 node.

    The frame is validated by the TelemetryFrame contract (length,
    sampling rate, finiteness), accumulated in RAM, and — when a
    32,768-sample window completes — pushed through the full DSP +
    gating + adaptive persistence chain. The response tells the firmware
    whether its frame was accepted and how full the current window is,
    and carries the window analysis when one was produced.
    """
    try:
        fill, analysis = await service.ingest(frame)
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Ingestion failure for motor %s", frame.motor_id)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}") from exc
    return FrameAck(
        accepted=True,
        motor_id=frame.motor_id,
        sequence=frame.sequence,
        window_fill_percent=fill,
        window_analyzed=analysis is not None,
        analysis=analysis,
    )


@app.get("/api/v2/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        service="sigmo-v2-backend",
        database_connected=service.database_connected,
        offline_buffer_depth=service.offline_buffer_depth,
        motors_streaming=service.motors_streaming,
    )


@app.get("/api/v2/motors", response_model=MotorsOverviewPayload)
async def motors_overview() -> MotorsOverviewPayload:
    """Live per-motor summary list for the dashboard status grid.

    One motor entry per asset with telemetry (live verdict union
    persisted Supabase row, freshest wins): gating status, health
    index, Stage 1-4 and the persisted v6c verdict. Read-only, no
    per-motor RUL reads (the Stage engine alone drives the list).
    """
    try:
        return await service.motors_overview()
    except Exception as exc:
        logger.exception("Motors overview failure")
        raise HTTPException(
            status_code=500, detail=f"Motors overview error: {exc}"
        ) from exc


@app.get("/api/v2/motors/{motor_id}/assessment", response_model=MotorAssessment)
async def motor_assessment(motor_id: str) -> MotorAssessment:
    """Complete per-motor status in one structured JSON response.

    Phase 4 comprehensive contract for the Replit dashboard: physical
    snapshot (latest steady-state window), the ACTIVE v6c dominant-fault
    verdict with full 13-class probabilities and temporal aggregation,
    and the deterministic Section 14 RUL estimate. Freshness precedence:
    the live in-RAM verdict of the latest analyzed window when this
    service has seen the motor, otherwise the latest persisted Supabase
    row. Raises 404 when the motor has no telemetry at all and 422 on
    invalid motor identifiers (ValueError handler).
    """
    try:
        return await service.assessment(motor_id)
    except ValueError as exc:
        if "No telemetry recorded" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise
    except Exception as exc:
        logger.exception("Assessment failure for motor %s", motor_id)
        raise HTTPException(status_code=500, detail=f"Assessment error: {exc}") from exc


@app.get("/api/v2/motors/{motor_id}/history", response_model=TrendsPayload)
async def motor_history(
    motor_id: str,
    hours: int = Query(168, ge=1, le=720),
    limit: int = Query(500, ge=1, le=2000),
) -> TrendsPayload:
    """Per-motor steady-state window history (Analytics trends source).

    The newest non-inrush telemetry windows of the last ``hours`` hours
    (default 168 = 7 days), capped at ``limit`` rows, returned
    oldest-first. Each point carries the Section 14.5 health index,
    worst-phase power-quality features (THD, unbalance, crest, rotor
    sideband) and the persisted v6c verdict. Raises 404 when the motor
    has no telemetry at all; an empty ``points`` list means the motor
    has telemetry, just none inside the requested window (honest
    absence, never fabricated rows).
    """
    try:
        return await service.telemetry_history(
            motor_id, hours=hours, limit=limit
        )
    except ValueError as exc:
        if "No telemetry recorded" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise
    except Exception as exc:
        logger.exception("History failure for motor %s", motor_id)
        raise HTTPException(
            status_code=500, detail=f"History error: {exc}"
        ) from exc


@app.get("/api/v2/faults", response_model=FaultLogsPayload)
async def faults(
    motor_id: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
) -> FaultLogsPayload:
    """Fault episode history (Section 10 permanent retention).

    Newest fault log rows first — all motors, or one motor when
    ``motor_id`` is given. Each row carries the exact taxonomy code,
    urgency stage, the four Section 11.6 anti-false-positive signals
    (with the database-computed four_signal_confirmed) and the
    quantitative spectral evidence of the detection window. Fault rows
    are written by the ingestion path when a NEW fault episode starts
    (episode dedup on the taxonomy code).
    """
    try:
        return await service.fault_logs(motor_id, limit)
    except Exception as exc:
        logger.exception("Fault logs read failure")
        raise HTTPException(
            status_code=500, detail=f"Fault logs error: {exc}"
        ) from exc


@app.post("/api/v2/ai/explain", response_model=AIExplainResponse)
async def ai_explain(request: AIExplainRequest) -> AIExplainResponse:
    """Real-LLM technician analysis of one motor (Amharic + English).

    Feeds the ACTUAL V2 evidence to the configured LLM provider
    (OPENAI_API_KEY / DEEPSEEK_API_KEY / GEMINI_API_KEY — never a graph
    image, never invented numbers): latest steady-state window, recent
    window history, the ACTIVE v6c verdict with probabilities and
    temporal aggregation, the deterministic stage and Section 14 RUL,
    the registered nameplate specs and the system's own Amharic repair
    protocol as grounding. The model must answer as an expert
    industrial motor technician: root cause tied to the measured
    values, then step-by-step actions, in Amharic AND English.
    """
    import time as _time

    motor_id = request.motor_id
    try:
        assessment = await service.assessment(motor_id)
        history = await service.telemetry_history(
            motor_id, hours=168, limit=8
        )
    except ValueError as exc:
        if "No telemetry recorded" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise
    except Exception as exc:
        logger.exception("AI explain context failure for motor %s", motor_id)
        raise HTTPException(
            status_code=500, detail=f"AI explain context error: {exc}"
        ) from exc

    asset: dict | None = None
    try:
        asset = await repository.get_motor_asset(motor_id)
    except Exception:
        asset = None  # nameplate specs are enrichment, never blocking

    fa = assessment.fault_assessment
    snap = assessment.snapshot
    predicted = fa.dominant_fault if fa is not None else None
    context: dict[str, object] = {
        "motor": {
            "motor_id": motor_id,
            "nameplate_kw": asset.get("nameplate_kw") if asset else None,
            "rated_rpm": asset.get("rated_rpm") if asset else None,
            "drive_type": asset.get("drive_type") if asset else None,
            "mcc_panel_id": asset.get("mcc_panel_id") if asset else None,
        },
        "current_window": {
            **(
                {
                    "recorded_at": snap.recorded_at.isoformat(),
                    "gate_status": snap.status,
                    "fundamental_hz": snap.fundamental_hz,
                    "rms_a": snap.rms_a,
                    "rms_b": snap.rms_b,
                    "rms_c": snap.rms_c,
                    "thd_percent_max": snap.thd_percent_max,
                    "current_unbalance_percent": (
                        snap.current_unbalance_percent
                    ),
                    "crest_factor_max": snap.crest_factor_max,
                    "rotor_sideband_db_max": snap.rotor_sideband_db_max,
                    "zscore_max_sigma": snap.zscore_max,
                    "health_percent": motor_health_percent(snap.zscore_max),
                }
                if snap is not None
                else {}
            ),
            "verdict_13class": predicted,
            "verdict_confidence": fa.confidence if fa else None,
            "verdict_probabilities_top5": (
                dict(
                    sorted(fa.probabilities.items(), key=lambda kv: -kv[1])[:5]
                )
                if fa
                else {}
            ),
            "temporal_aggregation": fa.temporal_aggregation if fa else None,
        },
        "rul": {
            "rul_days_estimate": assessment.rul.rul_days_estimate
            if assessment.rul
            else None,
            "rul_status": assessment.rul.rul_status if assessment.rul else None,
        },
        "recent_windows": [
            {
                "recorded_at": p.recorded_at.isoformat(),
                "gate_status": p.status,
                "verdict_13class": p.predicted_class,
                "health_percent": p.health_percent,
                "thd_percent_max": p.thd_percent_max,
                "current_unbalance_percent": p.unbalance_percent,
                "crest_factor_max": p.crest_factor_max,
                "rotor_sideband_db_max": p.rotor_sideband_db_max,
            }
            for p in history.points[-8:]
        ],
        "reference_limits": {
            "thd_percent_max_ieee519": 5.0,
            "current_unbalance_percent_max": 2.0,
            "crest_factor_healthy_band": [1.40, 1.45],
        },
    }
    if predicted is not None:
        context["system_reference_protocol_amharic"] = (
            AMHARIC_PROTOCOLS.get(predicted, "")
        )

    started = _time.monotonic()
    try:
        result = await ai_client.generate_explanation(context)
    except AIProviderNotConfigured as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AIExplanationError as exc:
        logger.warning("AI explanation failed for %s: %s", motor_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    elapsed_ms = int((_time.monotonic() - started) * 1000)

    return AIExplainResponse(
        motor_id=motor_id,
        provider=result["provider"],
        model=result["model"],
        generated_at=datetime.now(timezone.utc),
        analysis_am=result["analysis_am"],
        analysis_en=result["analysis_en"],
        elapsed_ms=elapsed_ms,
        data_context=context,
    )


@app.get("/api/v2/models/active")
async def active_model() -> dict[str, str | None]:
    """ACTIVE classifier metadata (registry version + load state)."""
    return {
        "model_version": service.model_version,
        "source": "models/registry.json" if service.model_version else None,
    }


@app.get("/api/v2/views/executive", response_model=ExecutiveViewPayload)
async def executive_view() -> ExecutiveViewPayload:
    """Section 6.a Executive payload — strictly isolated from the
    Technician view. One structured JSON with the exact rulebook fields;
    financial figures come exclusively from customer-configured rates
    (unconfigured rates surface as explicit statuses, never invented
    numbers)."""
    try:
        return await service.executive_view()
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("Executive view failure")
        raise HTTPException(
            status_code=500, detail=f"Executive view error: {exc}"
        ) from exc


@app.get(
    "/api/v2/views/technician/{motor_id}",
    response_model=TechnicianViewPayload,
)
async def technician_view(motor_id: str) -> TechnicianViewPayload:
    """Section 6.b Technician payload — strictly isolated from the
    Executive view. Amharic repair protocol, spectral evidence, and the
    Stage 1-4 urgency verdict for one motor. 404 for unknown motors."""
    try:
        return await service.technician_view(motor_id)
    except ValueError as exc:
        if "No telemetry recorded" in str(exc):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise
    except Exception as exc:
        logger.exception("Technician view failure for motor %s", motor_id)
        raise HTTPException(
            status_code=500, detail=f"Technician view error: {exc}"
        ) from exc


@app.put("/api/v2/admin/motor-assets/{motor_id}")
async def register_motor_asset(
    motor_id: str, asset: MotorAssetRegistration
) -> dict[str, str]:
    """Commissioning entry: register or update one motor's physical
    asset record (Section 6.b data source). Admin surface — run during
    commissioning with the real nameplate/MCC data."""
    try:
        await repository.upsert_motor_asset(
            {"motor_id": motor_id, **asset.model_dump()}
        )
    except ValueError:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Database unavailable: {exc}"
        ) from exc
    return {"motor_id": motor_id, "status": "registered"}


@app.get("/api/v2/admin/motor-assets")
async def list_motor_assets() -> list[dict]:
    """All registered motor assets (commissioning audit)."""
    try:
        return await repository.list_motor_assets()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Database unavailable: {exc}"
        ) from exc


@app.get("/api/v2/admin/plant-config")
async def get_plant_config() -> dict[str, dict]:
    """Configured business rates (Section 6.a financial basis)."""
    try:
        return await repository.get_plant_config()
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Database unavailable: {exc}"
        ) from exc


@app.put("/api/v2/admin/plant-config/{key}")
async def set_plant_config(
    key: str, entry: PlantConfigEntry
) -> dict[str, str]:
    """Set one business rate (e.g. energy_tariff_etb_per_kwh,
    default_downtime_cost_etb_per_hour, plant_power_factor,
    operating_hours_per_month). Customer-entered values only."""
    try:
        await repository.set_plant_config(
            key, entry.value, entry.unit, entry.description
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Database unavailable: {exc}"
        ) from exc
    return {"key": key, "status": "configured", "value": str(entry.value)}


@app.post("/api/v2/maintenance/retention-cleanup")
async def retention_cleanup() -> dict[str, int]:
    """Manually trigger the 90-day healthy-telemetry cleanup (Section 10).
    In production this is also wired to a scheduled Supabase cron job."""
    try:
        removed = await repository.run_retention_cleanup()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    return {"rows_removed": removed}
