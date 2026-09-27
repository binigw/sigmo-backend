# Sigmo V2 — MCSA Predictive Maintenance Backend

Production FastAPI service for the Sigmo V2 motor-condition-monitoring
system (MCSA — Motor Current Signature Analysis), carrying the ACTIVE
classifier bundle **v6.20260925.045900** (13-class fault taxonomy, accuracy
0.8995 / macro-F1 0.8475 held-out).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v2/telemetry` | ESP32 raw three-phase frame ingestion (32,768-sample windows) |
| GET | `/api/v2/health` | Liveness + DB connectivity + offline buffer depth |
| GET | `/api/v2/motors/{motor_id}/assessment` | Comprehensive per-motor payload (snapshot + verdict + Section 14 RUL) |
| GET | `/api/v2/views/executive` | Executive payload: plant health index, ETB risk exposure, RUL window, stage counts, energy penalty, directive |
| GET | `/api/v2/views/technician/{motor_id}` | Technician payload: MCC location, nameplate specs, exact fault taxonomy code, spectral evidence, Amharic repair protocol, tools & spares, Stage 1–4 urgency |
| GET | `/api/v2/models/active` | ACTIVE classifier version |
| PUT | `/api/v2/admin/motor-assets/{motor_id}` | Commissioning asset-registry entry |
| GET | `/api/v2/admin/motor-assets` | Registered assets |
| GET/PUT | `/api/v2/admin/plant-config/{key}` | Business rates (ETB tariff, downtime cost, PF, hours) |
| POST | `/api/v2/maintenance/retention-cleanup` | 90-day healthy-telemetry cleanup |

## Deployment

The repository is directly buildable:

- **Docker** (any builder): `docker build .` — the Dockerfile installs
  runtime dependencies, copies the code, and fetches the two large
  model artifacts (UBJSON booster + ISO gate) from this repository's
  pinned revision.
- **Hugging Face Space**: use as a Docker-SDK Space (requires HF PRO on
  current policy; `app_port: 7860`).

CORS is open (`*`) by default so external frontends (e.g. Replit) can
fetch immediately; lock it down by setting `CORS_ALLOW_ORIGINS`.

## Authentication (Section 7)

All protected endpoints require the `X-API-Key` header. Three scopes:
`device` (ESP32 → POST telemetry), `dashboard` (frontend → GET reads),
`admin` (commissioning → everything). Open paths: `/api/v2/health`,
`/api/v2/models/active`, `/docs`. **Fail-closed:** with no keys
configured, protected endpoints return 503 until real keys are
commissioned.

## Configuration (environment / Secrets)

- `DATABASE_URL` — Supabase PostgreSQL connection string (session
  pooler, port 5432, `?sslmode=require`). **Without it the service
  runs in degraded store-and-forward mode** (ingestion, gating, and ML
  verdicts still work; persisted history, RUL, and admin writes
  require the database).
- `SIGMO_API_KEYS` — comma-separated `key:scope` pairs, e.g.
  `abc123:device,def456:dashboard,ghi789:admin`.
- `CORS_ALLOW_ORIGINS` — optional comma-separated origin allowlist.

See `DEPLOYMENT.md` for the full commissioning guide.

The full rulebook governing this service (`SIGMO_RULES.md`) is included
in this repository.
