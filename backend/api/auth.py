"""Sigmo V2 — API-key authentication (SIGMO_RULES.md Section 7 hardening).

Three scopes, one header (``X-API-Key``):

  device     ESP32 nodes: POST /api/v2/telemetry only.
  dashboard  Read surface for the Replit frontend / mobile clients:
             GET /api/v2/motors/* and GET /api/v2/views/*.
  admin      Commissioning & maintenance: everything above plus
             /api/v2/admin/* and /api/v2/maintenance/*.

Keys live ONLY in the ``SIGMO_API_KEYS`` environment variable (Render
Secrets), as a comma-separated list of ``key:scope`` pairs —
credentials never touch the database or any file.

FAIL-CLOSED policy: when no key is configured, every protected endpoint
returns 503 with an explicit message. The service boots and reports
health, but the protected surface stays closed until commissioning
enters real keys — an unauthenticated open API is never an acceptable
default for a commercial deployment.

Deliberately open (no key required):
  /api/v2/health        — platform health checks (Render) and uptime probes
  /api/v2/models/active — ACTIVE version metadata only (no motor data)
  /docs, /openapi.json  — interactive API documentation

Timing-safe comparison (``hmac.compare_digest``) is used for every key
check. All responses are JSON so clients get a machine-readable error.
"""
from __future__ import annotations

import hmac
import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("sigmo.auth")

HEADER = "X-API-Key"

VALID_SCOPES = frozenset({"device", "dashboard", "admin"})

# Route protection matrix: (path prefix, methods, scopes that may call).
# The admin scope is intentionally absent from device/dashboard rows —
# scope grants are explicit, not hierarchical, except that the rows
# below are matched in order and /api/v2/admin + /api/v2/maintenance
# accept ONLY admin.
PROTECTED: tuple[tuple[str, tuple[str, ...], frozenset[str]], ...] = (
    ("/api/v2/telemetry", ("POST",), frozenset({"device", "admin"})),
    ("/api/v2/motors", ("GET",), frozenset({"dashboard", "admin"})),
    ("/api/v2/views", ("GET",), frozenset({"dashboard", "admin"})),
    ("/api/v2/admin", ("GET", "PUT", "POST", "DELETE"),
     frozenset({"admin"})),
    ("/api/v2/maintenance", ("POST",), frozenset({"admin"})),
)

# module-level key store; written by load_keys() at startup (and by the
# test suites, which inject their own keys before driving the app).
_KEYS: dict[str, str] = {}


def load_keys(raw: str) -> int:
    """Parse ``"key1:scope,key2:scope"`` into the live key store.

    Returns the number of keys loaded. Malformed entries are refused
    loudly (ValueError) — a typo in a Secrets value must never silently
    disable authentication.
    """
    keys: dict[str, str] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, scope = part.rpartition(":")
        key, scope = key.strip(), scope.strip().lower()
        if not sep or not key or scope not in VALID_SCOPES:
            raise ValueError(
                f"Invalid SIGMO_API_KEYS entry {part!r} — expected "
                f"'key:scope' with scope in {sorted(VALID_SCOPES)}"
            )
        keys[key] = scope
    _KEYS.clear()
    _KEYS.update(keys)
    if keys:
        logger.info(
            "API authentication active: %d key(s), scopes=%s",
            len(keys), sorted(set(keys.values())),
        )
    else:
        logger.warning(
            "SIGMO_API_KEYS is empty — protected endpoints will return "
            "503 until commissioning configures real keys (fail-closed)."
        )
    return len(keys)


def reset_keys() -> None:
    """Clear the key store (test isolation)."""
    _KEYS.clear()


def keys_configured() -> bool:
    return bool(_KEYS)


def _scope_for(presented: str) -> str | None:
    """Scope of the presented key, or None if it is not a valid key.

    Compares against EVERY configured key with constant-time equality —
    the number of comparisons does not leak which character positions
    matched.
    """
    for key, scope in _KEYS.items():
        if hmac.compare_digest(presented, key):
            return scope
    return None


def _rule_for(path: str, method: str) -> tuple[str, frozenset[str]] | None:
    """First matching protection rule for a request, or None (open)."""
    for prefix, methods, scopes in PROTECTED:
        if path.startswith(prefix) and method in methods:
            return prefix, scopes
    return None


async def enforce_api_key(request: Request, call_next):
    """HTTP middleware: enforce the protection matrix above.

    Order matters at registration time: this middleware is added BEFORE
    the CORS middleware so CORS ends up outermost and even 401/403/503
    responses carry the CORS headers browsers need to read them.
    """
    rule = _rule_for(request.url.path, request.method)
    if rule is None:
        return await call_next(request)

    _, allowed_scopes = rule
    if not keys_configured():
        return JSONResponse(
            status_code=503,
            content={
                "detail": "API authentication is not configured — set "
                "SIGMO_API_KEYS (key:scope pairs) in the service "
                "environment. Protected endpoints stay closed until "
                "real keys are commissioned (fail-closed policy)."
            },
        )

    presented = request.headers.get(HEADER, "")
    if not presented:
        return JSONResponse(
            status_code=401,
            content={"detail": f"Missing {HEADER} header."},
        )
    scope = _scope_for(presented)
    if scope is None:
        return JSONResponse(
            status_code=401, content={"detail": "Invalid API key."}
        )
    if scope not in allowed_scopes:
        return JSONResponse(
            status_code=403,
            content={
                "detail": f"API key scope '{scope}' is not permitted for "
                f"{request.method} {request.url.path}."
            },
        )
    return await call_next(request)
