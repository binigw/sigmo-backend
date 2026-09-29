"""Sigmo V2 — in-app technician invitations (admin-gated, server-side).

POST /api/v2/invitations   {"email": "..."}

Lets a signed-in ADMIN send a Supabase invite email straight from the
dashboard (Personnel page) so a technician's LOGIN account no longer
requires a trip to the Supabase console. The technician receives an
email link and sets their own password — passwords are never chosen
by the admin and never transit the app.

Security chain (three independent layers, ALL must pass):

  1. ``X-API-Key`` with dashboard/admin scope — the standard auth.py
     middleware gate every other endpoint uses.
  2. ``Authorization: Bearer <access token>`` of the signed-in Supabase
     user, verified SERVER-SIDE by token introspection
     (GET ``{SUPABASE_URL}/auth/v1/user``). Browser claims are never
     trusted; a missing/expired/revoked token is rejected.
  3. ``public.user_roles.role == 'admin'`` for that user (DB-backed
     RBAC). A missing role row is FAIL-CLOSED: the default role is
     technician, and technicians cannot invite.

The Supabase service-role key lives ONLY in this service's environment
(``SUPABASE_SERVICE_ROLE_KEY``) and is used exclusively for
server-to-server calls (the introspection apikey header and POST
/auth/v1/invite). It is never sent to any browser. With no key
configured the endpoint fails closed with 503 ``invite_not_configured``
(zero-fake-code directive: an honest error, never a silent no-op).

Role assignment (fail-closed by design): after a successful invite the
new user gets an explicit ``technician`` row in ``public.user_roles``
via INSERT .. ON CONFLICT DO NOTHING — an existing role (e.g. a
pre-provisioned admin) is never overwritten, and even if that insert
failed the frontend already defaults a missing role row to technician.
"""
from __future__ import annotations

import logging
import re

import httpx

from ..config import DB

logger = logging.getLogger("sigmo.invitations")

# Supabase auth API is fast; an invite must never hang the request.
HTTP_TIMEOUT_S = 20.0

# Practical RFC-5322-lite validation (the definitive check is Supabase's).
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# The only role this endpoint ever assigns (SIGMO_RULES invite policy).
DEFAULT_ROLE = "technician"


class InviteError(Exception):
    """Base invite failure — carries an HTTP status and a stable
    machine-readable code the frontend localizes (EN/AM)."""

    status: int = 502
    code: str = "invite_failed"

    def __init__(
        self, message: str, *, status: int | None = None, code: str | None = None
    ) -> None:
        super().__init__(message)
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code


class InviteNotConfigured(InviteError):
    status = 503
    code = "invite_not_configured"


class MissingSession(InviteError):
    status = 401
    code = "missing_session"


class InvalidSession(InviteError):
    status = 401
    code = "invalid_session"


class NotAdmin(InviteError):
    status = 403
    code = "not_admin"


class AlreadyRegistered(InviteError):
    status = 409
    code = "already_registered"


class RateLimited(InviteError):
    status = 429
    code = "email_rate_limited"


def normalize_email(raw: str) -> str:
    """Strip/lower/validate. Raises 400 ``invalid_email`` on garbage."""
    email = (raw or "").strip().lower()
    if not email or len(email) > 254 or not EMAIL_RE.match(email):
        raise InviteError(
            "Enter a valid email address.", status=400, code="invalid_email"
        )
    return email


def ensure_configured() -> None:
    if not DB.supabase_url or not DB.supabase_service_role_key:
        raise InviteNotConfigured(
            "Invitations are not configured — set SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY in the service environment."
        )


def _base_url() -> str:
    return DB.supabase_url.rstrip("/")


def _service_headers() -> dict[str, str]:
    return {
        "apikey": DB.supabase_service_role_key,
        "Authorization": f"Bearer {DB.supabase_service_role_key}",
    }


async def introspect_token(access_token: str) -> dict:
    """Verify the caller's Supabase session server-side and return the
    auth-user payload (id/email/...). Raises InvalidSession on anything
    but a clean 200."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            resp = await client.get(
                f"{_base_url()}/auth/v1/user",
                headers={**_service_headers(), "Authorization": f"Bearer {access_token}"},
            )
    except httpx.HTTPError as exc:
        raise InviteError(
            f"Supabase auth unreachable: {exc}",
            status=502,
            code="supabase_unreachable",
        ) from exc
    if resp.status_code != 200:
        raise InvalidSession("Your session is invalid or expired — sign in again.")
    data = resp.json()
    if not isinstance(data, dict) or not data.get("id"):
        raise InvalidSession("Unexpected Supabase auth response.")
    return data


async def send_invite(email: str) -> dict:
    """POST /auth/v1/invite with the service key. Supabase itself sends
    the invitation email (free tier: 2 auth emails/hour); the user sets
    their own password from the link. Every failure mode maps to a
    stable code."""
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            resp = await client.post(
                f"{_base_url()}/auth/v1/invite",
                headers=_service_headers(),
                json={"email": email},
            )
    except httpx.HTTPError as exc:
        raise InviteError(
            f"Supabase invite request failed: {exc}",
            status=502,
            code="supabase_unreachable",
        ) from exc
    if resp.status_code == 200:
        data = resp.json()
        return data if isinstance(data, dict) else {}
    body = resp.text[:300].lower()
    # Existing account (confirmed or still pending): honest 409, no email.
    if resp.status_code in (400, 409, 422) and "already" in body:
        raise AlreadyRegistered("This email address is already registered.")
    if resp.status_code == 429 or "rate limit" in body:
        raise RateLimited(
            "Supabase email rate limit reached (free tier: 2 per hour) — "
            "try again later."
        )
    logger.error("Supabase invite failed %s: %s", resp.status_code, body)
    raise InviteError(
        f"Supabase invite failed (HTTP {resp.status_code}).",
        status=502,
        code="invite_failed",
    )
