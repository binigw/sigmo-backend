"""Sigmo V2 — AI explanation client (real LLM integration).

Bridges raw V2 telemetry evidence and field technicians: feeds the
actual measured values (Health Index, THD, unbalance, crest factor,
rotor sidebands, per-phase RMS), the ACTIVE v6c verdict, the
deterministic stage/RUL and the system's own Amharic repair protocol
to a real LLM, which must answer as an expert industrial motor
technician in BOTH Amharic and English.

Providers (first configured key wins, or set AI_PROVIDER explicitly):
  openai    -> OPENAI_API_KEY    (model OPENAI_MODEL, default gpt-4o-mini)
  deepseek  -> DEEPSEEK_API_KEY  (model DEEPSEEK_MODEL, default deepseek-chat)
  gemini    -> GEMINI_API_KEY    (model GEMINI_MODEL, default gemini-3.8-flash)

Zero-fake-code directive: with no key configured the caller gets an
honest AIProviderNotConfigured error (HTTP 503 with setup guidance) —
never a fabricated "AI" response. All HTTP calls go through httpx with
a hard timeout; every provider failure surfaces as AIExplanationError.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger("sigmo.ai")

DEFAULT_TIMEOUT_S = 90.0
DEFAULT_MAX_OUTPUT_TOKENS = 2048

PROVIDER_ENV: dict[str, tuple[str, str, str]] = {
    # provider -> (api key env var, model env var, default model)
    "openai": ("OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o-mini"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"),
    "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL", "gemini-3.8-flash"),
}

SYSTEM_PROMPT = """\
You are a senior industrial electric-motor diagnostics technician with
20+ years of hands-on experience with MCC panels, squirrel-cage
induction motors, MCSA (Motor Current Signature Analysis), bearing
fault signatures (BPFO, BPFI, BSF, FTF), broken rotor bars, stator
winding interturn shorts, and power-quality standards (IEEE 519,
IEEE 1159, NEMA MG-1).

You are given REAL measured telemetry of one motor from a monitoring
system: the latest steady-state window, the recent window history, the
ML classifier verdict (13-class taxonomy) with confidence and
probabilities, the deterministic urgency stage (1-4) and the Remaining
Useful Life (RUL) estimate. A reference repair protocol from the
system's own rulebook is included as ground truth.

Your task — write an analysis for the LOCAL FIELD TECHNICIAN that:

1. ROOT CAUSE: explains exactly WHAT the measured values indicate and
   WHY they point to the diagnosed fault. Cite the actual numbers you
   were given (e.g. "THD 6.9% exceeds the IEEE 519 5% limit",
   "current unbalance 8.3% is far above the 2% limit",
   "crest factor 1.48 vs the 1.40-1.45 healthy band",
   "rotor sideband level ... dB"). Never invent a number that is not
   in the data; if evidence is insufficient, say so plainly.
2. STEP-BY-STEP ACTIONS: an ordered, concrete checklist the technician
   can execute with typical plant tools (clamp meter, vibration pen,
   megger, feeler gauge...). ALWAYS start physical work with power
   isolation / lockout-tagout. Include what to measure to CONFIRM the
   diagnosis and what result means "replace now" vs "monitor".
3. If the verdict is HEALTHY, explain the healthy evidence and give
   preventive-maintenance steps instead.

Language rules (BOTH sections are mandatory):
- "analysis_am": clear, practical Amharic for Ethiopian field
  technicians. Keep technical terms in Latin script inside parentheses
  where natural (e.g. "ቤሪንግ (Bearing)", "THD", "ክሬስት ፋክተር (Crest
  Factor)"). Use short numbered steps, not long paragraphs.
- "analysis_en": clear professional English covering the same content.

Respond with STRICT JSON only, no markdown fences:
{"analysis_am": "...", "analysis_en": "..."}
If JSON mode is unavailable to you, respond with exactly two marker
sections:
===AMHARIC===
<amharic text>
===ENGLISH===
<english text>
"""


class AIProviderNotConfigured(RuntimeError):
    """No LLM API key is configured — honest 503, never a fake answer."""


class AIExplanationError(RuntimeError):
    """The configured provider failed or returned an unusable answer."""


# Zero-width / bidi control characters survive .strip() but break real API
# keys when a key is pasted from chat apps or formatted documents (seen in
# production: a trailing U+200E left-to-right mark caused key rejection).
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")


def _clean_secret(raw: str | None) -> str:
    """Strip whitespace and invisible control characters from a secret."""
    if not raw:
        return ""
    return _INVISIBLE_RE.sub("", raw).strip()


def resolve_provider() -> tuple[str, str, str]:
    """(provider, api_key, model) from the environment.

    Priority: explicit AI_PROVIDER, then the first provider (openai,
    deepseek, gemini) whose API key env var is set. Raises
    AIProviderNotConfigured with exact setup guidance when nothing is
    configured.
    """
    explicit = os.environ.get("AI_PROVIDER", "").strip().lower()
    order: list[str]
    if explicit:
        if explicit not in PROVIDER_ENV:
            raise AIProviderNotConfigured(
                f"AI_PROVIDER={explicit!r} is unknown. Valid: "
                + ", ".join(PROVIDER_ENV)
            )
        order = [explicit] + [
            p for p in PROVIDER_ENV if p != explicit
        ]
    else:
        order = list(PROVIDER_ENV)
    for provider in order:
        key_env, model_env, default_model = PROVIDER_ENV[provider]
        api_key = _clean_secret(os.environ.get(key_env))
        if api_key:
            model = _clean_secret(os.environ.get(model_env)) or default_model
            return provider, api_key, model
    raise AIProviderNotConfigured(
        "No AI provider configured. Add ONE of OPENAI_API_KEY, "
        "DEEPSEEK_API_KEY or GEMINI_API_KEY to the service environment "
        "(Render -> Environment). Optional: AI_PROVIDER to force one, "
        "and OPENAI_MODEL / DEEPSEEK_MODEL / GEMINI_MODEL to override "
        "the model."
    )


def _parse_analysis(raw_text: str) -> dict[str, str]:
    """Extract analysis_am / analysis_en from the model output.

    Two tolerant strategies: strict JSON first, then the ===MARKER===
    fallback. Anything else is an honest failure.
    """
    text = raw_text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    # strategy 1: JSON object
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            am = data.get("analysis_am")
            en = data.get("analysis_en")
            if isinstance(am, str) and am.strip() and isinstance(en, str) and en.strip():
                return {"analysis_am": am.strip(), "analysis_en": en.strip()}
        except (json.JSONDecodeError, ValueError):
            pass
    # strategy 2: marker sections
    if "===AMHARIC===" in text and "===ENGLISH===" in text:
        am = text.split("===AMHARIC===", 1)[1].split("===ENGLISH===", 1)[0]
        en = text.split("===ENGLISH===", 1)[1]
        if am.strip() and en.strip():
            return {"analysis_am": am.strip(), "analysis_en": en.strip()}
    raise AIExplanationError(
        "The AI response could not be parsed into Amharic and English "
        "sections. Try again or switch AI_PROVIDER/model."
    )


async def _call_openai_compatible(
    provider: str, api_key: str, model: str, context_json: str
) -> str:
    """OpenAI-compatible chat completion (openai, deepseek)."""
    base = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com",
    }[provider]
    base = os.environ.get(f"{provider.upper()}_BASE_URL", base).rstrip("/")
    body = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_json},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
            response = await client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
        if response.status_code != 200:
            raise AIExplanationError(
                f"{provider} API returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])
    except httpx.HTTPError as exc:
        raise AIExplanationError(f"{provider} request failed: {exc}") from exc
    except (KeyError, IndexError, ValueError) as exc:
        raise AIExplanationError(
            f"{provider} response was missing choices/message content"
        ) from exc


async def _call_gemini(
    api_key: str, model: str, context_json: str
) -> str:
    """Google Gemini generateContent."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": context_json}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_S) as client:
            response = await client.post(
                url, headers={"x-goog-api-key": api_key}, json=body
            )
        if response.status_code != 200:
            raise AIExplanationError(
                f"gemini API returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        payload = response.json()
        return str(
            payload["candidates"][0]["content"]["parts"][0]["text"]
        )
    except httpx.HTTPError as exc:
        raise AIExplanationError(f"gemini request failed: {exc}") from exc
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AIExplanationError(
            "gemini response was missing candidates/content"
        ) from exc


async def generate_explanation(context: dict[str, Any]) -> dict[str, str]:
    """Run the configured provider on the motor context.

    Returns {"provider", "model", "analysis_am", "analysis_en"}.
    Raises AIProviderNotConfigured / AIExplanationError.
    """
    provider, api_key, model = resolve_provider()
    context_json = json.dumps(context, ensure_ascii=False, indent=1)
    logger.info(
        "AI explanation request provider=%s model=%s context_bytes=%d",
        provider, model, len(context_json),
    )
    if provider == "gemini":
        raw = await _call_gemini(api_key, model, context_json)
    else:
        raw = await _call_openai_compatible(
            provider, api_key, model, context_json
        )
    result = _parse_analysis(raw)
    logger.info(
        "AI explanation ok provider=%s model=%s am_chars=%d en_chars=%d",
        provider, model,
        len(result["analysis_am"]), len(result["analysis_en"]),
    )
    return {"provider": provider, "model": model, **result}
