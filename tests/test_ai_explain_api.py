"""Sigmo V2 — AI explanation API tests: POST /api/v2/ai/explain.

Drives the real ASGI app over HTTP with the same physically-modeled
waveforms and capture-repository harness as the ingestion suite. The
provider call is stubbed ONLY in AI4 (a test double over the real
route/context assembly, the same pattern as CaptureRepository) — the
suite asserts the REAL data context that would be fed to the LLM.

Test matrix:
  AI1: no provider key -> honest 503 with exact setup guidance
  AI2: auth matrix — no key header 401, device scope 403
  AI3: unknown motor -> 404 (context lookup precedes provider check)
  AI4: stubbed provider -> 200; context carries the real measured
       evidence (verdict, THD, stage), the reference limits and the
       Amharic protocol grounding; response echoes provider/model and
       both language sections
  AI5: provider resolution unit — explicit AI_PROVIDER, key priority,
       unknown provider refusal
  AI6: parser unit — strict JSON, marker fallback, honest failure
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "/home/user/sigmo_v2")
sys.path.insert(0, "/home/user/sigmo_v2/tests")

import httpx

import test_ingestion_api as harness
from backend import main as sigmo_main
from backend.ai import client as ai_client_module
from backend.ai.client import (
    AIProviderNotConfigured,
    _parse_analysis,
    resolve_provider,
)
from backend.api import auth

check = harness.check
failures = harness.failures


async def run() -> None:
    repo = harness.CaptureRepository()
    sigmo_main.service._repo = repo
    sigmo_main.repository = repo  # admin path (asset registry reads)
    auth.load_keys(
        f"{harness.TEST_DEVICE_KEY}:device,"
        f"{harness.TEST_DASHBOARD_KEY}:dashboard,"
        f"{harness.TEST_ADMIN_KEY}:admin"
    )

    # Ensure NO provider key leaks from the test environment.
    for var in (
        "AI_PROVIDER", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
    ):
        os.environ.pop(var, None)

    await repo.upsert_motor_asset({
        "motor_id": "MTR-AI", "mcc_panel_id": "MCC-01",
        "cabinet_number": "C-01", "line_zone_id": "LINE-A/ZONE-1",
        "nameplate_kw": 75.0, "rated_rpm": 2970, "drive_type": "DOL",
    })

    transport = httpx.ASGITransport(app=sigmo_main.app)
    async with (
        httpx.AsyncClient(
            transport=transport, base_url="http://sigmo-test",
            timeout=120.0, headers={"X-API-Key": harness.TEST_ADMIN_KEY},
        ) as admin,
        httpx.AsyncClient(
            transport=transport, base_url="http://sigmo-test",
            timeout=120.0, headers={"X-API-Key": harness.TEST_DASHBOARD_KEY},
        ) as client,
        httpx.AsyncClient(
            transport=transport, base_url="http://sigmo-test", timeout=120.0
        ) as noauth,
        httpx.AsyncClient(
            transport=transport, base_url="http://sigmo-test",
            timeout=120.0, headers={"X-API-Key": harness.TEST_DEVICE_KEY},
        ) as device_client,
    ):
        faulty = harness.three_phase_window((10.0, 10.0, 8.8), thd_frac=0.07)
        await harness.send_window(admin, "MTR-AI", faulty)

        # ---- AI1: honest 503 without a provider key ------------------
        r = await client.post("/api/v2/ai/explain", json={"motor_id": "MTR-AI"})
        detail = r.json().get("detail", "") if r.status_code != 200 else ""
        check(
            "AI1 no provider key -> 503 with setup guidance",
            r.status_code == 503
            and "OPENAI_API_KEY" in detail
            and "DEEPSEEK_API_KEY" in detail
            and "GEMINI_API_KEY" in detail,
            f"status={r.status_code}, detail={detail[:80]}",
        )

        # ---- AI2: auth matrix ----------------------------------------
        r_no = await noauth.post("/api/v2/ai/explain", json={"motor_id": "MTR-AI"})
        r_dev = await device_client.post(
            "/api/v2/ai/explain", json={"motor_id": "MTR-AI"}
        )
        check(
            "AI2 auth (no key 401, device scope 403)",
            r_no.status_code == 401 and r_dev.status_code == 403,
            f"noauth={r_no.status_code}, device={r_dev.status_code}",
        )

        # ---- AI3: unknown motor -> 404 --------------------------------
        r_404 = await client.post(
            "/api/v2/ai/explain", json={"motor_id": "MTR-NOPE"}
        )
        r_422 = await client.post("/api/v2/ai/explain", json={"motor_id": ""})
        check(
            "AI3 unknown motor 404, empty id 422 (before provider check)",
            r_404.status_code == 404 and r_422.status_code == 422,
            f"404={r_404.status_code}, 422={r_422.status_code}",
        )

        # ---- AI4: stubbed provider, real context ----------------------
        captured: dict = {}

        async def stub(context):
            captured.update(context)
            return {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "analysis_am": "ሙከራ ማብራሪያ (stub)",
                "analysis_en": "stub explanation",
            }

        original = sigmo_main.ai_client.generate_explanation
        sigmo_main.ai_client.generate_explanation = stub
        try:
            r = await client.post(
                "/api/v2/ai/explain", json={"motor_id": "MTR-AI"}
            )
        finally:
            sigmo_main.ai_client.generate_explanation = original
        body = r.json() if r.status_code == 200 else {}
        cw = captured.get("current_window") or {}
        proto = captured.get("system_reference_protocol_amharic") or ""
        check(
            "AI4 stubbed 200 (provider echo, both languages, elapsed)",
            r.status_code == 200
            and body.get("provider") == "deepseek"
            and body.get("model") == "deepseek-chat"
            and "ሙከራ" in body.get("analysis_am", "")
            and body.get("analysis_en") == "stub explanation"
            and isinstance(body.get("elapsed_ms"), int),
            f"status={r.status_code}",
        )
        check(
            "AI4b real evidence in the LLM context (verdict, THD, "
            "limits, protocol grounding, recent windows, nameplate)",
            cw.get("verdict_13class") in harness.CLASS_NAMES
            and isinstance(cw.get("thd_percent_max"), float)
            and 5.0 <= cw["thd_percent_max"] <= 10.0
            and 0.0 <= cw.get("verdict_confidence", -1) <= 1.0
            and isinstance(cw.get("verdict_probabilities_top5"), dict)
            and captured.get("reference_limits", {}).get(
                "thd_percent_max_ieee519"
            ) == 5.0
            and isinstance(captured.get("recent_windows"), list)
            and len(captured["recent_windows"]) >= 1
            and captured.get("motor", {}).get("nameplate_kw") == 75.0
            and len(proto) > 50
            and any("\u1200" <= ch <= "\u137F" for ch in proto),
            f"verdict={cw.get('verdict_13class')}, "
            f"thd={cw.get('thd_percent_max')}, proto={len(proto)} chars",
        )

    # ---- AI5: provider resolution unit -------------------------------
    os.environ["DEEPSEEK_API_KEY"] = "sk-test-ds"
    provider, key, model = resolve_provider()
    ok_ds = provider == "deepseek" and model == "deepseek-chat"
    os.environ["AI_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "sk-test-oa"
    provider2, _, model2 = resolve_provider()
    os.environ["AI_PROVIDER"] = "bogus"
    try:
        resolve_provider()
        bogus_refused = False
    except AIProviderNotConfigured:
        bogus_refused = True
    for var in (
        "AI_PROVIDER", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
    ):
        os.environ.pop(var, None)
    check(
        "AI5 provider resolution (key priority, AI_PROVIDER force, "
        "unknown refused)",
        ok_ds and provider2 == "openai" and model2 == "gpt-4o-mini"
        and bogus_refused,
        f"ds={ok_ds}, forced={provider2}/{model2}, bogus_refused={bogus_refused}",
    )

    # ---- AI6: response parser unit ------------------------------------
    try:
        _parse_analysis("no structure at all")
        parse_fail = False
    except Exception:
        parse_fail = True
    check(
        "AI6 parser (JSON, markers, honest failure)",
        _parse_analysis('{"analysis_am": "አ", "analysis_en": "e"}')
        == {"analysis_am": "አ", "analysis_en": "e"}
        and _parse_analysis(
            "===AMHARIC===\nአ\n===ENGLISH===\ne"
        ) == {"analysis_am": "አ", "analysis_en": "e"}
        and parse_fail,
        f"json_ok, markers_ok, garbage_fails={parse_fail}",
    )

    # ---- AI7: pasted-key hygiene + pinned default model ---------------
    os.environ["GEMINI_API_KEY"] = "AIza-clean\u200e\n"
    provider3, key3, model3 = resolve_provider()
    os.environ.pop("GEMINI_API_KEY", None)
    check(
        "AI7 key sanitization (invisible marks stripped, default model "
        "pinned)",
        provider3 == "gemini"
        and key3 == "AIza-clean"
        and model3 == "gemini-3.8-flash",
        f"provider={provider3}, key={key3!r}, model={model3}",
    )

    # ---- AI8: real-DB Decimal values survive prompt serialization ----
    import decimal
    import json as _json

    os.environ["GEMINI_API_KEY"] = "AIza-clean"

    async def fake_gemini(api_key, model, context_json):
        _json.loads(context_json)  # prompt MUST be valid JSON text
        return '{"analysis_am": "\u12a0", "analysis_en": "e"}'

    original_call = ai_client_module._call_gemini
    ai_client_module._call_gemini = fake_gemini
    try:
        res8 = await ai_client_module.generate_explanation(
            {
                "motor": {
                    "nameplate_kw": decimal.Decimal("75.0"),
                    "rated_rpm": decimal.Decimal("2970"),
                },
                "current_window": {
                    "thd_percent_max": decimal.Decimal("6.9")
                },
            }
        )
    finally:
        ai_client_module._call_gemini = original_call
        os.environ.pop("GEMINI_API_KEY", None)
    check(
        "AI8 DB Decimal values serialize into the LLM prompt",
        res8.get("provider") == "gemini"
        and res8.get("analysis_am") == "\u12a0"
        and res8.get("analysis_en") == "e",
        f"res={res8}",
    )

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("RESULT: ALL AI EXPLAIN API TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run())
