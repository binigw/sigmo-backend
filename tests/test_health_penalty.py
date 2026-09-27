"""Sigmo V2 — operator health-score consistency + episode-sync tests.

Commercial-release fix: a Stage 2 bearing fault showed "100 % HEALTH"
because the pure z-score mapping cannot see mechanical faults, and the
dashboard "Recent Fault Episodes" feed stayed empty because the episode
trigger required a gate ANOMALY that mechanical faults never produce.

Test matrix:
  H1: assessed_health_percent unit — deterministic stage bands
      (Stage 2 -> 60-80, Stage 3 -> 40-60, Stage 4 -> 0-40), z-position
      inside the band, healthy motors keep the pure z mapping
  H2: overview penalty over HTTP — gate-HEALTHY fault verdict (the
      exact MTR-PILOT-01 scenario) scores inside the Stage 2 band,
      never 100 %
  H3: cross-surface consistency — executive plant index equals the
      mean of the overview per-motor scores (one scoring rule)
  H4: episode sync — the same gate-HEALTHY fault detection opens a
      fault-episode row (unconfirmed: no gate breach, four-signal
      False) and surfaces in GET /api/v2/faults
  H5: reconciliation — a pre-existing active condition without a row
      (deploy bootstrap) is reconciled exactly once, idempotently,
      with honest provenance
  H6: AI context — the LLM sees the adjusted health and the stage,
      never "100 % health" next to a fault verdict
"""
from __future__ import annotations

import asyncio
import re
import sys

sys.path.insert(0, "/home/user/sigmo_v2")
sys.path.insert(0, "/home/user/sigmo_v2/tests")

import httpx

import test_ingestion_api as harness
from backend import main as sigmo_main
from backend.analysis.staging import (
    FAULT_STAGE_HEALTH_BANDS,
    assessed_health_percent,
    motor_health_percent,
)
from backend.api import auth

check = harness.check
failures = harness.failures


async def run() -> None:
    repo = harness.CaptureRepository()
    sigmo_main.service._repo = repo
    sigmo_main.repository = repo
    auth.load_keys(
        f"{harness.TEST_DEVICE_KEY}:device,"
        f"{harness.TEST_DASHBOARD_KEY}:dashboard,"
        f"{harness.TEST_ADMIN_KEY}:admin"
    )
    for var in (
        "AI_PROVIDER", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
    ):
        import os
        os.environ.pop(var, None)

    await repo.upsert_motor_asset({
        "motor_id": "MTR-HLT", "mcc_panel_id": "MCC-01",
        "cabinet_number": "C-01", "line_zone_id": "LINE-A/ZONE-1",
        "nameplate_kw": 75.0, "rated_rpm": 2970, "drive_type": "DOL",
    })

    # ---- H1: unit bands -------------------------------------------
    h1_checks = []
    for stage, (floor, ceiling) in FAULT_STAGE_HEALTH_BANDS.items():
        top = assessed_health_percent(
            0.0, stage=stage, predicted_class="BEARING_OUTER_RACE"
        )
        bottom = assessed_health_percent(
            5.0, stage=stage, predicted_class="BEARING_OUTER_RACE"
        )
        h1_checks.append(
            top == ceiling and bottom == floor
            and floor <= top <= 100.0
        )
    mid = assessed_health_percent(
        2.5, stage=2, predicted_class="X"
    )  # z-health 50 -> band middle
    healthy = assessed_health_percent(
        0.0, stage=1, predicted_class="HEALTHY"
    )
    no_verdict = assessed_health_percent(1.0, stage=1, predicted_class=None)
    anomaly_stage = assessed_health_percent(
        0.0, stage=2, predicted_class=None
    )  # gate-anomaly-driven Stage 2 penalizes too
    check(
        "H1 stage bands (S2 60-80, S3 40-60, S4 0-40, z-position "
        "inside band, healthy keeps pure z mapping)",
        all(h1_checks)
        and mid == 70.0
        and healthy == 100.0
        and no_verdict == motor_health_percent(1.0)
        and anomaly_stage == 80.0,
        f"bands={h1_checks}, mid={mid}, healthy={healthy}, "
        f"anomaly_stage2={anomaly_stage}",
    )

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
    ):
        # Clean 3-phase set: the classifier reads a fault (interturn)
        # while the electrical gates stay HEALTHY — the field scenario.
        wave = harness.three_phase_window((10.0, 10.0, 10.0))
        await harness.send_window(admin, "MTR-HLT", wave)

        # ---- H2: overview penalty (gate-HEALTHY fault verdict) ----
        r = await client.get("/api/v2/motors")
        entry = next(
            (
                m
                for m in (r.json().get("motors") or [])
                if m["motor_id"] == "MTR-HLT"
            ),
            None,
        )
        check(
            "H2 overview health penalized (Stage 2 band, never 100 %)",
            r.status_code == 200
            and entry is not None
            and entry["stage"] == 2
            and entry["is_fault"] is True
            and entry["status"] == "HEALTHY"
            and 60.0 <= entry["health_percent"] <= 80.0
            and entry["health_percent"] < 100.0,
            f"stage={entry and entry['stage']}, "
            f"gate={entry and entry['status']}, "
            f"health={entry and entry['health_percent']}",
        )

        # ---- H3: executive == overview mean (one scoring rule) ----
        r_exec = await client.get("/api/v2/views/executive")
        exec_payload = r_exec.json()
        overview_mean = None
        motors = r.json().get("motors") or []
        if motors:
            overview_mean = round(
                sum(m["health_percent"] for m in motors) / len(motors), 1
            )
        check(
            "H3 executive plant index == mean of overview scores",
            r_exec.status_code == 200
            and overview_mean is not None
            and exec_payload.get("plant_health_index") == overview_mean
            and exec_payload.get("plant_health_index") < 100.0,
            f"exec={exec_payload.get('plant_health_index')}, "
            f"overview_mean={overview_mean}",
        )

        # ---- H4: episode sync for the same detection ---------------
        r_f = await client.get("/api/v2/faults?motor_id=MTR-HLT")
        faults = r_f.json().get("faults") or []
        check(
            "H4 gate-HEALTHY detection surfaces in fault episodes "
            "(unconfirmed flags honest)",
            r_f.status_code == 200
            and len(faults) == 1
            and faults[0]["motor_id"] == "MTR-HLT"
            and faults[0]["urgency_stage"] == 2
            and faults[0]["absolute_threshold_breached"] is False
            and faults[0]["four_signal_confirmed"] is False,
            f"faults={len(faults)}, "
            f"abs={faults and faults[0]['absolute_threshold_breached']}",
        )

        # ---- H5: reconciliation (deploy bootstrap) ------------------
        repo.fault_rows = []  # simulate the pre-fix production state
        written = await sigmo_main.service.reconcile_fault_episodes()
        written_again = await sigmo_main.service.reconcile_fault_episodes()
        rows = [
            r_ for r_ in repo.fault_rows if r_["motor_id"] == "MTR-HLT"
        ]
        ev = (rows[0].get("spectral_evidence") or {}) if rows else {}
        check(
            "H5 reconciliation opens the missing episode exactly once "
            "(idempotent, honest provenance)",
            written == 1
            and written_again == 0
            and len(rows) == 1
            and bool(
                re.match(
                    r"^Stage [1-4] .+ \(.+\)$",
                    str(rows[0].get("taxonomy_code", "")),
                )
            )
            and ev.get("detected_via") == "startup_reconciliation"
            and isinstance(ev.get("thd_percent_max"), float),
            f"written={written}/{written_again}, rows={len(rows)}, "
            f"via={ev.get('detected_via')}",
        )

        # ---- H6: AI context carries adjusted health + stage ---------
        captured: dict = {}

        async def stub(context):
            captured.update(context)
            return {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "analysis_am": "ማብራሪያ",
                "analysis_en": "explanation",
            }

        original = sigmo_main.ai_client.generate_explanation
        sigmo_main.ai_client.generate_explanation = stub
        try:
            r_ai = await client.post(
                "/api/v2/ai/explain", json={"motor_id": "MTR-HLT"}
            )
        finally:
            sigmo_main.ai_client.generate_explanation = original
        cw = captured.get("current_window") or {}
        check(
            "H6 AI context: adjusted health + stage (never 100 % next "
            "to a fault verdict)",
            r_ai.status_code == 200
            and cw.get("stage") == 2
            and 60.0 <= (cw.get("health_percent") or 0.0) <= 80.0,
            f"stage={cw.get('stage')}, "
            f"health={cw.get('health_percent')}",
        )

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("RESULT: ALL HEALTH-PENALTY TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run())
