"""Sigmo V2 — Alerts API tests: GET /api/v2/alerts + dismiss / restore.

Drives the real ASGI app over HTTP with the same physically-modeled
waveforms and capture-repository harness as the ingestion suite. Alerts
are DERIVED on read (stage >= 2 or an active fault verdict) — the suite
asserts the real derivation over the real service state, and that the
operator's dismissals behave like a live inbox (auto-expiring) instead
of touching the immutable Section 10 fault_logs audit trail.

Test matrix:
  AL1: plant with no telemetry yet -> zero alerts, honest payload shape
  AL2: faulty motor -> one derived alert (id/key format, severity vs
      stage consistency, verdict echo, evidence fields); every derived
      alert satisfies the derivation rule (Stage >= 2 or is_fault)
  AL3: auth matrix — no key 401, device scope 403 (GET + DELETE),
      dashboard 200; empty alert_ids body -> 422
  AL4: single dismiss (DELETE) hides the alert; repeat is idempotent;
      unknown id -> honest 404
  AL5: include_dismissed=true flags it with dismissed_at; restore
      (POST) brings it back to active
  AL6: bulk dismiss — per-item accounting including unknown ids
  AL7: dismissal auto-expires — condition clears (healthy window) ->
      stale dismissal forgotten; the fault re-triggers -> alerts again
  AL8: fail-soft — during a database outage GET still derives alerts
      (no dismissal filtering possible), dismiss answers honest 503
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "/home/user/sigmo_v2")
sys.path.insert(0, "/home/user/sigmo_v2/tests")

import httpx

import test_ingestion_api as harness
from backend import main as sigmo_main
from backend.api import auth

check = harness.check
failures = harness.failures

SEVERITY_FOR_STAGE = {
    1: "medium",
    2: "high",
    3: "critical",
    4: "critical",
}


async def run() -> None:
    repo = harness.CaptureRepository()
    sigmo_main.service._repo = repo
    sigmo_main.repository = repo
    auth.load_keys(
        f"{harness.TEST_DEVICE_KEY}:device,"
        f"{harness.TEST_DASHBOARD_KEY}:dashboard,"
        f"{harness.TEST_ADMIN_KEY}:admin"
    )

    for mid in ("MTR-ALR", "MTR-ALR-OK", "MTR-ALR-2"):
        await repo.upsert_motor_asset({
            "motor_id": mid, "mcc_panel_id": "MCC-01",
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
        healthy = harness.three_phase_window((10.0, 10.0, 10.0))

        # ---- AL1: no telemetry yet -> zero alerts -------------------
        r = await client.get("/api/v2/alerts")
        body = r.json() if r.status_code == 200 else {}
        check(
            "AL1 no telemetry yet -> zero alerts (registered-but-silent "
            "motors never alarm)",
            r.status_code == 200
            and body.get("total_active") == 0
            and body.get("alerts") == []
            and "generated_at" in body
            and isinstance(body.get("database_connected"), bool),
            f"status={r.status_code}, body={str(body)[:120]}",
        )

        # ---- AL2: faulty motor -> one derived alert -----------------
        await harness.send_window(admin, "MTR-ALR", faulty)
        await harness.send_window(admin, "MTR-ALR-OK", healthy)
        r = await client.get("/api/v2/alerts")
        body = r.json() if r.status_code == 200 else {}
        alerts = body.get("alerts") or []
        target = next(
            (a for a in alerts if a.get("motor_id") == "MTR-ALR"), None
        )
        ok_shape = target is not None and (
            target.get("alert_id") == f"MTR-ALR:{target.get('alert_key')}"
            and target.get("predicted_class") in harness.CLASS_NAMES
            and target.get("is_fault") is True
            and isinstance(target.get("stage"), int)
            and 1 <= target["stage"] <= 4
            and target.get("severity") == SEVERITY_FOR_STAGE.get(
                target["stage"]
            )
            and target.get("dismissed") is False
            and isinstance(target.get("model_confidence"), (int, float))
            and "recorded_at" in target
        )
        derivation_rule = all(
            (a.get("stage") or 0) >= 2 or a.get("is_fault")
            for a in alerts
        )
        check(
            "AL2 faulty motor -> derived alert (id/key, severity vs "
            "stage, verdict echo, derivation rule)",
            ok_shape and derivation_rule,
            f"target={str(target)[:160]}, "
            f"rule_holds={derivation_rule} on {len(alerts)} alerts",
        )
        alert_id = target["alert_id"] if target else ""

        # ---- AL3: auth matrix ----------------------------------------
        r_no = await noauth.get("/api/v2/alerts")
        r_dev = await device_client.get("/api/v2/alerts")
        r_dev_del = await device_client.delete(
            f"/api/v2/alerts/{alert_id}"
        )
        r_dash = await client.get("/api/v2/alerts")
        r_empty = await client.post(
            "/api/v2/alerts/dismiss", json={"alert_ids": []}
        )
        check(
            "AL3 auth (no key 401, device 403 GET+DELETE, dashboard "
            "200, empty ids 422)",
            r_no.status_code == 401
            and r_dev.status_code == 403
            and r_dev_del.status_code == 403
            and r_dash.status_code == 200
            and r_empty.status_code == 422,
            f"noauth={r_no.status_code}, dev={r_dev.status_code}, "
            f"dev_del={r_dev_del.status_code}, dash={r_dash.status_code}, "
            f"empty={r_empty.status_code}",
        )

        # ---- AL4: single dismiss hides; idempotent; unknown 404 -----
        r_del = await client.delete(f"/api/v2/alerts/{alert_id}")
        r_after = await client.get("/api/v2/alerts")
        still = any(
            a.get("alert_id") == alert_id
            for a in (r_after.json().get("alerts") or [])
        )
        r_del2 = await client.delete(f"/api/v2/alerts/{alert_id}")
        r_bogus = await client.delete("/api/v2/alerts/MTR-NOPE:GHOST")
        check(
            "AL4 dismiss hides alert (idempotent repeat, unknown 404)",
            r_del.status_code == 200
            and r_del.json().get("affected") == 1
            and not still
            and r_del2.status_code == 200
            and r_bogus.status_code == 404,
            f"del={r_del.status_code}, still_visible={still}, "
            f"del2={r_del2.status_code}, bogus={r_bogus.status_code}",
        )

        # ---- AL5: include_dismissed + restore ------------------------
        r_inc = await client.get("/api/v2/alerts?include_dismissed=true")
        flagged = next(
            (
                a
                for a in (r_inc.json().get("alerts") or [])
                if a.get("alert_id") == alert_id
            ),
            None,
        )
        r_restore = await client.post(
            "/api/v2/alerts/restore", json={"alert_ids": [alert_id]}
        )
        r_back = await client.get("/api/v2/alerts")
        back_ids = [
            a.get("alert_id") for a in (r_back.json().get("alerts") or [])
        ]
        check(
            "AL5 include_dismissed flags with timestamp; restore "
            "reactivates",
            r_inc.status_code == 200
            and flagged is not None
            and flagged.get("dismissed") is True
            and flagged.get("dismissed_at")
            and r_restore.status_code == 200
            and r_restore.json().get("affected") == 1
            and alert_id in back_ids,
            f"flagged={str(flagged)[:120]}, restore="
            f"{r_restore.status_code}, back_active={alert_id in back_ids}",
        )

        # ---- AL6: bulk dismiss with per-item accounting --------------
        await harness.send_window(admin, "MTR-ALR-2", faulty)
        r_two = await client.get("/api/v2/alerts")
        ids = [
            a["alert_id"] for a in (r_two.json().get("alerts") or [])
        ]
        r_bulk = await client.post(
            "/api/v2/alerts/dismiss",
            json={"alert_ids": [*ids, "MTR-NOPE:GHOST"]},
        )
        bulk = r_bulk.json() if r_bulk.status_code == 200 else {}
        r_zero = await client.get("/api/v2/alerts")
        check(
            "AL6 bulk dismiss (per-item accounting, unknowns echoed)",
            r_bulk.status_code == 200
            and bulk.get("requested") == len(ids) + 1
            and bulk.get("affected") == len(ids)
            and bulk.get("unknown") == ["MTR-NOPE:GHOST"]
            and r_zero.json().get("total_active") == 0,
            f"bulk={str(bulk)[:140]}",
        )

        # ---- AL7: dismissal auto-expires when condition changes -----
        # MTR-ALR's BEARING_OUTER_RACE alert is dismissed (AL6). The
        # condition then changes (healthy waveform -> the model reads a
        # DIFFERENT signature): the stale dismissal must be forgotten so
        # the original fault alerts again when it re-triggers.
        rows_before = {
            (r["motor_id"], r["alert_key"])
            for r in repo.alert_dismissal_rows
        }
        await harness.send_window(admin, "MTR-ALR", healthy)
        r_chg = await client.get("/api/v2/alerts")
        chg_alerts = r_chg.json().get("alerts") or []
        alr_now = next(
            (a for a in chg_alerts if a.get("motor_id") == "MTR-ALR"), None
        )
        alr_rows = [
            r
            for r in repo.alert_dismissal_rows
            if r["motor_id"] == "MTR-ALR"
        ]
        others_preserved = {
            (r["motor_id"], r["alert_key"])
            for r in repo.alert_dismissal_rows
        } <= rows_before
        await harness.send_window(admin, "MTR-ALR", faulty)
        r_re = await client.get("/api/v2/alerts")
        re_ids = [
            a.get("alert_id") for a in (r_re.json().get("alerts") or [])
        ]
        check(
            "AL7 dismissal auto-expires (condition changed -> forgotten; "
            "re-trigger alerts again)",
            alr_now is not None
            and alr_now.get("alert_key") != "BEARING_OUTER_RACE"
            and len(alr_rows) == 0
            and others_preserved
            and "MTR-ALR:BEARING_OUTER_RACE" in re_ids,
            f"new_key={alr_now and alr_now.get('alert_key')}, "
            f"alr_rows_left={len(alr_rows)}, "
            f"others_preserved={others_preserved}, "
            f"re_alerted={'MTR-ALR:BEARING_OUTER_RACE' in re_ids}",
        )

        # ---- AL8: fail-soft during a DB outage -----------------------
        repo.outage = True
        r_out = await client.get("/api/v2/alerts")
        r_out_dismiss = await client.delete(
            f"/api/v2/alerts/{re_ids[0] if re_ids else 'MTR-ALR:GHOST'}"
        )
        repo.outage = False
        check(
            "AL8 outage fail-soft (GET derives live alerts; dismiss "
            "answers honest 503)",
            r_out.status_code == 200
            and r_out.json().get("total_active", 0) >= 1
            and r_out_dismiss.status_code == 503,
            f"get={r_out.status_code}/"
            f"{r_out.json().get('total_active')}, "
            f"dismiss={r_out_dismiss.status_code}",
        )

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("RESULT: ALL ALERTS API TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run())
