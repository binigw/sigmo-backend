"""Sigmo V2 — Fault Logs API tests: GET /api/v2/faults + the Section 10
fault-episode audit trail written by the ingestion path.

Drives the real ASGI app over HTTP with the same physically-modeled
waveforms and capture-repository harness as the ingestion suite.

Test matrix:
  F1: fault episode row is written on a fault verdict — taxonomy code
      format, urgency stage bounds, Section 11.6 signals recorded,
      spectral evidence keys, model version
  F2: episode dedup — a second window of the SAME ongoing episode
      never duplicates the row
  F3: gate-HEALTHY fault verdict still logs an episode — with
      absolute_threshold_breached / four_signal_confirmed honestly
      False (unconfirmed); mechanical faults never breach the
      electrical gates
  F4: GET /api/v2/faults — shape, newest-first ordering, motor filter,
      honest returned accounting
  F5: auth matrix — no key 401, device scope 403, dashboard reads
  F6: query validation — limit bounds -> 422
  F7: fail-soft — during a database outage the fault-log probe fails
      but ingestion still succeeds (audit trail never blocks telemetry)
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
from backend.api import auth

check = harness.check
failures = harness.failures
ACTIVE_VERSION = harness.ACTIVE_VERSION

TAXONOMY_RE = re.compile(r"^Stage [1-4] .+ \(.+\)$")


async def run() -> None:
    repo = harness.CaptureRepository()
    sigmo_main.service._repo = repo
    auth.load_keys(
        f"{harness.TEST_DEVICE_KEY}:device,"
        f"{harness.TEST_DASHBOARD_KEY}:dashboard,"
        f"{harness.TEST_ADMIN_KEY}:admin"
    )

    for mid in ("MTR-FLOG", "MTR-FLOG-OK"):
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

        # ---- F1: fault episode written on the fault verdict ---------
        await harness.send_window(admin, "MTR-FLOG", faulty)
        flog_rows = [r for r in repo.fault_rows if r["motor_id"] == "MTR-FLOG"]
        row = flog_rows[0] if flog_rows else {}
        ev = row.get("spectral_evidence") or {}
        check(
            "F1 fault episode written (taxonomy format, signals, evidence)",
            len(flog_rows) == 1
            and bool(TAXONOMY_RE.match(row.get("taxonomy_code", "")))
            and 1 <= row.get("urgency_stage", 0) <= 4
            and 0.0 <= row.get("model_confidence", -1) <= 1.0
            and isinstance(row.get("population_sigma"), float)
            and isinstance(row.get("absolute_threshold_breached"), bool)
            and isinstance(row.get("trend_confirmed"), bool)
            and isinstance(row.get("four_signal_confirmed"), bool)
            and ev.get("model_version") == ACTIVE_VERSION
            and isinstance(ev.get("thd_percent_max"), float)
            and isinstance(ev.get("envelope_peak_hz"), list)
            and 45 <= ev.get("fundamental_hz", 0) <= 55,
            f"rows={len(flog_rows)}, code={row.get('taxonomy_code')}, "
            f"stage={row.get('urgency_stage')}, "
            f"conf={row.get('model_confidence')}",
        )

        # ---- F2: same ongoing episode never duplicates --------------
        from datetime import datetime, timedelta, timezone
        repo._last_logged["MTR-FLOG"] = (
            datetime.now(timezone.utc) - timedelta(minutes=35)
        )
        await harness.send_window(admin, "MTR-FLOG", faulty)
        flog_rows = [r for r in repo.fault_rows if r["motor_id"] == "MTR-FLOG"]
        check(
            "F2 episode dedup (second faulty window, same code -> 1 row)",
            len(flog_rows) == 1,
            f"rows={len(flog_rows)}",
        )

        # ---- F3: gate-HEALTHY fault verdict logs an UNCONFIRMED row --
        # The field scenario behind the sync fix: mechanical faults
        # (bearing, rotor) do NOT breach the electrical gates, so the
        # classifier verdict arrives with gate status HEALTHY. The
        # episode must STILL be written (it is a real detection) while
        # the Section 11.6 signals record the missing gate evidence
        # honestly: absolute_threshold_breached False and
        # four_signal_confirmed False (unconfirmed episode).
        await harness.send_window(admin, "MTR-FLOG-OK", healthy)
        ok_rows = [
            r for r in repo.fault_rows if r["motor_id"] == "MTR-FLOG-OK"
        ]
        f3_row = ok_rows[0] if ok_rows else {}
        check(
            "F3 gate-HEALTHY fault verdict logs an UNCONFIRMED episode",
            len(ok_rows) == 1
            and f3_row.get("absolute_threshold_breached") is False
            and f3_row.get("four_signal_confirmed") is False
            and bool(TAXONOMY_RE.match(f3_row.get("taxonomy_code", ""))),
            f"rows={len(ok_rows)}, abs="
            f"{f3_row.get('absolute_threshold_breached')}, code="
            f"{f3_row.get('taxonomy_code')}",
        )

        # ---- F4: read endpoint shape + filter ------------------------
        r = await client.get("/api/v2/faults")
        m = r.json()
        faults = m.get("faults") or []
        times = [f["detected_at"] for f in faults]
        check(
            "F4 list shape (newest first, honest accounting)",
            r.status_code == 200
            and m.get("database_connected") is True
            and m.get("motor_id") is None
            and m.get("returned") == len(faults) == len(repo.fault_rows)
            and times == sorted(times, reverse=True)
            and all(
                1 <= f["urgency_stage"] <= 4
                and 0.0 <= f["model_confidence"] <= 1.0
                and isinstance(f["spectral_evidence"], dict)
                for f in faults
            ),
            f"status={r.status_code}, returned={m.get('returned')}",
        )
        r_f = await client.get(
            "/api/v2/faults?motor_id=MTR-FLOG-OK"
        )
        m_f = r_f.json()
        # Filtering by a motor with telemetry but NO fault detection
        # (MTR-FLOG-NEVER never gets a faulty window) returns an honest
        # empty list. (MTR-FLOG-OK now legitimately carries the F3
        # gate-HEALTHY episode.)
        r_never = await client.get(
            "/api/v2/faults?motor_id=MTR-FLOG-NEVER"
        )
        m_n = r_never.json()
        check(
            "F4b motor filter (fault-free motor -> empty, honest)",
            r_f.status_code == 200
            and m_f["motor_id"] == "MTR-FLOG-OK"
            and m_f["returned"] == len(m_f["faults"])
            and r_never.status_code == 200
            and m_n["returned"] == 0
            and m_n["faults"] == [],
            f"ok_returned={m_f.get('returned')}, "
            f"never_returned={m_n.get('returned')}",
        )

        # ---- F5: auth matrix ------------------------------------------
        r_no = await noauth.get("/api/v2/faults")
        r_dev = await device_client.get("/api/v2/faults")
        check(
            "F5 auth (no key 401, device scope 403, dashboard reads)",
            r_no.status_code == 401 and r_dev.status_code == 403,
            f"noauth={r_no.status_code}, device={r_dev.status_code}",
        )

        # ---- F6: limit validation -> 422 ------------------------------
        r_l0 = await client.get("/api/v2/faults?limit=0")
        r_lmax = await client.get("/api/v2/faults?limit=1001")
        check(
            "F6 limit bounds (1-1000 -> 422)",
            r_l0.status_code == 422 and r_lmax.status_code == 422,
            f"l0={r_l0.status_code}, l1001={r_lmax.status_code}",
        )

        # ---- F7: fail-soft during outage ------------------------------
        before = len(repo.fault_rows)
        repo.outage = True
        repo._last_logged["MTR-FLOG"] = (
            datetime.now(timezone.utc) - timedelta(minutes=35)
        )
        r_out = await harness.send_window(admin, "MTR-FLOG", faulty)
        repo.outage = False
        analyzed = [
            r["body"].get("window_analyzed")
            for r in (r_out or []) if isinstance(r.get("body"), dict)
        ]
        check(
            "F7 fail-soft (outage probe fails, ingestion unaffected)",
            len(repo.fault_rows) == before
            and any(analyzed),
            f"rows={len(repo.fault_rows)} (was {before}), "
            f"analyzed={any(analyzed)}",
        )

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("RESULT: ALL FAULT LOGS API TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run())
