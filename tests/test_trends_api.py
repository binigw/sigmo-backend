"""Sigmo V2 — Trends API tests: GET /api/v2/motors/{motor_id}/history.

Drives the real ASGI app over HTTP with the same physically-modeled
waveforms and capture-repository harness as the ingestion suite. The
history payload feeds the dashboard Analytics trends charts, so the
contract is verified field-by-field: honest window accounting (hours /
limit / returned), oldest-first ordering, the Section 14.5 health
index per window, worst-phase power-quality derivation, INRUSH
exclusion, and the Section 7 auth matrix.

Test matrix:
  T1: 200 shape — window echo, model_version, database_connected,
      ascending points
  T2: per-point contract — health index bounds, worst-phase THD of the
      physically modeled 7% THD window, rms sanity, verdict validity
  T3: 404 for a motor without any telemetry
  T4: 422 validation — hours=0, hours=721, limit=0
  T5: hours window filter + oldest-first ordering (backdated row)
  T6: limit cap keeps the NEWEST rows only
  T7: auth — no key 401; device key 403 (read is dashboard/admin);
      dashboard key reads (implicit in T1-T6)
  T8: INRUSH_SUPPRESSED windows never appear in the series
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/home/user/sigmo_v2")
sys.path.insert(0, "/home/user/sigmo_v2/tests")

import httpx

import test_ingestion_api as harness
from backend import main as sigmo_main
from backend.api import auth
from backend.ml.features import CLASS_NAMES

check = harness.check
failures = harness.failures
ACTIVE_VERSION = harness.ACTIVE_VERSION


async def run() -> None:
    repo = harness.CaptureRepository()
    sigmo_main.service._repo = repo
    auth.load_keys(
        f"{harness.TEST_DEVICE_KEY}:device,"
        f"{harness.TEST_DASHBOARD_KEY}:dashboard,"
        f"{harness.TEST_ADMIN_KEY}:admin"
    )

    # Commissioning precedes telemetry (Section 11 Cold Start) — mirrors
    # the production FK order even though the harness only simulates it.
    await repo.upsert_motor_asset({
        "motor_id": "MTR-TREND",
        "mcc_panel_id": "MCC-01",
        "cabinet_number": "C-01",
        "line_zone_id": "LINE-A/ZONE-1",
        "nameplate_kw": 75.0,
        "rated_rpm": 2970,
        "drive_type": "DOL",
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
        # ---- telemetry fixtures (same physical rig as the A/D suites) --
        # Adaptive persistence cadence (30-min healthy / 1-min anomaly)
        # would drop back-to-back windows of one motor, so the cadence
        # clock is rewound between sends — the A/D suites get the same
        # effect by using one window per motor.
        healthy = harness.three_phase_window((10.0, 10.0, 10.0))
        faulty = harness.three_phase_window((10.0, 10.0, 8.8), thd_frac=0.07)
        inrush = harness.three_phase_window((10.0, 10.0, 10.0), inrush=True)
        await harness.send_window(admin, "MTR-TREND", healthy)
        repo._last_logged["MTR-TREND"] = (
            datetime.now(timezone.utc) - timedelta(minutes=35)
        )
        await harness.send_window(admin, "MTR-TREND", faulty)
        repo._last_logged["MTR-TREND"] = (
            datetime.now(timezone.utc) - timedelta(minutes=35)
        )
        await harness.send_window(admin, "MTR-TREND", inrush)

        # Backdate the first (healthy) row by 30 hours: outside a 24h
        # window, inside a 72h window — the basis of T5.
        trend_rows = [
            r for r in repo.telemetry_rows if r["motor_id"] == "MTR-TREND"
        ]
        check(
            "T0 fixture (3 rows captured incl. the inrush window)",
            len(trend_rows) == 3
            and "INRUSH_SUPPRESSED" in {r["status"] for r in trend_rows},
            f"rows={[r['status'] for r in trend_rows]}",
        )
        oldest = min(trend_rows, key=lambda r: r["recorded_at"])
        oldest["recorded_at"] -= timedelta(hours=30)

        # ---- T1: shape contract --------------------------------------
        r = await client.get("/api/v2/motors/MTR-TREND/history?hours=72")
        m = r.json()
        pts = m.get("points") or []
        w = m.get("window") or {}
        times = [p["recorded_at"] for p in pts]
        check(
            "T1 history shape (200, window echo, ascending, model version)",
            r.status_code == 200
            and m.get("motor_id") == "MTR-TREND"
            and m.get("model_version") == ACTIVE_VERSION
            and m.get("database_connected") is True
            and w.get("hours") == 72
            and w.get("limit") == 500
            and w.get("returned") == len(pts) == 2
            and times == sorted(times),
            f"status={r.status_code}, window={w}, n={len(pts)}",
        )

        # ---- T2: per-point contract ----------------------------------
        max_thd = max((p["thd_percent_max"] for p in pts), default=-1.0)
        min_thd_pt = min(pts, key=lambda p: p["thd_percent_max"]) if pts else {}
        check(
            "T2 point contract (health bounds, worst-phase THD ~7%, "
            "rms ~7.07 A sinusoid, verdict validity)",
            all(0.0 <= p["health_percent"] <= 100.0 for p in pts)
            and all(
                p["status"] in ("HEALTHY", "ANOMALY") for p in pts
            )
            and all(
                p["predicted_class"] is None
                or p["predicted_class"] in CLASS_NAMES
                for p in pts
            )
            and all(
                p["model_confidence"] is None
                or 0.0 <= p["model_confidence"] <= 1.0
                for p in pts
            )
            and 5.0 <= max_thd <= 10.0
            and 6.5 <= min_thd_pt.get("rms_a", -1) <= 7.6
            and all(
                p["crest_factor_max"] > 0
                and p["rotor_sideband_db_max"] != 0
                and p["unbalance_percent"] >= 0
                and 45 <= p["fundamental_hz"] <= 55
                for p in pts
            ),
            f"max_thd={max_thd}, "
            f"clean_rms_a={min_thd_pt.get('rms_a')}",
        )

        # ---- T3: unknown motor -> 404 --------------------------------
        r404 = await client.get("/api/v2/motors/MTR-NOPE/history")
        check(
            "T3 unknown motor 404",
            r404.status_code == 404,
            f"status={r404.status_code}",
        )

        # ---- T4: query validation -> 422 -----------------------------
        r_h0 = await client.get("/api/v2/motors/MTR-TREND/history?hours=0")
        r_hmax = await client.get(
            "/api/v2/motors/MTR-TREND/history?hours=721"
        )
        r_l0 = await client.get("/api/v2/motors/MTR-TREND/history?limit=0")
        check(
            "T4 bounds validation (hours 1-720, limit 1-2000 -> 422)",
            r_h0.status_code == 422
            and r_hmax.status_code == 422
            and r_l0.status_code == 422,
            f"h0={r_h0.status_code}, h721={r_hmax.status_code}, "
            f"l0={r_l0.status_code}",
        )

        # ---- T5: hours window filter ---------------------------------
        r24 = await client.get("/api/v2/motors/MTR-TREND/history?hours=24")
        m24 = r24.json()
        fresh_time = max(p["recorded_at"] for p in pts)
        check(
            "T5 hours filter (backdated row outside 24h, inside 72h)",
            r24.status_code == 200
            and m24["window"]["returned"] == 1
            and m24["points"][0]["recorded_at"] == fresh_time,
            f"returned={m24.get('window', {}).get('returned')}",
        )

        # ---- T6: limit keeps the NEWEST rows -------------------------
        r1 = await client.get(
            "/api/v2/motors/MTR-TREND/history?hours=72&limit=1"
        )
        m1 = r1.json()
        newest_time = max(p["recorded_at"] for p in pts)
        check(
            "T6 limit cap keeps newest row",
            r1.status_code == 200
            and m1["window"]["returned"] == 1
            and m1["points"][0]["recorded_at"] == newest_time,
            f"returned={m1.get('window', {}).get('returned')}",
        )

        # ---- T7: auth matrix ------------------------------------------
        r_no = await noauth.get("/api/v2/motors/MTR-TREND/history")
        r_dev = await device_client.get("/api/v2/motors/MTR-TREND/history")
        check(
            "T7 auth (no key 401, device scope 403, dashboard reads)",
            r_no.status_code == 401 and r_dev.status_code == 403,
            f"noauth={r_no.status_code}, device={r_dev.status_code}",
        )

        # ---- T8: INRUSH never in the series ---------------------------
        statuses = {p["status"] for p in m.get("points", [])} | {
            p["status"] for p in m24.get("points", [])
        }
        check(
            "T8 INRUSH_SUPPRESSED excluded from the series",
            "INRUSH_SUPPRESSED" not in statuses,
            f"statuses={statuses}",
        )

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("RESULT: ALL TRENDS API TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run())
