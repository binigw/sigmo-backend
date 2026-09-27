"""Sigmo V2 — Phase 5 integration tests: Dual-View UX API
(SIGMO_RULES.md Section 6, STRICT EXPLICIT DATA SCHEMAS).

Drives the real ASGI app over HTTP with the same physically-modeled
waveforms and capture-repository harness as the ingestion suite. The
two payloads are verified field-by-field against the EXACT Section 6.a
/ 6.b contracts — no generic fields, no placeholder estimates: unset
business rates must surface as explicit statuses, never invented
numbers.

Test matrix:
  D1:  stage engine unit — all four stages from z / RUL evidence
  D2:  health-index math — z=0 -> 100%, z=5 -> 0%, plant mean
  D3:  financial formulas — exposure & energy penalty arithmetic
  D4:  admin endpoints — asset registration + plant config (HTTP)
  D5:  executive payload, rates UNCONFIGURED — explicit
       rate_not_configured statuses (zero-fake directive)
  D6:  executive payload, rates configured — honest PARTIAL computation
       naming exactly which at-risk motors lack rates
  D6b: executive payload, all assets registered — fully computed figures
  D7:  executive RUL window — Section 14 protocol status, binding motor
  D8:  executive directive — deterministic strategic sentence
  D9:  technician payload (faulty motor) — all Section 6.b fields,
       taxonomy code format, Amharic protocol, spectral evidence
  D10: technician unregistered motor — honest "not registered" flags
  D11: technician registered motor — bearing part number flows through
  D12: technician 404 for unknown motor
  D13: inrush-only motor -> 404 (no steady-state telemetry yet)
  D14: strict view isolation — no field bleeding between payloads
  D15: CORS preflight on the view endpoints (external Replit frontend)
  D16: authentication — no key 401; dashboard key CANNOT post telemetry
  D17: authentication — admin key may drive the full surface
  D18: authentication — fail-closed 503 when no keys are configured
"""
from __future__ import annotations

import asyncio
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/user/sigmo_v2")
sys.path.insert(0, "/home/user/sigmo_v2/tests")

import httpx

import test_ingestion_api as harness
from backend import main as sigmo_main
from backend.api import auth
from backend.analysis.staging import (
    classify_stage,
    energy_penalty_monthly_etb,
    financial_risk_exposure_etb,
    motor_health_percent,
    plant_health_index,
)
from backend.config import VIEW

check = harness.check
failures = harness.failures  # check() appends into this list
ACTIVE_VERSION = harness.ACTIVE_VERSION

# Stage 1-4 regex for the taxonomy code contract
TAXONOMY_RE = re.compile(r"^Stage [1-4] .+ \(.+\)$")
AMHARIC_RE = re.compile(r"[\u1200-\u137F]")  # Ethiopic block


def _step_count(protocol: str) -> int:
    """Number of numbered steps in a protocol body (1. … 10. …)."""
    return sum(
        1 for line in protocol.splitlines()
        if line[:2].rstrip(".").isdigit()
    )


async def run() -> None:
    repo = harness.CaptureRepository()
    sigmo_main.service._repo = repo  # service path (ingest + views)
    sigmo_main.repository = repo     # admin path (assets + plant config)
    # Section-7 keys; the client authenticates with the ADMIN key, which
    # may drive the whole surface (telemetry + reads + admin). Scope
    # separation is asserted by the dedicated auth checks below and in
    # the A-suite (A12-A16).
    auth.load_keys(
        f"{harness.TEST_DEVICE_KEY}:device,"
        f"{harness.TEST_DASHBOARD_KEY}:dashboard,"
        f"{harness.TEST_ADMIN_KEY}:admin"
    )
    # Commissioning precedes telemetry (Section 11 Cold Start) — the
    # production FK on telemetry_features.motor_id demands it. The rig
    # motors are commissioned with MINIMAL assets (no rates/bearing):
    # MTR-FAULTY is fully registered in D4, and MTR-HEALTHY's entry is
    # removed below (registry-correction scenario) so the honest
    # unregistered-motor paths stay covered without violating the
    # commissioning order the API now enforces.
    for mid in ("MTR-HEALTHY", "MTR-FAULTY", "MTR-STARTUP"):
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
        ) as client,
        httpx.AsyncClient(
            transport=transport, base_url="http://sigmo-test", timeout=120.0
        ) as noauth,
    ):

        # ---- telemetry fixtures (same physical rig as the A-suite) ----
        healthy = harness.three_phase_window((10.0, 10.0, 10.0))
        faulty = harness.three_phase_window((10.0, 10.0, 8.8), thd_frac=0.07)
        inrush = harness.three_phase_window((10.0, 10.0, 10.0), inrush=True)
        await harness.send_window(client, "MTR-HEALTHY", healthy)
        await harness.send_window(client, "MTR-FAULTY", faulty)
        await harness.send_window(client, "MTR-STARTUP", inrush)

        # ---- D1: stage engine unit checks -----------------------------
        st1 = classify_stage(
            zscore_max=0.8, gating_status="HEALTHY",
            predicted_class="HEALTHY",
            rul_days_numeric=None, rul_status="CALCULATING_BUILDING_TREND",
        )
        st2 = classify_stage(
            zscore_max=1.2, gating_status="HEALTHY",
            predicted_class="BEARING_OUTER_RACE",
            rul_days_numeric=None, rul_status=None,
        )
        st2b = classify_stage(
            zscore_max=1.2, gating_status="ANOMALY",
            predicted_class="HEALTHY",
            rul_days_numeric=None, rul_status=None,
        )
        st3 = classify_stage(
            zscore_max=4.2, gating_status="ANOMALY",
            predicted_class="BROKEN_ROTOR_BAR",
            rul_days_numeric=None, rul_status=None,
        )
        st4 = classify_stage(
            zscore_max=1.0, gating_status="HEALTHY",
            predicted_class="HEALTHY",
            rul_days_numeric=0,
            rul_status="CRITICAL_THRESHOLD_REACHED",
        )
        check(
            "D1 stage engine",
            st1.stage == 1 and st1.max_operating_hours is None
            and st2.stage == 2 and st2b.stage == 2
            and st3.stage == 3 and st4.stage == 4
            and st4.max_operating_hours == 0,
            f"stages: {st1.stage}/{st2.stage}/{st2b.stage}/"
            f"{st3.stage}/{st4.stage}, hours4={st4.max_operating_hours}",
        )

        # ---- D2: health index math ------------------------------------
        h0, h5, hmid, plant = (
            motor_health_percent(0.0), motor_health_percent(5.0),
            motor_health_percent(2.5), plant_health_index([0.0, 5.0]),
        )
        check(
            "D2 health index math",
            h0 == 100.0 and h5 == 0.0 and hmid == 50.0
            and plant == 50.0 and plant_health_index([]) is None,
            f"z0={h0}, z5={h5}, z2.5={hmid}, mean={plant}",
        )

        # ---- D3: financial formulas -----------------------------------
        exposure = financial_risk_exposure_etb(
            [(1, 9999.0), (2, 4500.0), (3, 4500.0), (4, 1000.0)]
        )
        energy = energy_penalty_monthly_etb(
            nameplate_kw=90.0, thd_percent_max=6.85,
            unbalance_percent=8.33, tariff_etb_per_kwh=10.0,
        )
        check(
            "D3 financial formulas",
            exposure == 8 * 4500 + 24 * 4500 + 48 * 1000
            and energy > 0,
            f"exposure={exposure:,.0f} ETB, energy={energy:,.0f} ETB/mo",
        )

        # ---- D5: executive BEFORE rates configured (zero-fake gate) ---
        r = await client.get("/api/v2/views/executive")
        e0 = r.json()
        risk0 = e0["financial_risk_exposure_etb"]
        energy0 = e0["energy_efficiency_loss_penalty_etb"]
        check(
            "D5 executive unconfigured honesty",
            r.status_code == 200
            and e0["model_version"] == ACTIVE_VERSION
            and e0["motors_monitored"] == 2  # HEALTHY + FAULTY (not STARTUP)
            and isinstance(e0["plant_health_index"], float)
            and e0["plant_health_index"] > 0
            and risk0["value_etb"] is None
            and risk0["status"] == "rate_not_configured"
            and "MTR-FAULTY" in risk0["unconfigured_items"]
            and energy0["value_etb"] is None
            and energy0["status"] == "rate_not_configured",
            f"phi={e0['plant_health_index']}%, risk={risk0['status']}, "
            f"energy={energy0['status']}, motors={e0['motors_monitored']}",
        )

        # Registry correction between D5 and D4: MTR-HEALTHY's asset
        # entry is withdrawn (motor identity re-commissioned / moved to
        # another panel) while its telemetry history remains — the only
        # way an unregistered motor with telemetry can legitimately
        # exist now that the API enforces commissioning-first. This
        # preserves the honest-unregistered scenarios of D6/D10/D6b.
        repo.assets.pop("MTR-HEALTHY", None)

        # ---- D4: admin registration + plant config --------------------
        asset_payload = {
            "mcc_panel_id": "MCC-04", "cabinet_number": "C-12",
            "line_zone_id": "LINE-B/ZONE-2", "nameplate_kw": 90.0,
            "rated_rpm": 2975, "drive_type": "DOL",
            "asset_tag": "ASM-2214", "bearing_part_number": "6316-C3",
            "spare_capacitor_spec": "50 kVAr 415V delta",
            "downtime_cost_etb_per_hour": 4500.0,
        }
        r = await client.put(
            "/api/v2/admin/motor-assets/MTR-FAULTY", json=asset_payload
        )
        # MTR-HEALTHY stays unregistered (popped above) for D6/D10.
        # The listing shows MTR-FAULTY (full asset) plus the MTR-STARTUP
        # commissioning stub — sorted, so MTR-FAULTY is first.
        r3 = await client.put(
            "/api/v2/admin/plant-config/energy_tariff_etb_per_kwh",
            json={"value": 10.0, "unit": "ETB/kWh",
                  "description": "EEU industrial tariff entered at "
                                 "commissioning"},
        )
        listing = (await client.get("/api/v2/admin/motor-assets")).json()
        cfg = (await client.get("/api/v2/admin/plant-config")).json()
        check(
            "D4 admin registration",
            r.status_code == 200 and r3.status_code == 200
            and len(listing) == 2  # MTR-FAULTY (full) + MTR-STARTUP (stub)
            and listing[0]["motor_id"] == "MTR-FAULTY"
            and listing[0]["bearing_part_number"] == "6316-C3"
            and cfg["energy_tariff_etb_per_kwh"]["value"] == 10.0,
            f"assets={len(listing)} ({[a['motor_id'] for a in listing]}), "
            f"cfg_keys={sorted(cfg)}",
        )

        # ---- D6: executive WITH rates configured ----------------------
        r = await client.get("/api/v2/views/executive")
        e1 = r.json()
        risk1 = e1["financial_risk_exposure_etb"]
        energy1 = e1["energy_efficiency_loss_penalty_etb"]
        summary = e1["critical_asset_risk_summary"]
        # MTR-HEALTHY is Stage 2 on this synthetic rig but unregistered
        # (no nameplate / no cost) -> both figures must be honest
        # PARTIAL computations naming exactly what is missing.
        check(
            "D6 executive partial-rate honesty",
            risk1["value_etb"] == 36000.0  # MTR-FAULTY: 8h x 4500 ETB
            and risk1["status"] == "partial_rates_configured"
            and risk1["unconfigured_items"] == ["MTR-HEALTHY"]
            and energy1["value_etb"] is not None
            and energy1["value_etb"] > 1000.0
            and energy1["status"] == "partial_rates_configured"
            and energy1["unconfigured_items"] == ["MTR-HEALTHY"]
            and summary["total_monitored"] == 2
            and summary["healthy"] + summary["stage_2_moderate_risk"]
            + summary["stage_3_4_critical_failure_risk"] == 2
            and summary["stage_2_moderate_risk"]
            + summary["stage_3_4_critical_failure_risk"] >= 1,
            f"risk={risk1['value_etb']:,.0f} ETB, "
            f"energy={energy1['value_etb']:,.0f} ETB/mo, "
            f"missing={risk1['unconfigured_items']}, "
            f"summary=({summary['healthy']}/"
            f"{summary['stage_2_moderate_risk']}/"
            f"{summary['stage_3_4_critical_failure_risk']})",
        )

        # ---- D7/D8: RUL window + directive ----------------------------
        rul = e1["rul_days_estimate"]
        directive = e1["executive_decision_directive"]
        check(
            "D7 executive RUL window",
            rul["status"] in (
                "CALCULATING_INSUFFICIENT_DATA",
                "CALCULATING_BUILDING_TREND",
            )
            and isinstance(rul["estimate"], str)
            and rul["binding_motor_id"] in ("MTR-FAULTY", "MTR-HEALTHY")
            and rul["lower"] is None and rul["upper"] is None,
            f"estimate={rul['estimate']!r}, binding={rul['binding_motor_id']}",
        )
        check(
            "D8 executive directive",
            isinstance(directive, str) and len(directive) > 20
            and ("Plan" in directive or "Schedule" in directive
                 or "CRITICAL" in directive
                 or "no intervention" in directive),
            directive[:90],
        )

        # ---- D9: technician payload (faulty, registered) --------------
        r = await client.get("/api/v2/views/technician/MTR-FAULTY")
        t = r.json()
        loc, spec = t["mcc_panel_location"], t["motor_specifications"]
        ev, stage = t["spectral_evidence_data"], t["fault_urgency_stage"]
        tools = t["required_tools_and_spares"]
        stage_hours = VIEW.max_operating_hours[stage["stage"]]
        check(
            "D9 technician payload",
            r.status_code == 200
            and t["motor_id"] == "MTR-FAULTY"
            and loc["registered"] and loc["mcc_panel_id"] == "MCC-04"
            and loc["cabinet_number"] == "C-12"
            and loc["line_zone"] == "LINE-B/ZONE-2"
            and spec["registered"] and spec["asset_id"] == "ASM-2214"
            and spec["nameplate_kw"] == 90.0 and spec["rated_rpm"] == 2975
            and spec["drive_type"] == "DOL"
            and TAXONOMY_RE.match(t["exact_fault_taxonomy_code"])
            and "BPFO" in t["exact_fault_taxonomy_code"]
            and t["dominant_fault"] == "BEARING_OUTER_RACE"
            and AMHARIC_RE.search(t["amharic_repair_protocol"])
            and len(t["amharic_repair_protocol"]) > 200
            and "repair_protocol_en" in t
            and not AMHARIC_RE.search(t["repair_protocol_en"])
            and len(t["repair_protocol_en"]) > 200
            and "LOTO" in t["repair_protocol_en"]
            and _step_count(t["repair_protocol_en"])
            == _step_count(t["amharic_repair_protocol"])
            and len(tools["items"]) >= 3
            and len(tools["items_en"]) == len(tools["items"])
            and not any(AMHARIC_RE.search(i) for i in tools["items_en"])
            and tools["asset_bearing_part_number"] == "6316-C3"
            and len(ev["fft_peak_hz"]) == 3
            and len(ev["sideband_50hz_db"]) == 3
            and ev["thd_percent_max"] > 5.0
            and ev["zscore_baseline_deviation"] >= 0
            and stage["stage"] in (2, 3, 4)
            and stage["max_operating_hours_before_trip"] == stage_hours
            and len(stage["basis"]) > 10,
            f"code={t['exact_fault_taxonomy_code'][:60]}, "
            f"stage={stage['stage']} ({stage['label'][:30]}), "
            f"hours={stage['max_operating_hours_before_trip']}, "
            f"thd={ev['thd_percent_max']:.2f}%",
        )

        # ---- D10: technician unregistered motor -----------------------
        r = await client.get("/api/v2/views/technician/MTR-HEALTHY")
        t = r.json()
        loc, spec, tools = (
            t["mcc_panel_location"], t["motor_specifications"],
            t["required_tools_and_spares"],
        )
        check(
            "D10 technician unregistered honesty",
            r.status_code == 200
            and not loc["registered"] and loc["mcc_panel_id"] is None
            and not spec["registered"] and spec["nameplate_kw"] is None
            and tools["asset_bearing_part_number"] is None
            and "not yet registered" in tools["registry_note"]
            and len(t["repair_protocol_en"]) > 100
            and not AMHARIC_RE.search(t["repair_protocol_en"]),
            f"registered={loc['registered']}, note={tools['registry_note'][:60]}",
        )

        # ---- D11: bearing part number flows (already in D9; explicit) --
        check(
            "D11 spares from registry",
            t["required_tools_and_spares"]["asset_bearing_part_number"] is None
            and (
                await client.get("/api/v2/views/technician/MTR-FAULTY")
            ).json()["required_tools_and_spares"]
            ["asset_bearing_part_number"] == "6316-C3",
            "unregistered motor: null; registered motor: 6316-C3",
        )

        # ---- D6b: full registration -> fully computed figures ---------
        await client.put(
            "/api/v2/admin/motor-assets/MTR-HEALTHY",
            json=dict(asset_payload, nameplate_kw=55.0,
                      asset_tag="ASM-2201", rated_rpm=1480,
                      bearing_part_number="6312-C3",
                      downtime_cost_etb_per_hour=2000.0),
        )
        e2 = (await client.get("/api/v2/views/executive")).json()
        risk2 = e2["financial_risk_exposure_etb"]
        energy2 = e2["energy_efficiency_loss_penalty_etb"]
        check(
            "D6b executive fully computed",
            risk2["status"] == "computed"
            and risk2["value_etb"] == 36000.0 + 8 * 2000.0
            and energy2["status"] == "computed"
            and energy2["value_etb"] > e1["energy_efficiency_loss_penalty_etb"]["value_etb"],
            f"risk={risk2['value_etb']:,.0f} ETB, "
            f"energy={energy2['value_etb']:,.0f} ETB/mo",
        )

        # ---- D12/D13: 404 paths ---------------------------------------
        r_unknown = await client.get("/api/v2/views/technician/MTR-NONE")
        r_inrush = await client.get("/api/v2/views/technician/MTR-STARTUP")
        check(
            "D12/D13 technician 404 paths",
            r_unknown.status_code == 404
            and "No telemetry recorded" in r_unknown.json()["detail"]
            and r_inrush.status_code == 404,
            f"unknown={r_unknown.status_code}, "
            f"inrush-only={r_inrush.status_code}",
        )

        # ---- D14: strict view isolation --------------------------------
        exec_keys = set(e1.keys())
        tech_keys = set(t.keys())
        check(
            "D14 view isolation",
            "amharic_repair_protocol" not in exec_keys
            and "mcc_panel_location" not in exec_keys
            and "plant_health_index" not in tech_keys
            and "financial_risk_exposure_etb" not in tech_keys,
            f"exec_fields={len(exec_keys)}, tech_fields={len(tech_keys)}, "
            "zero bleed",
        )

        # ---- D15: CORS preflight on the view endpoints -----------------
        pre = await client.options(
            "/api/v2/views/executive",
            headers={
                "Origin": "https://sigmo-dashboard.replit.app",
                "Access-Control-Request-Method": "GET",
            },
        )
        check(
            "D15 CORS on views",
            pre.status_code == 200
            and pre.headers.get("access-control-allow-origin") is not None
            and "GET" in (pre.headers.get("access-control-allow-methods") or ""),
            f"preflight={pre.status_code}, "
            f"methods={pre.headers.get('access-control-allow-methods')}",
        )

        # ---- D16: auth — no key 401; dashboard cannot post telemetry --
        r_no = await noauth.get("/api/v2/views/executive")
        r_dash = await client.post(
            "/api/v2/telemetry", json={},
            headers={"X-API-Key": harness.TEST_DASHBOARD_KEY},
        )
        check(
            "D16 auth scopes on views/telemetry",
            r_no.status_code == 401 and r_dash.status_code == 403,
            f"no-key={r_no.status_code}, dashboard-on-telemetry="
            f"{r_dash.status_code}",
        )

        # ---- D17: admin key drives the full surface --------------------
        r_tel = await client.post("/api/v2/telemetry", json={})  # 422 from
        # contract validation (NOT 401/403) proves auth passed
        r_view = await client.get("/api/v2/views/executive")
        r_adm = await client.get("/api/v2/admin/motor-assets")
        check(
            "D17 admin full surface",
            r_tel.status_code == 422 and r_view.status_code == 200
            and r_adm.status_code == 200,
            f"telemetry={r_tel.status_code} (422=contract, auth OK), "
            f"view={r_view.status_code}, admin={r_adm.status_code}",
        )

        # ---- D18: fail-closed without keys -----------------------------
        auth.reset_keys()
        r = await noauth.get("/api/v2/views/executive")
        check(
            "D18 fail-closed without keys",
            r.status_code == 503 and "not configured" in r.json()["detail"],
            f"status={r.status_code}",
        )
        auth.load_keys(
            f"{harness.TEST_DEVICE_KEY}:device,"
            f"{harness.TEST_DASHBOARD_KEY}:dashboard,"
            f"{harness.TEST_ADMIN_KEY}:admin"
        )

        # ---- D19: restart survival — views from persisted rows only --
        # The production state after every service restart / free-tier
        # wake has NO live verdicts in RAM; both views must serve the
        # Supabase rows with the complete evidence set. Regression test
        # for the KeyError('envelope_peak_hz_a') that broke both views
        # after the first free-tier sleep.
        sigmo_main.service._live_verdicts.clear()
        re_exec = (await client.get("/api/v2/views/executive")).json()
        re_tech = (await client.get(
            "/api/v2/views/technician/MTR-FAULTY")).json()
        ev19 = re_tech.get("spectral_evidence_data") or {}
        check(
            "D19 restart survival (views from Supabase)",
            re_exec.get("motors_monitored") == 2
            and re_exec.get("model_version") == ACTIVE_VERSION
            and re_tech.get("motor_id") == "MTR-FAULTY"
            and re_tech.get("dominant_fault") == "BEARING_OUTER_RACE"
            and len(ev19.get("fft_peak_hz") or []) == 3
            and len(ev19.get("sideband_50hz_db") or []) == 3
            and isinstance(ev19.get("crest_factor_max"), float)
            and ev19.get("zscore_baseline_deviation") is not None,
            f"exec_motors={re_exec.get('motors_monitored')}, "
            f"fault={re_tech.get('dominant_fault')}, "
            f"fft={len(ev19.get('fft_peak_hz') or [])}, "
            f"crest={ev19.get('crest_factor_max')}",
        )

    # ---- D20: bilingual protocol library parity ------------------
    from backend.analysis.repair_protocols_am import (
        AMHARIC_PROTOCOLS as _AM,
        TOOLS_AND_SPARES as _TS_AM,
    )
    from backend.analysis.repair_protocols_en import (
        ENGLISH_PROTOCOLS as _EN,
        TOOLS_AND_SPARES_EN as _TS_EN,
    )
    check(
        "D20 bilingual protocol library parity",
        set(_EN) == set(_AM) == set(_TS_AM) == set(_TS_EN)
        and len(_EN) == 13
        and all(not AMHARIC_RE.search(v) for v in _EN.values())
        and all(AMHARIC_RE.search(v) for v in _AM.values())
        and all(len(v) > 100 for v in _EN.values())
        and all(_step_count(_EN[k]) == _step_count(_AM[k]) for k in _AM)
        and all(len(_TS_EN[k]) == len(_TS_AM[k]) for k in _TS_AM)
        and all(len(v) >= 3 for v in _TS_EN.values()),
        f"classes={len(_EN)}, en_min_len={min(map(len, _EN.values()))}, "
        f"steps_parity={all(_step_count(_EN[k]) == _step_count(_AM[k]) for k in _AM)}",
    )

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("RESULT: ALL DUAL-VIEW API TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(run())
