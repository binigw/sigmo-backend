"""Sigmo V2 — English Repair Protocols & Tools (display edition).

A faithful English translation of the canonical Amharic rulebook
content in ``repair_protocols_am.py`` (SIGMO_RULES.md Section 6.b), so
the dashboard can render the repair protocol in the operator's
selected UI language. The Amharic edition remains the canonical
rulebook artifact; this module only mirrors it — every limit, step,
tool and standard reference is identical (LOTO, 80°C, 110°C, 2,000 h,
3%, ±10%, 5 min / 50 V, 1 MΩ, IEEE 519, IEEE 1159, NEMA MG-1,
ISO 21940).

Contract: ``ENGLISH_PROTOCOLS`` and ``TOOLS_AND_SPARES_EN`` carry
EXACTLY the same keys as their Amharic counterparts (asserted by the
dual-view test suite).
"""
from __future__ import annotations

ENGLISH_PROTOCOLS: dict[str, str] = {
    "HEALTHY": (
        "Normal healthy condition confirmed. No corrective repair is "
        "required.\n"
        "1. Verify the motor continues its normal duty cycle — check "
        "weekly that sound, temperature and running feel remain "
        "regular.\n"
        "2. Verify the cooling-air intake openings are clean of dust "
        "and debris.\n"
        "3. Replenish the lubricating grease per the manufacturer's "
        "instructions every 2,000 operating hours.\n"
        "4. Verify terminal covers are in place and the air passages "
        "are not obstructed.\n"
        "5. Confirm the Sigmo node keeps reading steadily — the "
        "prediction remains at this stage."
    ),
    "BEARING_OUTER_RACE": (
        "Outer race fault detected. Before proceeding with the repair, "
        "follow these steps:\n"
        "1. Safety first: isolate the motor from the power source and "
        "apply LOTO (Lock-Out/Tag-Out); if energy-storage capacitors "
        "are present, ensure they are fully discharged.\n"
        "2. Verification: measure the bearing-housing temperature with "
        "an IR thermometer; above 80°C the repair is immediately "
        "required.\n"
        "3. Remove the coupling guard and verify shaft alignment with "
        "a laser instrument — if the fault is intermittent, do not "
        "continue before correcting the alignment.\n"
        "4. Decouple the motor from the driven load and open the "
        "bearing housing (end shield).\n"
        "5. Remove the damaged bearing with a bearing puller; clean "
        "the housing and shaft with grease-free solvent.\n"
        "6. Replace the bearing ONLY with the part number registered "
        "for this motor (from the asset registry); heat it with an "
        "induction heater to no more than 110°C before fitting.\n"
        "7. Repeat the same procedure on the other side; replacing "
        "both bearings together is recommended.\n"
        "8. Re-assemble with the correct torque; verify the shaft "
        "rotates freely by hand.\n"
        "9. Start the motor and run it unloaded (no-load) for 10 "
        "minutes; track sound, temperature and sideband dB in the "
        "Sigmo node.\n"
        "10. Once the repair is confirmed, record the repair report "
        "in the system (Supabase fault log)."
    ),
    "BEARING_INNER_RACE": (
        "Inner race fault detected — this defect progresses faster "
        "than the outer-race type.\n"
        "1. Safety: isolate power, apply LOTO, ensure stored charge "
        "is discharged.\n"
        "2. Verification: cross-check the 1x modulation growl heard "
        "as the shaft turns, with a stethoscope; measure the "
        "bearing-housing temperature.\n"
        "3. With the load removed from the motor, verify the shaft "
        "turns freely for a short time — if it does, the problem may "
        "lie somewhere other than the bearing.\n"
        "4. Inspect the rotor and stator mating surfaces for rub "
        "marks — confirm the inner-race fault has not contacted the "
        "rotor.\n"
        "5. Remove the damaged bearing with a bearing puller; clean "
        "the shaft and housing thoroughly.\n"
        "6. Replace only with the part number registered in the asset "
        "registry; heat with an induction heater to no more than "
        "110°C and fit.\n"
        "7. Inspect the inner seal; replenish the grease per the "
        "manufacturer's instructions.\n"
        "8. Re-assemble with the correct torque; run a 10-minute "
        "no-load test.\n"
        "9. Verify the Sigmo trace recovers under full load; the "
        "sideband level must drop from the first reading.\n"
        "10. Record the repair report."
    ),
    "BEARING_BALL": (
        "Bearing ball (rolling-element) defect detected.\n"
        "1. Safety: isolate power, apply LOTO.\n"
        "2. Verification: confirm the BSF spin signatures with a "
        "vibration meter if available; a rough gravel-like sound is "
        "usually audible alongside.\n"
        "3. Take a grease sample — metallic flakes confirm the "
        "defect.\n"
        "4. Inspect the bearing housing and rotor surfaces for "
        "scoring; if heavy scoring is found, the complete bearing "
        "assembly must be replaced.\n"
        "5. Replace the bearing with the part number registered in "
        "the asset registry.\n"
        "6. Inspect the housing bore surface — an ovalized bore "
        "requires repairing or replacing the housing.\n"
        "7. Apply the correct grease type and quantity — excess grease "
        "creates problems by itself.\n"
        "8. Re-assemble, run a no-load test, and track the verified "
        "trace in the Sigmo node.\n"
        "9. Record the repair report."
    ),
    "SHAFT_MISALIGNMENT": (
        "Shaft misalignment detected — the 2x signature is elevated.\n"
        "1. Safety: stop the motor, isolate power, apply LOTO.\n"
        "2. Mount a dial indicator or laser alignment instrument on "
        "the coupling.\n"
        "3. Measure angular misalignment — take readings at 0°, 90°, "
        "180° and 270° and compute the difference.\n"
        "4. Measure parallel (offset) misalignment — the shafts must "
        "run true and straight.\n"
        "5. Correct the alignment by adding or removing coupling "
        "shims — stay within the instrument manufacturer's tolerance.\n"
        "6. Check the motor feet for soft foot — every foot must rest "
        "solidly.\n"
        "7. Repeat until the laser alignment instrument confirms the "
        "result.\n"
        "8. Re-assemble with the correct torque; start the motor and "
        "verify the 2x signature has dropped in the Sigmo node.\n"
        "9. Record the repair report."
    ),
    "MECH_UNBALANCE_ECCENTRICITY": (
        "Mechanical unbalance / eccentricity detected — the 1x "
        "component is dominant.\n"
        "1. Safety: isolate power, apply LOTO.\n"
        "2. Inspect the machine for accumulated dust, caked dirt or "
        "an unbalanced attachment (for example a fan caked with "
        "deposits) — this is the most common cause.\n"
        "3. Clean the fan blades; adjusting the weight of added "
        "shims or blade material may be sufficient.\n"
        "4. If a load component is mounted on the shaft (pulley, "
        "rope winding), verify its balance.\n"
        "5. If everything else is excluded and the rotor itself is "
        "unbalanced: send the rotor to a laboratory for field or "
        "dynamic balancing service (per ISO 21940).\n"
        "6. Check the rotor-stator air gap for unevenness with a "
        "feeler gauge — if eccentric, inspect the complete bearing "
        "assembly.\n"
        "7. After the repair, verify the 1x component has dropped in "
        "the Sigmo node.\n"
        "8. Record the repair report."
    ),
    "BROKEN_ROTOR_BAR": (
        "Broken rotor bar detected — the 2sf sideband is elevated. "
        "This fault progresses over time and ends in a total loss of "
        "motor capacity.\n"
        "1. Safety: isolate power, apply LOTO.\n"
        "2. Verification: with the motor on load, confirm the "
        "excessive slip and above-normal heat build-up; rotor-bar "
        "faults reveal themselves under load.\n"
        "3. Perform a single-phase stator test: at 10% voltage, turn "
        "the rotor slowly by hand and measure the current variation "
        "— a variation above 3% confirms the fault.\n"
        "4. Inspect the rotor-stator air gap for bar rash with a "
        "borescope.\n"
        "5. If the damage is severe: the rotor must be sent to a "
        "laboratory for re-barring or replacement — this is "
        "specialized work.\n"
        "6. If it is a minor end-ring joint fault, it can be "
        "corrected with ordinary workshop repair.\n"
        "7. After the repair, verify the 2sf sideband has dropped at "
        "full load in the Sigmo node.\n"
        "8. Record the repair report."
    ),
    "STATOR_WINDING_INTERTURN": (
        "Stator-winding interturn short detected — this is a "
        "high-consequence fault.\n"
        "1. Safety: stop immediately, isolate power, apply LOTO. "
        "Verify the supply is truly disconnected — this fault can "
        "cause a fire.\n"
        "2. Check the air gap for a burnt smell or scorched "
        "insulation.\n"
        "3. Measure the insulation resistance with a megger "
        "(500 V/1000 V) — below 1 MΩ the winding is damaged.\n"
        "4. Measure the phase-to-phase and phase-to-ground "
        "resistances of the stator winding.\n"
        "5. Perform a surge comparison test if available — it "
        "confirms interturn shorts precisely.\n"
        "6. If confirmed: the winding must go to a rewinding "
        "laboratory, or the stator/motor must be replaced — motor "
        "size and service profile decide the repair.\n"
        "7. After the repair, verify the phase-current balance and "
        "the recovered signature in the Sigmo node.\n"
        "8. Record the repair report."
    ),
    "PHASE_CURRENT_UNBALANCE": (
        "Phase-current unbalance detected — above the NEMA MG-1 "
        "limit (2%).\n"
        "1. Safety: isolate power, apply LOTO.\n"
        "2. Measure the voltage on all three phases with a clamp "
        "meter — if the voltage unbalance exceeds 1%, the problem is "
        "the supply; report it to the power utility.\n"
        "3. If the voltage is balanced: inside the MCC panel, inspect "
        "the service contactors, overload-relay terminals and cable "
        "lugs for loose terminals — with a thermal camera or by "
        "hand. This is the most common cause.\n"
        "4. Tighten the loose terminals to the correct torque; "
        "replace damaged contactors.\n"
        "5. Verify the supply-cable cross-section and length are "
        "equal on all three phases.\n"
        "6. If the unbalance persists: an internal winding problem "
        "is likely — follow the STATOR_WINDING protocol.\n"
        "7. After the repair, verify the unbalance % has dropped "
        "(below 2%) in the Sigmo node.\n"
        "8. Record the repair report."
    ),
    "VOLTAGE_SAG_SWELL": (
        "Voltage sag/swell condition detected — outside the IEEE "
        "1159 limits.\n"
        "1. Safety: stand by — this is usually a supply-side issue; "
        "do not start repair work on the motor itself.\n"
        "2. Record the event time, depth and frequency with a power "
        "quality analyzer (if available).\n"
        "3. Correlate the timing with large-motor starting and "
        "capacitor-bank switching in the area.\n"
        "4. Check the input transformer tap changer for balance.\n"
        "5. Where supportable, evaluate a voltage-stability solution "
        "(voltage stabilizer / soft-starter / VFD with ride-through) "
        "for the affected motors.\n"
        "6. If a swell occurred: verify the neutral-grounding system "
        "and the tap setting.\n"
        "7. If the condition originates with the utility, file a "
        "formal report — the recorded data (time, depth) is "
        "required.\n"
        "8. After mitigation, verify the voltage condition is being "
        "recorded again in the Sigmo node."
    ),
    "CAPACITOR_BANK_FAILURE": (
        "Capacitor bank failure detected — a reactive-power supply "
        "and harmonic-system change was identified.\n"
        "1. Safety (critical): capacitors store charge! Wait at "
        "least 5 minutes after de-energizing, then take a "
        "verification discharge with a discharging rod and confirm "
        "below 50 V with a meter.\n"
        "2. Inspect the bank's service fuses and current changes — "
        "blown fuses have isolated part of the bank.\n"
        "3. Measure each capacitor with a capacitance meter — "
        "outside ±10% of its specification it has failed.\n"
        "4. Replace failed capacitors ONLY with the specification "
        "registered in the asset registry (kVAr, voltage, wiring).\n"
        "5. Check the discharge resistors — failed resistors are "
        "dangerous.\n"
        "6. Inspect the switching contactor / reactor system — a "
        "failed inrush reactor creates switching transients.\n"
        "7. After the repair, verify the bank's power-factor "
        "correction is working again.\n"
        "8. Record the repair report."
    ),
    "CAPACITOR_SWITCHING_TRANSIENT": (
        "Capacitor switching transient detected.\n"
        "1. Safety: before starting any work on the capacitor, "
        "verify its charge state — capacitors store energy.\n"
        "2. Establish whether the transient occurs only at switch-on "
        "— if so, the cause is the inrush current.\n"
        "3. Check for an inrush current-limiting reactor or detuning "
        "reactor — install one if absent.\n"
        "4. If the switching contactor is damaged (pitted contact "
        "surfaces), replace it with a capacitor-duty contactor.\n"
        "5. If point-on-wave switching can be implemented it greatly "
        "reduces the transient — this works with a modern "
        "controller.\n"
        "6. After the repair, verify the switching signatures have "
        "disappeared in the Sigmo node.\n"
        "7. Record the repair report."
    ),
    "HARMONICS_HIGH_THD": (
        "High harmonics detected (THD above 5% — IEEE 519).\n"
        "1. Safety: isolate power, apply LOTO.\n"
        "2. Record the harmonic spectrum with a power quality "
        "analyzer — identify which orders (5th, 7th, 11th, 13th) are "
        "dominant.\n"
        "3. Check whether VFD drives, welding machines and rectifier "
        "loads in the area are the harmonic sources.\n"
        "4. Check the VFD loads for a line reactor / DC choke — "
        "install one if absent (typically 3–5% impedance).\n"
        "5. Evaluate installing a harmonic filter or a detuned "
        "capacitor bank.\n"
        "6. If metering is affected, check for CT saturation errors "
        "(a false reading can mimic real harmonics).\n"
        "7. After the repair, verify the THD has dropped below 5% "
        "in the Sigmo node.\n"
        "8. Record the repair report."
    ),
}

# English edition of the baseline tools & spares lists (same keys as
# TOOLS_AND_SPARES in repair_protocols_am.py; asset-specific part
# numbers are still appended from the motors registry at call time).
TOOLS_AND_SPARES_EN: dict[str, list[str]] = {
    "HEALTHY": [
        "Standard grease gun and the manufacturer-specified grease",
        "IR thermometer",
        "Cleaning tools (brush, cloth)",
    ],
    "BEARING_OUTER_RACE": [
        "New bearing — exact part number registered in the asset registry",
        "Bearing puller and induction heater",
        "Dial indicator / laser alignment kit",
        "Grease and cleaning solvent, torque wrench",
        "LOTO kit (lock, tag, power verifier)",
    ],
    "BEARING_INNER_RACE": [
        "New bearing — part number registered in the asset registry",
        "Bearing puller and induction heater (110°C maximum only)",
        "Feeler gauge (air-gap gauge), torque wrench",
        "Spare seal and manufacturer-specified grease",
        "LOTO kit",
    ],
    "BEARING_BALL": [
        "Complete bearing assembly — part number registered in the "
        "asset registry",
        "Bearing puller, induction heater, torque wrench",
        "Grease sampling kit",
        "Bearing-housing inspection — replacement housing if heavy "
        "scoring is found",
        "LOTO kit",
    ],
    "SHAFT_MISALIGNMENT": [
        "Laser alignment instrument (kit) or dial-indicator kit",
        "Shim set (stainless)",
        "Torque wrench and ring-spanner set",
        "Foundation level gauge",
        "LOTO kit",
    ],
    "MECH_UNBALANCE_ECCENTRICITY": [
        "Fan cleaning tools (brush, scraper)",
        "Feeler gauge (air-gap gauge)",
        "Field balancing service (ISO 21940) — when a rotor fault is "
        "confirmed",
        "Registered shim / balancing-weight adjustment set",
        "LOTO kit",
    ],
    "BROKEN_ROTOR_BAR": [
        "Rotor laboratory service (re-barring / dynamic balancing)",
        "Borescope (internal inspection)",
        "Single-phase test measurement set (low-voltage variac + "
        "ammeter)",
        "Serviceable spare rotor (if stocked)",
        "LOTO kit",
    ],
    "STATOR_WINDING_INTERTURN": [
        "Megger (500 V / 1000 V insulation tester)",
        "Surge comparison tester (if available)",
        "Winding laboratory service (rewinding) or replacement stator",
        "CO₂ fire extinguisher (on standby during the repair)",
        "LOTO kit",
    ],
    "PHASE_CURRENT_UNBALANCE": [
        "Clamp meter (three-phase current measurement)",
        "Thermal camera or IR thermometer (loose-terminal detection)",
        "Contactor and overload-relay spare parts",
        "Torque wrench and correctly sized terminal spanners",
        "LOTO kit",
    ],
    "VOLTAGE_SAG_SWELL": [
        "Power quality analyzer (voltage event recorder)",
        "Transformer tap-adjustment tool",
        "Mitigation options: voltage stabilizer / soft-starter / VFD "
        "evaluation",
        "Event-time evidence log (for the utility report)",
    ],
    "CAPACITOR_BANK_FAILURE": [
        "New capacitors — specification registered in the asset "
        "registry (kVAr / V / wiring)",
        "Capacitance meter",
        "Discharging rod and voltmeter (verify below 50 V)",
        "Discharge-resistor and fuse spares",
        "Capacitor-duty contactor",
        "LOTO kit and insulating gloves",
    ],
    "CAPACITOR_SWITCHING_TRANSIENT": [
        "Inrush / detuning reactor (if absent)",
        "Capacitor-duty contactor (replacement)",
        "Discharging rod and voltmeter",
        "Point-on-wave switching evaluation (with a modern controller)",
        "LOTO kit",
    ],
    "HARMONICS_HIGH_THD": [
        "Power quality analyzer (harmonic-spectrum measurement)",
        "Line reactor / DC choke (for VFD loads — 3–5% impedance)",
        "Harmonic filter or detuned capacitor bank evaluation",
        "CT saturation check tooling",
    ],
}
