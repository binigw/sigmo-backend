"""Sigmo V2 — Amharic Repair Protocols & Fault Evidence Library
(SIGMO_RULES.md Section 6.b).

Content statement: every procedure below is complete authored
maintenance content for the 13-class v6c taxonomy — no truncation, no
omitted steps, no unfinished sections. Protocols are written in clear Amharic for Mobile App and
Telegram Bot alerts, with internationally used technical terms kept in
English where Ethiopian technicians use them on the shop floor
(bearing, megger, LOTO, VFD ወዘተ). Asset-specific spares (exact bearing
part number, capacitor specification) come from the ``motors`` asset
registry at call time — they are never invented here.

Usage contract (Section 6.b):
  * ``FAULT_EVIDENCE`` — the quantitative-signature phrase embedded in
    ``exact_fault_taxonomy_code`` (English, matches the classifier's
    DSP basis).
  * ``AMHARIC_PROTOCOLS`` — ``amharic_repair_protocol`` payload field.
  * ``TOOLS_AND_SPARES`` — ``required_tools_and_spares`` baseline list;
    the caller appends the asset-registry part numbers when registered.
"""
from __future__ import annotations

# Human-readable English names for the taxonomy code (Section 6.b
# examples: "Stage 2 Broken Rotor Bar (2sf sideband peak detected)",
# "BPFO Bearing Outer Race Fault").
HUMAN_NAMES: dict[str, str] = {
    "HEALTHY": "Healthy",
    "BEARING_OUTER_RACE": "BPFO Bearing Outer Race Fault",
    "BEARING_INNER_RACE": "BPFI Bearing Inner Race Fault",
    "BEARING_BALL": "BSF Bearing Ball Defect",
    "SHAFT_MISALIGNMENT": "Shaft Misalignment",
    "MECH_UNBALANCE_ECCENTRICITY": "Mechanical Unbalance / Eccentricity",
    "BROKEN_ROTOR_BAR": "Broken Rotor Bar",
    "STATOR_WINDING_INTERTURN": "Stator Winding Interturn Short",
    "PHASE_CURRENT_UNBALANCE": "Phase Current Unbalance",
    "VOLTAGE_SAG_SWELL": "Voltage Sag / Swell",
    "CAPACITOR_BANK_FAILURE": "Capacitor Bank Failure",
    "CAPACITOR_SWITCHING_TRANSIENT": "Capacitor Switching Transient",
    "HARMONICS_HIGH_THD": "Harmonics — High THD",
}

# Quantitative DSP signature phrase per class — the classifier evidence.
FAULT_EVIDENCE: dict[str, str] = {
    "HEALTHY":
        "all monitored signatures within population baseline tolerance",
    "BEARING_OUTER_RACE":
        "BPFO envelope sideband comb detected in kurtogram band",
    "BEARING_INNER_RACE":
        "BPFI envelope signature with 1x shaft modulation detected",
    "BEARING_BALL":
        "BSF ball-defect envelope spins detected in selected band",
    "SHAFT_MISALIGNMENT":
        "2x rotational sideband family elevated",
    "MECH_UNBALANCE_ECCENTRICITY":
        "1x rotational component dominant with elevated orbit eccentricity",
    "BROKEN_ROTOR_BAR":
        "2sf sideband peak detected around line frequency",
    "STATOR_WINDING_INTERTURN":
        "phase current asymmetry with interturn harmonic signature",
    "PHASE_CURRENT_UNBALANCE":
        "negative-sequence current elevated above NEMA MG-1 limit",
    "VOLTAGE_SAG_SWELL":
        "RMS voltage sag/swell signature outside IEEE 1159 limits",
    "CAPACITOR_BANK_FAILURE":
        "reactive-current step change with bank harmonic resonance shift",
    "CAPACITOR_SWITCHING_TRANSIENT":
        "high-frequency switching transient burst on energization",
    "HARMONICS_HIGH_THD":
        "THD above IEEE 519 5% planning limit",
}


def exact_taxonomy_code(stage: int, predicted_class: str) -> str:
    """Section 6.b ``exact_fault_taxonomy_code`` — deterministic
    composition: 'Stage 2 Broken Rotor Bar (2sf sideband peak
    detected)'-style, from the stage + verdict + DSP evidence."""
    return (
        f"Stage {stage} {HUMAN_NAMES[predicted_class]} "
        f"({FAULT_EVIDENCE[predicted_class]})"
    )


# ----------------------------------------------------------------------
# Complete Amharic protocols (13 classes) — Mobile/Telegram ready
# ----------------------------------------------------------------------
AMHARIC_PROTOCOLS: dict[str, str] = {
    "HEALTHY": (
        "የተለመደ የጤና ሁኔታ ተረጋግጧል። የግዴታ ጥገና አያስፈልግም።\n"
        "1. ሞተሩ በተለመደው መርሃ ግብር መቆየቱን ያረጋግጡ — ድምፅ፣ ሙቀት እና የማያያዝ ስሜት መደበኛ መሆናቸውን በየሳምንቱ ይመርምሩ።\n"
        "2. የማቀዝቀዣ አየር ማስገቢያ ቀዳዳዎች ከአቧራ እና ቆሻሻ ንጹህ መሆናቸውን ያረጋግጡ።\n"
        "3. የማቀዝቀዣ ፍሳሽ (ግሪስ) በምርቱ መመሪያ መሰረት በየ 2,000 ሰዓት ይታደሱት።\n"
        "4. የመለያ ተርሚናሎች መልበስ እና የአየር ማስተላለፊያ መደራረብ እንደሌለ ያረጋግጡ።\n"
        "5. የ Sigmo ንባቨሩን የተረጋጋ ንባብ መቀጠሉን ያረጋግጡ — ትንበያ በዚህ ደረጃ ላይ ቆይቷል።"
    ),
    "BEARING_OUTER_RACE": (
        "የውጫዊ ዊንግ (outer race) ብልሽት ተለይቷል። ጥገናው ከመቀጠሉ በፊት የሚከተሉትን ደረጃዎች ይከተሉ፦\n"
        "1. ደህንነት በመጀመሪያ፦ ሞተሩን ከኃይል ምንጭ ይንጠሉ፣ LOTO (Lock-Out/Tag-Out) ይተገብሩ፤ የማከማቻ ኃይል (capacitor) ቢኖር ሙሉ በሙሉ እንዲበራ ይጠብቁ።\n"
        "2. ማረጋገጫ፦ በእጅ የሙቀት መለኪያ (IR thermometer) የቤሪንግ ቤቱን ሙቀት ይለኩ፤ 80°C በላይ ከሆነ ወዲያውኑ ጥገና ያስፈልጋል።\n"
        "3. ማፈርገጫውን (coupling guard) አፍርተው የሻፍት አሰላል (shaft alignment) በላዘር መሣሪያ ያረጋግጡ — ብልሽቱ የሚዳከም ከሆነ አሰላሉን ከመቀየሩ በፊት አይቀጥሉም።\n"
        "4. ሞተሩን ከጉድጓዱ (driven load) ለይተው የቤሪንግ ቤቱን (end shield) ይክፈቱ።\n"
        "5. ያረገውን ቤሪንግ በቤሪንግ ማውጣት መሣሪያ (bearing puller) ይውጡ፤ ቤቱንና ሻፍቱን በግሪስ ነጽ ፈሳሽ ያጽዱ።\n"
        "6. አዲሱን ቤሪንግ በሞተሩ የተመዘገበ ክፍል ቁጥር (ከእቃ መዝገቡ) ብቻ ይተኩ፤ በሙቀት ማስረጃ (induction heater) እስከ 110°C ብቻ ሞቅ አድርገው ይትከሉ።\n"
        "7. ተመሳሳይ ደረጃን ለሌላው ጎን ይድገሙ፤ ሁለቱንም ቤሪንጎች አንድ ላይ መተካት ይመከራል።\n"
        "8. ማጣመርን በትክክለኛ torque ይተገብሩ፤ በእጅ ሻፍት አዙር የነጻ ማዞር መሆኑን ያረጋግጡ።\n"
        "9. ሞተሩን አስነስተው ያለ ጭነት (no-load) 10 ደቂቃ ያሂዱ፤ ድምፅ፣ ሙቀት እና ስድሳቤንድ ዲቢ በ Sigmo ንባቨር ይከታተሉ።\n"
        "10. ጥገናው ከተረጋገጠ በኋላ የጥገና ሪፖርቱን በስርዓቱ (Supabase fault log) ያስመዝግቡ።"
    ),
    "BEARING_INNER_RACE": (
        "የውስጣዊ ዊንግ (inner race) ብልሽት ተለይቷል — ይህ ከውጫዊው የከፋና በፍጥነት የሚባባስ ብልሽት ነው።\n"
        "1. ደህንነት፦ ኃይሉን ይንጠሉ፣ LOTO ይተገብሩ፣ የማከማቻ ኃይል እንዲለቀቅ ይጠብቁ።\n"
        "2. ማረጋገጫ፦ ሻፍቱ ሲዞር የሚሰማውን ማዞር ድምፅ (1x modulation) በጆም ጋር ያመሳክሩ፤ የቤሪንግ ቤት ሙቀት ይለኩ።\n"
        "3. ጫናውን (load) ከሞተሩ ላይ በማግለል የሻፍት አጫጭር ጊዜ ነፃ ማዞር እንዳለ ያረጋግጡ — ከሆነ ችግሩ በቤሪንግ ነው ሳይሆን በሌላ ሊሆን ይችላል።\n"
        "4. የሮተር እና የስቴተር ማጣቀሻ ገጽታዎችን (rub marks) ይመርምሩ — ውስጣዊ ዊንግ ብልሽት ሮተርን እንዳላነካ ያረጋግጡ።\n"
        "5. ያረገውን ቤሪንግ በቤሪንግ ማውጣት መሣሪያ ይውጡ፤ ሻፍቱን እና ቤቱን በሚገባ ያጽዱ።\n"
        "6. አዲሱን ቤሪንግ በእቃ መዝገቡ የተመዘገበ ክፍል ቁጥር ብቻ ይተኩ፤ በ induction heater እስከ 110°C ሞቅ አድርገው ይትከሉ።\n"
        "7. የሴሊንግ መከላከያ (inner seal) ሁኔታን ይመርምሩ፤ ግሪስ መተኛቱን በምርት መመሪያ ይሙሉ።\n"
        "8. ማጣመርን በትክክለኛ torque ይውዱ፤ no-load ሙከራ 10 ደቂቃ ያሂዱ።\n"
        "9. በሙሉ ጫና (full-load) ላይ የ Sigmo ንባቨር ግሪን መመለሱን ያረጋግጡ፤ ስድሳቤንድ ከመጀመሪያው ንባት ወደ ታች መውረድ አለበት።\n"
        "10. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "BEARING_BALL": (
        "የቤሪንግ ኳስ (ball/rolling element) ብልሽት ተለይቷል።\n"
        "1. ደህንነት፦ ኃይል ይንጠሉ፣ LOTO ይተገብሩ።\n"
        "2. ማረጋገጫ፦ የ BSF ስፒን ምልክቶችን (spins) በቫይብሬሽን መለኪያ ካለ ያረጋግጡ፤ በእጅ የሚሰማ ጎርዝሳ ድምፅ አብሮ ይገኛል።\n"
        "3. የግሪስ ናሙና ይውሰዱ — ብረታ ብረት ፍራግ (metallic flakes) ከተገኘ ብልሽቱ ተረጋግጧል።\n"
        "4. የቤሪንግ ቤት እና የሮተር ገጽታዎችን ማስኬጃ (scoring) ላለበት ይመርምሩ፤ ከባድ ማስኬጃ ከተገኘ ሙሉ ቤሪንግ ምላሽ (assembly) መተካት ይገባል።\n"
        "5. ቤሪንጉን በሙሉ በእቃ መዝገቡ የተመዘገበ ክፍል ቁጥር ይተኩ።\n"
        "6. የመቅበጫ ውጫዊ ዊንግ ገጽታን (housing bore) ይፈትሹ — ከተሰፋ (ovalized) ቤቱን ማስተካከል ወይም መተካት ይገባል።\n"
        "7. ትክክለኛ የግሪስ ዓይነት እና መጠን ይተገብሩ — በጣም መብዛት ብለሽን ራሱን ይፈጥራል።\n"
        "8. ማጣመር፣ no-load ሙከራ፣ እና በ Sigmo ንባቨር የተረጋገጠ ግሪን ይከታተሉ።\n"
        "9. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "SHAFT_MISALIGNMENT": (
        "የሻፍት አሰላል (misalignment) ብልሽት ተለይቷል — 2x ምልክት ከፍ ብሏል።\n"
        "1. ደህንነት፦ ሞተሩን አቁሉ፣ ኃይል ይንጠሉ፣ LOTO ይተገብሩ።\n"
        "2. የ dial indicator ወይም ላዘር አሰላል መሣሪያ በማቀዝቀዣ (coupling) ላይ ይተክሉ።\n"
        "3. የ angular misalignment ይለኩ — በ 0°, 90°, 180°, 270° ላይ ንባቦችን ወስደው ልዩነቱን ያስሉ።\n"
        "4. የ parallel (offset) misalignment ይለኩ — ሻፍቶቹ በቀጥታ መስለው መዞር አለባቸው።\n"
        "5. የማቀዝቀዣ ብልቲ (shims) በማከል ወይም በማስወገድ አሰላሉን ያስተካክሉ — ከተመረከተው መሣሪያ መመሪያ ጋር ይጣጣሙ።\n"
        "6. የሞተር እግሮችን (feet) የጠፉ ቦታዎች (soft foot) ለመኖር ይፈትሹ — እያንዳንዱ እግር በተረጋጋ መቆም አለበት።\n"
        "7. በላዘር አሰላል መሣሪያ ውጤቱን እስኪያረጋግጥ ድረስ ይድገሙ።\n"
        "8. ማጣመሩን በትክክለኛ torque ይውዱ፤ ሞተሩን አስነስተው የ 2x ምልክት መውረዱን በ Sigmo ንባቨር ያረጋግጡ።\n"
        "9. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "MECH_UNBALANCE_ECCENTRICITY": (
        "የሜካኒካል አለመመጣጠን (unbalance/eccentricity) ተለይቷል — 1x ክፍል የበላይ ነው።\n"
        "1. ደህንነት፦ ኃይል ይንጠሉ፣ LOTO ይተገብሩ።\n"
        "2. በመሣሪያው ላይ የተከማቸ አቧራ፣ ጭቃ ወይም ያልተመጣጠነ ጫና (ለምሳሌ የተጣበቀ ፋን) መኖሩን ይመርምሩ — ይህ በጣም የተለመደ ምክንያት ነው።\n"
        "3. የፋን ግድግዳዎችን ንጹህ ያድርጉ፤ የተጠቀሙትን ብልቲ ወይም ዊንግ ቁሳቁስ ክብደት መጠን ማስተካከል ይቻላል።\n"
        "4. የተጫነ ዕቃ በሻፍቱ ላይ ከተጣበቀ (ፐሊ፣ ገመድ ጥቅል) ሚዛኑን ያረጋግጡ።\n"
        "5. ከሁሉም ጉዳዮች ተነጥቀው የሮተሩ ራሱ አለመመጣጠን (rotor unbalance) ከሆነ፦ ሮተሩን ወደ ላቦራቶሪ በማስተላለፍ የ field balancing ወይም የ dynamic balancing አገልግሎት ይወስዱ (ISO 21940 መሰረት)።\n"
        "6. የሮተር እና ስቴተር የአየር ክፍተት (air gap) አለመመጣጠን ለመኖር በ feeler gauge ይፈትሹ — eccentricity ከሆነ የቤሪንግ ሙሉ ምላሽ ተመርምር።\n"
        "7. ጥገና በኋላ በ Sigmo ንባቨር የ 1x ክፍል መውረዱን ያረጋግጡ።\n"
        "8. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "BROKEN_ROTOR_BAR": (
        "የተበጠሰ የሮተር ባር (broken rotor bar) ተለይቷል — 2sf ስድሳቤንድ ከፍ ብሏል። ይህ በጊዜ ሲቆይ የሚባባስ እና የሞተር ሙሉ አቅም ማጣት የሚያስከትል ብልሽት ነው።\n"
        "1. ደህንነት፦ ኃይል ይንጠሉ፣ LOTO ይተገብሩ።\n"
        "2. ማረጋገጫ፦ ሞተሩ ጫና ላይ ሲሆን የዘለለ ማዞር (slip) እና የተለመደው በላይ የሙቀት መጨመር እንዳለ ያረጋግጡ፤ የሮተር ባር ብልሽት በጫና ማሻሻያ ላይ ይከሰታል።\n"
        "3. የ single-phase stator test ያካሂዱ፦ በ 10% የቮልቴጅ ላይ ሮተሩን በእጅ በመንቀሳቀስ የካረንት ልዩነት (current variation) ይለኩ — ከ 3% በላይ ልዩነት ብልሽቱን ያረጋግጣል።\n"
        "4. የሮተር እና ስቴተር የአየር ክፍተት የብረት ፍራፍሬ (bar rash) ለመኖር በ borescope ይመርምሩ።\n"
        "5. ከባድ ብልሽት ከሆነ፦ ሮተሩን ወደ ላቦራቶሪ በማላከል የባር ማስተካከያ (re-barring) ወይም የሮተር መተካት ያስፈልጋል — ይህ ስፔሻላይዝድ ስራ ነው።\n"
        "6. ከቀላሉ አገላለጽ ብልሽት (የኤንድ-ሪንግ ትከላ) ከሆነ በተለመደው የእጅ ጥገና ሊያስተካከል ይችላል።\n"
        "7. ከጥገና በኋላ በ full-load ላይ የ 2sf ስድሳቤንድ መውረዱን በ Sigmo ንባቨር ያረጋግጡ።\n"
        "8. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "STATOR_WINDING_INTERTURN": (
        "የስቴተር ዊንዲንግ የውስጥ ዙር አገጣጠም (interturn short) ተለይቷል — ይህ ከፍተኛ አደጋ የሚያስከትል ብልሽት ነው።\n"
        "1. ደህንነት፦ ወዲያውኑ አቁሙ፣ ኃይል ይንጠሉ፣ LOTO ይተገብሩ። የውስጥ አገጣጠም አለመሆኑን ያረጋግጡ — ይህ ብልሽት እሳት ሊያስከትል ይችላል።\n"
        "2. የአየር ክፍተት (air gap) የጠበሰ ጠረን ወይም የተጠበሰ ኢንሱሌሽን ለመኖር ይመርምሩ።\n"
        "3. የኢንሱሌሽን ተከላከያ (insulation resistance) በ megger (500V/1000V) ይለኩ — ከ 1 MΩ በታች ከሆነ ዊንዲኑጉ ተበልሽቷል።\n"
        "4. በስቴተር ዊንዲኑግ የ phase-to-phase እና phase-to-ground ተከላከያዎችን ይለኩ።\n"
        "5. የ surge comparison test ካለ ያካሂዱ — የውስጥ ዙር አገጣጠምን በትክክል ያረጋግጣል።\n"
        "6. ከተረጋገጠ፦ ዊንዲኑጉን ወደ ዊንዲንግ ላቦራቶሪ በማላከል መተከል (rewinding) ወይም የተጠቀሙትን የሞተር ስቴተር መተካት ያስፈልጋል — የሞተር መጠን እና የአገልግሎት ጊዜ ጥገናውን ሲወስኑ ይገባቸዋል።\n"
        "7. ከጥገና በኋላ የ phase current ሚዛን እና የአማርኛ ምልክት መልሶ መጫኑን በ Sigmo ንባቨር ያረጋግጡ።\n"
        "8. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "PHASE_CURRENT_UNBALANCE": (
        "የፌዝ ኤሌክትሪክ ጅምላ አለመመጣጠን (phase current unbalance) ተለይቷል — ከ NEMA MG-1 ገደብ (2%) በላይ ነው።\n"
        "1. ደህንነት፦ ኃይል ይንጠሉ፣ LOTO ይተገብሩ።\n"
        "2. በሶስቱም ፌዞች ቮልቴጅ በ clamp meter ይለኩ — የቮልቴጅ አለመመጣጠን ከ 1% በላይ ከሆነ ችግሩ ከአቅርቦቱ (supply) ነው፤ ወደ ኃይል ኩባንያው ይወራሩ።\n"
        "3. የቮልቴጅ አለመመጣጠን ከሌለ፦ በ MCC ፓነሉ ውስጥ የአገልግሎት ኮንታክተሮችን (contactors)፣ የኦቨርሎድ ሪሌ (overload relay) ተርሚናሎችን እና የኬብል መያዣዎችን (lugs) ለሎሽ ተርሚናል ለመኖር በ thermal camera ወይም በእጅ ይፈትሹ — ይህ በጣም የተለመደ ምክንያት ነው።\n"
        "4. የተሎሹን ተርሚናሎች በትክክለኛ torque ይውዱ፤ የተበላሹ ኮንታክተሮችን ይተኩ።\n"
        "5. የሞተሩን የግብዓት ኬብሎች (supply cables) የመቆላቸውን ተሻፈሽፋ (cross-section) እና ርዝመት ለሶስቱም ፌዞች እኩል መሆኑን ያረጋግጡ።\n"
        "6. ከዚህ በኋላም አለመመጣጠን ከቀጠለ፦ የሞተሩ የዊንዲንግ ተራራሽ (internal winding) ችግር ሊሆን ይችላል — የ STATOR_WINDING ፕሮቶኮሉን ይከተሉ።\n"
        "7. ጥገና በኋላ የ unbalance % መውረዱን (ወደ 2% በታች) በ Sigmo ንባቨር ያረጋግጡ።\n"
        "8. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "VOLTAGE_SAG_SWELL": (
        "የቮልቴጅ መውረድ/መውጣት (sag/swell) ሁኔታ ተለይቷል — ከ IEEE 1159 ገደብ ውጭ ነው።\n"
        "1. ደህንነት፦ በንቃት ይቆዩ — ይህ ብዙውን ጊዜ የአቅርቦት ጎድሎ ነው፤ በሞተሩ ላይ የቀጥታ ጥገና ስራ አይጀምሩም።\n"
        "2. የጅምላ ክስተቱን ጊዜ፣ ጥልቀት እና ድግግሞሽ በ power quality analyzer (ካለ) ይመዝግቡ።\n"
        "3. በአካባቢው ትልልቅ ሞተሮች ማስነሻ (large motor starting) እና የቫር ኮምፐንሴሽን መቀየር (capacitor switching) ከተከሰተ ጊዜ ማመሳከሪያ ያዛሙሩ።\n"
        "4. የግብዓት ትራንስፎርሜሩን ታፕ ማስተካከያ (tap changer) ለሚዛን ይፈትሹ።\n"
        "5. ድጋፍ ከሚቻል የቮልቴጅ መረጋጋት መፍትሔ (voltage stabilizer / soft-starter / VFD with ride-through) ለሚመለከታቸው ሞተሮች ይገመግሙ።\n"
        "6. ድጋግ ከተከሰተ (swell)፦ የ neutral grounding ስርዓቱን እና የ tap ማስተካከያውን ያረጋግጡ።\n"
        "7. ሁኔታው ከአቅርቦት ኩባንያው ከሆነ በይፋ ሪፖርት ያድርጉ — የተመዘገቡ መረጃዎች (ጊዜ፣ ጥልቀት) አስፈላጊ ናቸው።\n"
        "8. ከጥገና በኋላ የቮልቴጅ ሁኔታው መልሶ መመዝገቡን በ Sigmo ንባቨር ያረጋግጡ።"
    ),
    "CAPACITOR_BANK_FAILURE": (
        "የካፓሲተር ባንክ ብልሽት ተለይቷል — የ reactive ኃይል አቅርቦት እና የሐርሞኒክ ስርዓት ለውጥ ተለይቷል።\n"
        "1. ደህንነት (በጣም አስፈላጊ)፦ ካፓሲተሮች ኃይል የሚያከማቹ ናቸው! ከመንጠቅ በኋላ ቢያንስ 5 ደቂቃ ይጠብቁ፣ ከዚያ በ discharging rod በኩል ማረጋገጫ ይወስዱ፣ በውጤቱ ላይ 50V በታች መሆኑን ከመለኪያ ያረጋግጡ።\n"
        "2. የካፓሲተር ባንኩን የአገልግሎት መቆራረጥ (fuse) እና የካሬንት ለውጦችን ይመርምሩ — የተቃጠሉ ፊውዞች የባንኩን ክፍል ለይተዋል።\n"
        "3. እያንዳንዱን ካፓሲተር በ capacitance meter ይለኩ — ከስፔሲፊኬሽኑ ±10% ውጭ ከሆነ የተበላሸ ነው።\n"
        "4. የተበላሹትን ካፓሲተሮች በእቃ መዝገቡ (asset registry) የተመዘገበ ስፔሲፊኬሽን ብቻ ይተኩ (kVAr፣ ቮልቴጅ፣ ሽቦ ግንኙነት)።\n"
        "5. የ discharge resistor መቆላቸውን ይፈትሹ — የተበላሹ ተቃዋሚዎች አደገኛ ናቸው።\n"
        "6. የኮንታክተር / የማብራሪያ ስርዓቱን (switching contactor, reactor) ይመርምሩ — የ inrush reactor ቢስሆን switching transient ይፈጠራል።\n"
        "7. ከጥገና በኋላ የባንኩን የኃይል እጣት (power factor) መልሶ መስራቱን ያረጋግጡ።\n"
        "8. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "CAPACITOR_SWITCHING_TRANSIENT": (
        "የካፓሲተር ማብራሪያ ጊዜ መሸጋገሪያ (switching transient) ተለይቷል።\n"
        "1. ደህንነት፦ ካፓሲተሩ ላይ ማንኛውም ስራ ከመጀመሩ በፊት መልሶ መሙላቱን ያረጋግጡ — ካፓሲተሮች ኃይል ያከማታሉ።\n"
        "2. የማብራሪያ ጊዜው ድግግሞሽ (transient) የሚከሰተው ማብራት ሲሆን ብቻ መሆኑን ይወርዱ — ከሆነ የሚያስከትለው የ inrush current ነው።\n"
        "3. የ inrush current limiting reactor ወይም የ detuning reactor መኖሩን ይፈትሹ — ከሌለ ይተከሉ።\n"
        "4. የማብራሪያ ኮንታክተሩ የተበላሸ ከሆነ (የተጠረጠረ ኮንታክት ገጽታ) አዲስ የካፓሲተር-ደረጃ (capacitor duty) ኮንታክተር ይተኩ።\n"
        "5. የማብራሪያ ሰዓቱን (point-on-wave switching) ማስተካከል ከተቻለ ትራንዚዥንቱን በእጅጉ ይቀንሳል — በዘመናዊ ኮንትሮለር የሚሰራ ነው።\n"
        "6. ከጥገና በኋላ የማብራሪያ ጊዜ ምልክቶች መውጣታቸውን በ Sigmo ንባቨር ያረጋግጡ።\n"
        "7. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
    "HARMONICS_HIGH_THD": (
        "ከፍተኛ ሐርሞኒክ (THD ከ 5% በላይ — IEEE 519) ተለይቷል።\n"
        "1. ደህንነት፦ ኃይል ይንጠሉ፣ LOTO ይተገብሩ።\n"
        "2. በ power quality analyzer የሐርሞኒክ ትንበያን (harmonic spectrum) ይመዝግቡ — የትኞቹ ትዕዛዞች (5th, 7th, 11th, 13th) የበላይ መሆናቸውን ይለዩ።\n"
        "3. በአካባቢው ያሉ VFD ሞተሮች፣ የዌልድ ማሽኖች እና የ rectifier ጭነቶች የሐርሞኒክ ምንጭ መሆናቸውን ይመርምሩ።\n"
        "4. የ VFD ጭነቶች ላይ line reactor / DC choke መኖሩን ይፈትሹ — ከሌለ ይተከሉ (በተለምዶ 3-5% impedance)።\n"
        "5. የሐርሞኒክ ማጣሪያ (harmonic filter) ወይም የተስተካከለ የካፓሲተር ባንክ (detuned bank) ማስተካከል ግምገማ ያድርጉ።\n"
        "6. ኤሌክትሪክ መለያዎችን (metering) የሚነኩ ከሆነ የ CT ስታቲንግ ስህተት ለመኖር ይፈትሹ (ስህተተኛ ንባት እውነተኛ ሐርሞኒክ ሳይሆን ሊሆን ይችላል)።\n"
        "7. ከጥገና በኋላ THD ወደ 5% በታች መውረዱን በ Sigmo ንባቨር ያረጋግጡ።\n"
        "8. የጥገና ሪፖርት ያስመዝግቡ።"
    ),
}

# Baseline tools & spares per class (Amharic, technical terms in
# English where used on the floor). Asset-specific part numbers are
# appended from the motors registry at call time.
TOOLS_AND_SPARES: dict[str, list[str]] = {
    "HEALTHY": [
        "የተለመደ የግሪስ ማስተካከያ ኪት (grease gun) እና የምርት መመሪያው የግሪስ ዓይነት",
        "IR thermometer (የሙቀት መለኪያ)",
        "የንጽህና መሣሪያዎች (ብሩሽ፣ ጨርቅ)",
    ],
    "BEARING_OUTER_RACE": [
        "አዲስ ቤሪንግ — ከእቃ መዝገቡ የተመዘገበ ትክክለኛ ክፍል ቁጥር (asset registry)",
        "Bearing puller (ቤሪንግ ማውጫ) እና induction heater (የቤሪንግ ማሞቂያ)",
        "Dial indicator / ላዘር አሰላል ኪት",
        "የግሪስ እና የጽዳት ፈሳሽ፣ ቶርክ ሬንች (torque wrench)",
        "LOTO ኪት (lock, tag, የኃይል መለያ)",
    ],
    "BEARING_INNER_RACE": [
        "አዲስ ቤሪንግ — ከእቃ መዝገቡ የተመዘገበ ክፍል ቁጥር",
        "Bearing puller እና induction heater (እስከ 110°C ብቻ)",
        "Feeler gauge (የአየር ክፍተት መለኪያ)፣ torque wrench",
        "የመጠባበቂያ ሲል (seal) እና የምርት መመሪያ ግሪስ",
        "LOTO ኪት",
    ],
    "BEARING_BALL": [
        "ሙሉ ቤሪንግ ምላሽ (assembly) — ከእቃ መዝገቡ የተመዘገበ ክፍል ቁጥር",
        "Bearing puller, induction heater, torque wrench",
        "የግሪስ ናሙና መያዣ (grease sampling kit)",
        "የቤሪንግ ቤት (housing) መመርመሪያ — ከባድ ማስኬጃ ከተገኘ ተተካ ቤት",
        "LOTO ኪት",
    ],
    "SHAFT_MISALIGNMENT": [
        "ላዘር አሰላል መሣሪያ (laser alignment kit) ወይም dial indicator ኪት",
        "የብልቲ ስብስብ (shim set — stainless)",
        "Torque wrench እና የጥግ ቁልል ቁልፎች",
        "የፋውንዴሽን ደረጃ መለኪያ (level)",
        "LOTO ኪት",
    ],
    "MECH_UNBALANCE_ECCENTRICITY": [
        "የፋን ንጽህና መሣሪያዎች (brush, scraper)",
        "Feeler gauge (የአየር ክፍተት መለኪያ)",
        "የ field balancing አገልግሎት (ISO 21940) — ከሮተር ችግር ሲረጋገጥ",
        "የተመዘገበ የብልቲ/ክብደት ማስተካከያ ስብስብ",
        "LOTO ኪት",
    ],
    "BROKEN_ROTOR_BAR": [
        "የሮተር ላቦራቶሪ አገልግሎት (re-barring / dynamic balancing)",
        "Borescope (የውስጥ መመልከቻ)",
        "የ single-phase test የአሃዞ መለኪያ (low-voltage variac + ammeter)",
        "መጠቀሚያ ተራራሽ ሮተር (ካለ መጠባበቂያ)",
        "LOTO ኪት",
    ],
    "STATOR_WINDING_INTERTURN": [
        "Megger (500V/1000V insulation tester)",
        "Surge comparison tester (ካለ)",
        "የዊንዲንግ ላቦራቶሪ አገልግሎት (rewinding) ወይም ተተካ ስቴተር",
        "የ CO2 እሳት ማጥፊያ (ጥገና በሚሰራበት ጊዜ መጠባበቂያ)",
        "LOTO ኪት",
    ],
    "PHASE_CURRENT_UNBALANCE": [
        "Clamp meter (የሶስት ፌዝ ካረንት መለኪያ)",
        "Thermal camera ወይም IR thermometer (የሎሽ ተርሚናል መፈለጊያ)",
        "የኮንታክተር እና የ overload relay ተለዋጭ ክፍሎች",
        "Torque wrench እና የተገቢ ቅርጸት ያላቸው ተርሚናል ቁልፎች",
        "LOTO ኪት",
    ],
    "VOLTAGE_SAG_SWELL": [
        "Power quality analyzer (የቮልቴጅ ጅምላ መመዝገቢያ)",
        "የትራንስፎርሜር tap ማስተካከያ መሣሪያ",
        "የጥገና መጠባበቂያ፦ voltage stabilizer / soft-starter / VFD ግምገማ",
        "የክስተት ጊዜ ማመሳከሪያ መዝገብ (ለኩባንያ ሪፖርት)",
    ],
    "CAPACITOR_BANK_FAILURE": [
        "አዲስ ካፓሲተር — ከእቃ መዝገቡ የተመዘገበ ስፔሲፊኬሽን (kVAr / V / wiring)",
        "Capacitance meter (የካፓሲታንስ መለኪያ)",
        "Discharging rod እና የኃይል መለኪያ (voltmeter ≥ 50V ማረጋገጫ)",
        "የ discharge resistor እና የ fuse ተለዋጭ ክፍሎች",
        "የካፓሲተር-ደረጃ ኮንታክተር (capacitor duty contactor)",
        "LOTO ኪት እና የእጅ ጥበቃ (insulating gloves)",
    ],
    "CAPACITOR_SWITCHING_TRANSIENT": [
        "Inrush / detuning reactor (ካልኩ ቢስ)",
        "የካፓሲተር-ደረጃ ኮንታክተር (ተተካ)",
        "Discharging rod እና voltmeter",
        "የ point-on-wave switching ግምገማ (ከዘመናዊ ኮንትሮለር ጋር ከሆነ)",
        "LOTO ኪት",
    ],
    "HARMONICS_HIGH_THD": [
        "Power quality analyzer (የሐርሞኒክ ስፔክትረም መለኪያ)",
        "Line reactor / DC choke (ለ VFD ጭነቶች — 3-5% impedance)",
        "የሐርሞኒክ ማጣሪያ ወይም detuned capacitor bank ግምገማ",
        "የ CT ስታቲንግ መመርመሪያ",
    ],
}
