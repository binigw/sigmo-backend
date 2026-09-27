# Sigmo V2 — የ Production Deployment መመሪያ (አማርኛ)

ይህ መረጃ የመጀመሪያ ደንበኛ (first customer) አገልግሎት ለመጀመር የሚያስፈልጉትን
ሁሉንም ደረጃዎች ይዟል። አሁን ያለው ቀጥታ አገልግሎት፦
**https://sigmo-backend-w4cx.onrender.com** (Render, Frankfurt)።

---

## 1. ማጣቀሻ (Architecture)

```
ESP32 ሞተሮች ──(X-API-Key: device)──► Render (FastAPI + v6c model)
                                        │
Replit Dashboard ──(X-API-Key: dashboard)─┤
                                        │
አስተዳደር (እርስዎ) ──(X-API-Key: admin)─────┤
                                        ▼
                                  Supabase PostgreSQL
                        (telemetry, RUL history, assets, rates)
```

---

## 2. API Keys (አስተዳደር)

ሁሉም የተጠበቁ endpoints `X-API-Key` header ይጠይቃሉ። ቁልፎች በየ scope እንደሚከተለው፡

| Scope | ምን ማድረግ ይችላል | ማን ይጠቀመው |
|---|---|---|
| `device` | `POST /api/v2/telemetry` ብቻ | ESP32 ኖዶች |
| `dashboard` | `GET /api/v2/motors/*` + `GET /api/v2/views/*` | Replit frontend / ሞባይል አፕ |
| `admin` | ሁሉም — telemetry + reads + `/api/v2/admin/*` + maintenance | እርስዎ (commissioning) |

**ክፍት endpoints (ቁልፍ አያስፈልጋቸውም):** `/api/v2/health`, `/api/v2/models/active`,
`/docs`, `/openapi.json`።

ቁልፎች በ `SIGMO_API_KEYS` environment variable ውስጥ ብቻ ይኖራሉ (Render Secrets)፡
ቅርጸት፡ `key1:device,key2:dashboard,key3:admin`

**መቀየር (rotation):** አዲስ ቁልፍ ፍጠሩ → በ Render → Environment ውስጥ
`SIGMO_API_KEYS` ያዘምኑ → Save & Deploy። ያለ ቁልፍ ማንኛውም የተጠበቀ endpoint
**503** ይመልሳል (fail-closed) — ስለዚህ ቁልፍ ሳይሰበሰብ አገልግሎት በጭራሽ ክፍት አይሆንም።

**ESP32 firmware ማስተካከያ** (HTTP POST ላይ አንድ መስመር ይጨምሩ)፡
```cpp
httpPost.addHeader("X-API-Key", "DEVICE-KEY-እዚህ");
```

**Replit frontend fetch ማስተካከያ**፡
```javascript
fetch(`${BASE}/api/v2/views/executive`, {
  headers: { "X-API-Key": "DASHBOARD-KEY-እዚህ" }
})
```

---

## 3. Supabase PostgreSQL ማዘጋጀት (የመጨረሻው የግድ እርምጃ!)

ያለ Supabase አገልግሎቱ ingest + verdict ብቻ ይሰጣል — telemetry አይቀመጥም፣
RUL አይሰላም፣ asset registry አይሰራም (degraded mode)።

1. **[supabase.com](https://supabase.com)** በግባ → **New project**
2. - Name: `sigmo-production`
   - Database Password: **ጠንካራ የይለፍ ቃል ፍጠሩ እና ያስቀምጡ** (ለምሳሌ 20+ ቁምፊዎች)
   - Region: **Central EU (Frankfurt)** — ከ Render ጋር አንድ ክልል (ዝቅተኛ latency)
3. Project ሲነሳ (≈2 ደቂቃ)፡ **Connect** (ወይም Project Settings → Database) →
   **Connection pooling → Session mode** URI ቅዳ። ቅርጸቱ እንዲህ ይመስላል፡
   ```
   postgresql://postgres.abcdefghijklmnop:[YOUR-PASSWORD]@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
   ```
4. `[YOUR-PASSWORD]` የፈጠሩትን የይለፍ ቃል በመተካት ይስተካክሉ፣ እና በመጨረሻ
   `?sslmode=require` ይጨምሩ፡
   ```
   postgresql://postgres.abcdefghijklmnop:YourPassword@aws-0-eu-central-1.pooler.supabase.com:5432/postgres?sslmode=require
   ```
   > በጥንቃቄ፡ **Session mode (port 5432)** ብቻ ይጠቀሙ — Transaction mode
   > (port 6543) ከ asyncpg prepared statements ጋር አይስማማም።
5. ይህን ሙሉ ሐረግ (connection string) በ **Render → sigmo-backend →
   Environment → Add Environment Variable** ውስጥ፡
   - Key: `DATABASE_URL`
   - Value: (ከደረጃ 4 የወጣው)
   ከዚያ **Save** (በራስ-ሰር redeploy ያደርጋል)።
6. **ማረጋገጫ፡** `https://sigmo-backend-w4cx.onrender.com/api/v2/health` ሲጎትቱ
   `"database_connected": true` መሆኑን ያዩ።
7. **Schema በራስ-ሰር ይፈጠራል** — ምንም SQL በእጅ መሮጥ አያስፈልግም፤ አገልግሎቱ ሲነሳ
   `schema.sql` በ idempotent መንገድ ይተገበራል (motors, telemetry_features,
   plant_config, fault_logs, baselines + 90-day cleanup)።

---

## 4. Commissioning Checklist (የመጀመሪያ ደንበኛ)

በዚህ ቅደም ተከተል፡

1. ✅ Supabase ተያይዟል (ከላይ ያለው ክፍል 3)
2. ✅ Asset registry ሙላ — `PUT /api/v2/admin/motor-assets/{motor_id}`
   (MCC Panel, Cabinet, Line/Zone, kW, RPM, Drive Type, የቤሪንግ ክፍል ቁጥር,
   downtime ወጪ/ሰዓት) — admin key ያስፈልጋል
   ⚠️ **ይሄ ከ ESP32 ቴሌሜትሪ በፊት መጀመር አለበት** — ያልተመዘገበ ሞተር ቴሌሜትሪ
   አሁን በ API ይታገዳል (422 + ግልጽ መልእክት፤ §11 Cold Start)። ምክንያቱም
   `telemetry_features.motor_id` የሚመለከተው `motors` ሠንጠረዥ ነው (FK)።
3. ✅ የንግድ ወጪዎች ሙላ — `PUT /api/v2/admin/plant-config/energy_tariff_etb_per_kwh`
   (የ EEU ታሪፍ) + `default_downtime_cost_etb_per_hour` + ከተለካ
   `plant_power_factor` + `operating_hours_per_month`
4. ✅ ESP32 ኖዶችን በ device key ያዋቅሩ (16 × 2048-ናሙና frames @ 10240 Hz)
5. ✅ Replit dashboard ላይ dashboard key ያስገቡ
6. ✅ የመጀመሪያ 24 ሰዓት telemetry ሲሰበሰብ RUL በራሱ ይጀምራል (§14.3 cold-start)፤
   በመጀመሪያ `"Calculating... (Insufficient Data)"` ማየት መደበኛ ነው

---

## 5. የተለመዱ ችግሮች (Troubleshooting)

| ምልክት | ምክንያት / መፍትሔ |
|---|---|
| ሁሉም ጥሪ 503 | `SIGMO_API_KEYS` አልተሰበሰበም — Render Environment ያስተካክሉ |
| 401 | ቁልፍ የለም ወይም የተሳሳተ — `X-API-Key` header ያረጋግጡ |
| 403 | ትክክለኛ ቁልፍ ግን የተሳሳተ scope — ለምሳሌ dashboard key በ telemetry POST ላይ |
| telemetry POST 422 "not registered" | ሞተሩ በ Asset registry አልተመዘገበም — መጀመሪያ `PUT /api/v2/admin/motor-assets/{motor_id}` ቁምሉ፣ ከዚያ ESP32 እንደገና ይላክ (በኋላ በራሱ ይቀጥላል) |
| `database_connected: false` | `DATABASE_URL` የለም ወይም የተቆራረጠ — ክፍል 3 ይድገሙ |
| የመጀመሪያ ጥሪ ቀላስ | Render free tier ከ15 ደቂቃ ስራ በሌለ በኋላ ያንቀላፋል (≈60ሰ ማስነሻ) — ለ24/7 Starter $7/ወር |

---

## 6. የምስራች ፋይሎች

| Repo | ይዘት |
|---|---|
| `github.com/binigw/sigmo-backend` | የሙሉ codebase (Render ከዚህ ይገነባል) |
| `huggingface.co/Bini-K/Sigmo-backend` | Model artifacts (tag-pinned `v6.20260925.045900`) |
| ይህ workspace (`sigmo_v2/`) | Source of truth — tests, corpora, trainer |

## AI Technician Analysis (optional env vars)

`POST /api/v2/ai/explain` feeds the real V2 telemetry evidence (health
index, THD, unbalance, crest factor, rotor sidebands, per-phase RMS,
the v6c verdict + probabilities, stage, RUL and the system's own
Amharic repair protocol) to a real LLM, which answers as an expert
motor technician in Amharic AND English. The route returns an honest
503 until one provider key is configured:

| Variable | Meaning |
|---|---|
| `OPENAI_API_KEY` | Use OpenAI (default model `gpt-4o-mini`) |
| `DEEPSEEK_API_KEY` | Use DeepSeek (default model `deepseek-chat`) |
| `GEMINI_API_KEY` | Use Google Gemini (default model `gemini-2.0-flash`) |
| `AI_PROVIDER` | Optional: force `openai`, `deepseek` or `gemini` |
| `OPENAI_MODEL` / `DEEPSEEK_MODEL` / `GEMINI_MODEL` | Optional model override |

Priority: explicit `AI_PROVIDER`, then the first key found
(openai -> deepseek -> gemini). No key = honest 503, never a fabricated
answer. Scope: dashboard/admin keys only.
