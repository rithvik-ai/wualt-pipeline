"""
Minimal FastAPI server that exposes the standalone preprocessing pipeline
with a single-page frontend.

Run:
    uvicorn pipeline_server:app --reload --port 8001

Open:
    http://localhost:8001/

Click "Refresh" — a new random raw sensor frame is generated, pushed through
the pipeline (preprocessing_pipeline.AnomalyInputBuilder), and both the raw
input and processed output are rendered side-by-side.
"""

import random
from typing import Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from preprocessing_pipeline import AnomalyInputBuilder


app = FastAPI(title="WUALT Preprocessing Pipeline — Demo")

# One stateful builder for the whole process. Despikers, EMAs, baseline,
# rolling window — all persist across refreshes so you can watch the
# cold-start → warmed-up transition.
builder = AnomalyInputBuilder(window_seconds=30, baseline_warmup=5)

# Sequence counter — incremented every /process call.
_seq = {"n": 0}


def _random_raw_frame() -> Dict:
    """Generate one synthetic raw sensor frame in the firmware shape.

    Every field the firmware emits is populated here so the pipeline
    actually exercises all of its gates (PPG quality from adc_raw, thermal
    comp from die_temp, charging/battery gates from vbat_mv/vbus_present).
    """
    _seq["n"] += 1

    scenario = random.choices(
        ["rest", "stress", "exercise", "vigorous_exercise", "fever",
         "mild_hypoxemia", "hypoxia", "charging", "low_battery", "noisy"],
        weights=[38, 12, 8, 6, 6, 6, 6, 6, 6, 6],
    )[0]

    # --- Physiological defaults ---
    if scenario == "rest":
        hr, spo2, temp, acc = random.gauss(72, 3), random.gauss(98.5, 0.4), random.gauss(36.6, 0.15), random.gauss(1.0, 0.02)
    elif scenario == "stress":
        hr, spo2, temp, acc = random.gauss(105, 5), random.gauss(97.5, 0.6), random.gauss(37.1, 0.15), random.gauss(1.0, 0.03)
    elif scenario == "exercise":
        hr, spo2, temp, acc = random.gauss(140, 6), random.gauss(97.0, 0.7), random.gauss(37.3, 0.2), random.gauss(1.15, 0.08)
    elif scenario == "vigorous_exercise":
        # 185–200 bpm — previously (wrongly) flagged as out-of-range.
        # Now accepted thanks to the 200 bpm SQI upper bound.
        hr, spo2, temp, acc = random.gauss(188, 5), random.gauss(96.5, 0.8), random.gauss(37.5, 0.2), random.gauss(1.2, 0.1)
    elif scenario == "fever":
        # Elevated skin temp → surfaces `elevated_skin_temp` clinical flag
        # without affecting SQI or rejection.
        hr, spo2, temp, acc = random.gauss(95, 5), random.gauss(97.8, 0.4), random.gauss(37.9, 0.2), random.gauss(1.0, 0.02)
    elif scenario == "mild_hypoxemia":
        # SpO2 91–93% → `spo2_clinical_concern` clinical flag.
        hr, spo2, temp, acc = random.gauss(92, 4), random.gauss(92.0, 0.8), random.gauss(36.8, 0.15), random.gauss(1.0, 0.02)
    elif scenario == "hypoxia":
        hr, spo2, temp, acc = random.gauss(110, 5), random.gauss(86.0, 1.5), random.gauss(36.8, 0.15), random.gauss(1.0, 0.02)
    elif scenario == "charging":
        # Charging → off-wrist, die_temp elevated, skin temp biased upward
        # (the thermal compensator will correct this downward).
        hr, spo2, temp, acc = random.gauss(72, 3), random.gauss(98.5, 0.4), random.gauss(38.2, 0.3), random.gauss(1.0, 0.01)
    elif scenario == "low_battery":
        hr, spo2, temp, acc = random.gauss(72, 3), random.gauss(98.5, 0.4), random.gauss(36.6, 0.15), random.gauss(1.0, 0.02)
    else:  # noisy — bad HR, bad ADC, sometimes finger off
        hr   = random.choice([random.gauss(72, 3), 240.0, None])
        spo2 = random.gauss(98.0, 0.5)
        temp = random.gauss(36.6, 0.15)
        acc  = random.gauss(1.0, 0.1)

    # --- ADC raw (PPG photodiode) ---
    # Healthy pulsatile signal lives in mid-range with visible jitter.
    # Bad frames: rail-saturated, flat-lined, or just missing.
    if scenario == "noisy":
        adc_raw = random.choice([
            random.gauss(110000, 800),            # flat — "no pulse"
            260000,                               # saturated high
            500,                                  # saturated low
            random.gauss(110000, 4000),           # occasionally fine
        ])
    elif scenario == "charging":
        adc_raw = random.gauss(800, 200)          # off-wrist → near zero
    else:
        adc_raw = random.gauss(110000, 4500)      # ~4.5k σ = healthy pulse

    # --- Accelerometer split into 3 axes (gravity dominates z) ---
    az = acc * random.uniform(0.88, 0.98)
    ax = random.uniform(-0.1, 0.1)
    ay = random.uniform(-0.1, 0.1)

    # --- Battery / charger / thermal ---
    if scenario == "low_battery":
        vbat_mv = random.randint(3250, 3450)      # below VBAT_LOW_MV
    else:
        vbat_mv = random.randint(3700, 4100)

    if scenario == "charging":
        charger_stat = random.choice(["charging", "fast_charge"])
        vbus_present = True
        die_temp = random.gauss(42.0, 0.5)        # chip is hot
    else:
        charger_stat = "idle"
        vbus_present = False
        die_temp = random.gauss(34.5, 0.4)

    finger_on = (scenario != "charging") and (random.random() > 0.05)

    return {
        "accel_x":      round(ax, 4),
        "accel_y":      round(ay, 4),
        "accel_z":      round(az, 4),
        "adc_raw":      int(max(0, adc_raw)),
        "heart_rate":   round(hr, 2) if hr is not None else None,
        "spo2":         round(spo2, 2),
        "temperature":  round(temp, 2),
        "vbat_mv":      vbat_mv,
        "die_temp":     round(die_temp, 2),
        "charger_stat": charger_stat,
        "vbus_present": vbus_present,
        "finger_on":    finger_on,
        "sequence":     _seq["n"],
        "status":       "ok",
        "_scenario":    scenario,                 # for display only
    }


@app.get("/process")
def process() -> JSONResponse:
    raw = _random_raw_frame()
    out = builder.step(raw)
    return JSONResponse({
        "raw":   raw,
        "out":   out,
        "stats": builder.pre.stats(),
    })


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ---------------------------------------------------------------------------
# Single-file frontend
# ---------------------------------------------------------------------------

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WUALT Preprocessing Pipeline</title>
<style>
  :root {
    --bg:       #17181a;   /* matte charcoal */
    --card:     #1e1f22;   /* flat slate */
    --card-2:   #232428;   /* nested card */
    --fg:       #d6d3cc;   /* warm off-white */
    --muted:    #797974;
    --accent:   #a8b5a0;   /* desaturated sage */
    --accent-2: #c9bfa8;   /* muted sand */
    --warn:     #b87a6f;   /* muted terracotta */
    --ok:       #8ea689;   /* muted olive */
    --border:   #2a2b2f;
  }
  * { box-sizing: border-box; }
  html, body {
    margin:0; padding:0;
    background: var(--bg);
    color: var(--fg);
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    min-height:100vh;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "ss01", "cv11";
  }
  header {
    padding: 26px 36px;
    border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; gap:16px;
    background: var(--bg);
  }
  header h1 {
    font-size:15px; margin:0; font-weight:500;
    letter-spacing:1.6px; text-transform:uppercase;
    color: var(--fg);
  }
  header .sub {
    color:var(--muted); font-size:11px; margin-top:5px;
    letter-spacing:.4px;
  }
  button {
    background: transparent;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 9px 22px;
    font-family: inherit; font-weight:500; font-size:12px;
    border-radius: 2px;
    cursor: pointer;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    transition: background .15s, color .15s;
  }
  button:hover  { background: var(--accent); color: var(--bg); }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: .5; cursor: default; }
  main {
    display:grid; grid-template-columns:1fr 1fr; gap:20px;
    padding: 24px 36px;
  }
  @media (max-width: 900px) { main { grid-template-columns:1fr; } }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 20px 22px;
  }
  .card h2 {
    font-size:10px; margin:0 0 16px; letter-spacing:2px;
    color:var(--muted); font-weight:500; text-transform:uppercase;
  }
  pre {
    margin:0; white-space:pre-wrap; word-break:break-word;
    font-size:11.5px; line-height:1.65; color: var(--fg);
  }
  .pill {
    display:inline-block; padding:3px 10px; border-radius:2px;
    font-size:10px; font-weight:500; letter-spacing:1.1px;
    text-transform: uppercase;
    border: 1px solid var(--border);
  }
  .pill.ok   { color: var(--ok);   border-color: rgba(142,166,137,.45); }
  .pill.bad  { color: var(--warn); border-color: rgba(184,122,111,.45); }
  .row {
    display:flex; gap:10px; align-items:center; margin-bottom:12px;
    flex-wrap:wrap;
  }
  .k {
    color:var(--muted); font-size:10px; text-transform:uppercase;
    letter-spacing:1.2px;
  }
  .big { font-size:20px; font-weight:500; color: var(--fg); }
  .metrics {
    display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-bottom:16px;
  }
  @media (max-width: 700px) { .metrics { grid-template-columns:repeat(2, 1fr); } }
  .metric {
    background: var(--card-2);
    padding: 11px 13px;
    border-radius: 3px;
    border: 1px solid var(--border);
  }
  .metric .k { font-size:9px; }
  .metric .v {
    font-size:15px; font-weight:500; margin-top:3px; color: var(--accent-2);
  }
  .sqi-bar {
    height: 3px; background: var(--card-2); border-radius: 0;
    overflow: hidden; margin-top: 6px;
    border: 1px solid var(--border);
  }
  .sqi-bar > div {
    height:100%; background: var(--accent); transition: width .25s ease;
  }
  .reason {
    display:inline-block;
    background: transparent;
    color: var(--warn);
    border: 1px solid rgba(184,122,111,.35);
    padding: 3px 9px; border-radius: 2px;
    font-size: 10px; margin: 3px 5px 3px 0;
    letter-spacing: .4px;
  }
  .clinical {
    display:inline-block;
    background: transparent;
    color: var(--accent-2);
    border: 1px dashed rgba(201,191,168,.55);
    padding: 3px 9px; border-radius: 2px;
    font-size: 10px; margin: 3px 5px 3px 0;
    letter-spacing: .4px;
  }
  footer {
    padding: 14px 36px; color: var(--muted); font-size: 10px;
    border-top: 1px solid var(--border);
    letter-spacing: .4px;
  }
  ::selection { background: rgba(168,181,160,.25); }
</style>
</head>
<body>
<header>
  <div>
    <h1>WUALT — Preprocessing Pipeline</h1>
    <div class="sub">raw sensor frame → validate → clean → normalize → SQI → baseline → window</div>
  </div>
  <button id="refresh">↻ Refresh</button>
</header>

<main>
  <div class="card">
    <h2>Raw Sensor Frame</h2>
    <div id="scenario-pill"></div>
    <pre id="raw">click Refresh to start</pre>
  </div>

  <div class="card">
    <h2>Processed Output (rule-engine input)</h2>
    <div class="row">
      <span class="k">accepted</span>
      <span id="accepted" class="pill ok">—</span>
      <span class="k" style="margin-left:14px;">baseline</span>
      <span id="baseline" class="pill">—</span>
      <span class="k" style="margin-left:14px;">charging</span>
      <span id="charging" class="pill">—</span>
    </div>

    <div class="metrics">
      <div class="metric"><div class="k">HR</div><div class="v" id="m-hr">—</div></div>
      <div class="metric"><div class="k">SpO₂</div><div class="v" id="m-spo2">—</div></div>
      <div class="metric"><div class="k">Temp (corr.)</div><div class="v" id="m-temp">—</div></div>
      <div class="metric"><div class="k">dyn |a|</div><div class="v" id="m-acc">—</div></div>
      <div class="metric"><div class="k">ADC raw</div><div class="v" id="m-adc">—</div></div>
      <div class="metric"><div class="k">Battery</div><div class="v" id="m-vbat">—</div></div>
      <div class="metric"><div class="k">Die temp</div><div class="v" id="m-die">—</div></div>
      <div class="metric"><div class="k">Thermal bias</div><div class="v" id="m-bias">—</div></div>
    </div>

    <div class="row">
      <span class="k">SQI overall</span>
      <span id="sqi-val" class="big">—</span>
    </div>
    <div class="sqi-bar"><div id="sqi-fill" style="width:0%"></div></div>

    <div id="sqi-breakdown" style="margin-top:10px; display:grid; grid-template-columns:repeat(6,1fr); gap:8px;"></div>

    <div id="reasons" style="margin-top:10px;"></div>
    <div id="clinical" style="margin-top:4px;"></div>

    <div style="margin-top:14px;">
      <div class="k" style="margin-bottom:6px;">Full output JSON</div>
      <pre id="out">—</pre>
    </div>
  </div>
</main>

<footer>
  Stateful: despiker, EMA, personal baseline, 30 s rolling window — all persist across refreshes.
  Watch the z-scores and window stats stabilize after ~5 frames.
</footer>

<script>
const $ = (id) => document.getElementById(id);
const fmt = (v, d=2) => v === null || v === undefined ? "—" : (typeof v === "number" ? v.toFixed(d) : v);

async function refresh() {
  const btn = $("refresh");
  btn.disabled = true; btn.textContent = "…";
  try {
    const res = await fetch("/process");
    const data = await res.json();
    render(data);
  } catch (e) {
    $("out").textContent = "error: " + e;
  } finally {
    btn.disabled = false; btn.textContent = "↻ Refresh";
  }
}

function render(data) {
  const raw = data.raw;
  const sample = data.out.sample;
  const sqi = sample.sqi;

  // Raw frame JSON
  $("raw").textContent = JSON.stringify(raw, null, 2);

  // Scenario pill
  const scen = raw._scenario;
  $("scenario-pill").innerHTML =
    `<span class="pill ${scen === 'rest' ? 'ok' : 'bad'}">scenario: ${scen}</span>`;

  // Accepted / baseline / charging pills
  $("accepted").textContent = sample.accepted ? "yes" : "no";
  $("accepted").className = "pill " + (sample.accepted ? "ok" : "bad");
  $("baseline").textContent = data.out.baseline_ready ? "ready" : "warming up";
  $("baseline").className = "pill " + (data.out.baseline_ready ? "ok" : "bad");
  $("charging").textContent = sample.charging ? "yes" : "no";
  $("charging").className = "pill " + (sample.charging ? "bad" : "ok");

  // Metric cards
  $("m-hr").textContent   = fmt(sample.hr, 1) + (sample.hr !== null ? " bpm" : "");
  $("m-spo2").textContent = fmt(sample.spo2, 1) + (sample.spo2 !== null ? " %" : "");
  $("m-temp").textContent = fmt(sample.temp, 2) + (sample.temp !== null ? " °C" : "");
  $("m-acc").textContent  = fmt(sample.dyn_acc_mag, 3) + " g";
  $("m-adc").textContent  = sample.adc_raw !== null ? sample.adc_raw.toLocaleString() : "—";
  $("m-vbat").textContent = sample.battery_mv !== null ? sample.battery_mv + " mV" : "—";
  $("m-die").textContent  = fmt(sample.die_temp, 2) + (sample.die_temp !== null ? " °C" : "");
  $("m-bias").textContent = fmt(sample.thermal_bias, 2) + " °C";

  // SQI bar + breakdown
  $("sqi-val").textContent = fmt(sqi.overall, 2);
  $("sqi-fill").style.width = Math.round(sqi.overall * 100) + "%";

  const sqiKeys = [["hr","HR"],["hrv","HRV"],["temp","Temp"],["acc","Acc"],["spo2","SpO₂"],["ppg","PPG"]];
  $("sqi-breakdown").innerHTML = sqiKeys.map(([k,label]) => `
    <div class="metric" style="padding:6px 8px;">
      <div class="k">${label}</div>
      <div style="font-size:12px;">${sqi[k].toFixed(2)}</div>
      <div class="sqi-bar" style="height:4px;margin-top:4px;"><div style="width:${Math.round(sqi[k]*100)}%;background:var(--accent);height:100%;"></div></div>
    </div>
  `).join("");

  // Reject reasons (quality failures)
  const reasons = sample.reject_reasons || [];
  $("reasons").innerHTML = reasons.length
    ? reasons.map(r => `<span class="reason">${r}</span>`).join("")
    : "";

  // Clinical flags (advisory — don't block acceptance)
  const clinical = sample.clinical_flags || [];
  $("clinical").innerHTML = clinical.length
    ? clinical.map(r => `<span class="clinical">⚕ ${r}</span>`).join("")
    : "";

  // Full output JSON
  $("out").textContent = JSON.stringify(data.out, null, 2);
}

$("refresh").addEventListener("click", refresh);
refresh();
</script>
</body>
</html>
"""
