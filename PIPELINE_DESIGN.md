# WUALT — Preprocessing Pipeline Design

Companion document to `preprocessing_pipeline.py`. Explains *why* each stage
exists, *what* it does, and *what assumptions* it makes.

```
raw frame ──► validate ──► clean ──► normalize ──► SQI ──► baseline ──► window ──► rule engine
              (stage 1)   (stage 2)  (stage 3)    (stage 4) (stage 5)  (stage 6)
```

Every stage is O(1) per frame. For a 1 Hz sensor stream the end-to-end cost
is < 1 ms on a Raspberry Pi-class CPU, comfortably inside the 700 ms budget.

---

## Assumptions

- **Sample rate**: ~1 Hz (firmware pushes one `latest_sensor_data` per second).
- **No per-beat timestamps**: the firmware only exposes instantaneous `heart_rate`,
  not R-peak times — that forces the HRV proxy decision in §3.
- **Single user per ring**: the personal baseline is per-device, no identity
  switching logic.
- **Gravity-normalised accelerometer**: the firmware already reports g, so the
  resting `acc_mag` sits near 1.0.
- **Stationary device ≠ stationary wearer**: `finger_on` is the truth signal
  for PPG contact; `accel_mag` variance is the truth signal for motion.

---

## 1. Signal Cleaning

**Goals**: survive null/invalid/out-of-range values, surface bad data instead
of silently imputing.

| Signal       | Valid range         | On violation                      |
|--------------|--------------------|-----------------------------------|
| `heart_rate` | 30 – 220 bpm       | hold last-good, flag `hr_missing` |
| `spo2`       | 60 – 100 %         | hold last-good                    |
| `temperature`| 25 – 42 °C         | hold last-good, flag              |
| `accel_x/y/z`| treated as raw g   | defaulted to 0.0 if null          |
| `finger_on`  | bool               | missing → `False`                 |

**Why hold-last-good rather than NaN propagation?** A rule engine that sees a
brief null will often flap between states. Holding the last good value for a
few seconds plus flagging the frame as rejected gives the rule engine a
clean "ignore this frame" signal without destabilizing downstream buffers.

**Despike** — *median-of-3 with max-jump gate*:

```python
if abs(value - last_good) > MAX_JUMP[signal]:
    value = median(last_3_values + [value])
```

Catches the classic PPG artifact where a movement spike briefly reports HR
like 240 bpm. Max jumps: HR 25 bpm/s, SpO₂ 5 %/s, temp 0.6 °C/s.

Reference: `_Despiker` class in the pipeline file.

---

## 2. Noise Reduction

Keep it cheap. No FFTs, no Kalman, no ML.

| Signal       | Filter              | Rationale                                   |
|--------------|---------------------|---------------------------------------------|
| `heart_rate` | EMA α=0.5           | Removes jitter, ~1 s lag                    |
| `spo2`       | EMA α=0.4           | SpO₂ is slow — a little extra smoothing ok  |
| `temperature`| EMA α=0.3           | Skin temp is very slow                      |
| `acc_mag`    | *no filter* on raw  | We *want* high-frequency jitter for the SQI |

EMA (Exponential Moving Average):

```
y_t = α·x_t + (1 − α)·y_{t−1}
```

O(1) state, O(1) cost, trivially real-time.

**Why not a moving average?** Moving average over N samples needs a ring
buffer and has a harder group delay. EMA is strictly cheaper and its lag is
predictable.

**Accelerometer** — we keep a 10-sample rolling buffer specifically to compute
variance for the acc-SQI score. We do **not** smooth `acc_mag` before feeding
the rule engine, because the rule engine uses sudden jumps (falls, shocks).

---

## 3. Feature Transformation

### 3.1 `acc_mag`

```python
acc_mag = sqrt(ax² + ay² + az²)      # in g
```

Resting ring ≈ 1.0 g (gravity). Walking ≈ 1.05–1.15 g. Fall spike ≈ 3–5 g.

### 3.2 HRV

**Gold standard**: RMSSD over successive R-R intervals from an ECG or PPG
R-peak stream. We don't have that — the firmware only gives us one HR
number per second.

**MVP proxy**:

```
rr_proxy_ms = 60000 / hr
rmssd = sqrt( mean( (rr_t − rr_{t-1})² ) )   over a rolling 30-sample buffer
```

This captures HR *stability* changes (stress, panic, bradycardia) well
enough for short-window rules, but is **not** clinical RMSSD. We flag this
honestly by capping `sqi.hrv ≤ 0.6`.

Upgrade path: when firmware can expose inter-beat intervals, swap
`_RRBuffer` for a real IBI-driven implementation; the rule engine contract
doesn't change.

---

## 4. Normalization — Personal Baseline Z-Scores

**Why not population norms?** A 62 bpm resting HR is healthy for one user and
bradycardia for another. Rule-based anomaly detection works best on
**deviation from this user's own baseline**.

### Rolling baseline

Per-signal 300-sample rolling buffer (≈ 5 minutes at 1 Hz). On every
*accepted* frame:

```
mean_i = Σx / n
std_i  = sqrt( Σ(x − mean)² / (n − 1) )
z_i    = (x − mean_i) / std_i
```

Only frames with `accepted == True` are used to update the baseline, so
junk frames don't poison the reference.

### Cold-start problem

The first ~60 seconds don't have enough samples for a meaningful z-score.
Strategy:

| State           | `baseline_ready` | Z-scores emitted | Rule engine should...                 |
|-----------------|------------------|------------------|---------------------------------------|
| frames < 60     | `False`          | all 0.0          | apply **absolute** rules only         |
| frames ≥ 60     | `True`           | real             | apply absolute + **deviation** rules  |

Absolute rules (e.g. `spo2 < 88`, `temp > 38.5`, `fall spike > 3 g`) are
always valid. Deviation rules (`|z_hr| > 2.5`) wait for warmup.

Reference: `PersonalBaseline` class.

---

## 5. Signal Quality Index (SQI)

Simple weighted scoring — no ML.

| Signal | Score logic                                                             |
|--------|-------------------------------------------------------------------------|
| `hr`   | 0.95 if range-valid AND `finger_on`, 0.3 if not, 0.2 if out-of-range    |
| `hrv`  | `min(0.6, sqi.hr)` (capped because it's a proxy)                        |
| `temp` | 0.9 if 30 ≤ T ≤ 40.5, else 0.3                                          |
| `spo2` | 0.9 if range-valid AND `finger_on`, 0.3 if not                          |
| `acc`  | `1 − clamp(50·var(acc_mag over 10 samples))` — low jitter → high SQI    |

**Overall**:

```
overall = Σ(w_i · s_i) / Σ(w_i)     over signals with s_i > 0
weights = {hr:0.35, hrv:0.15, temp:0.15, acc:0.20, spo2:0.15}
```

**Hard gates that reject the frame**:

- `finger_on == False` → `finger_off`
- `hr` out of range or missing
- `sqi.hr < 0.7`
- `sqi.acc < 0.7` (excessive motion → PPG unreliable)
- `temp` missing

A rejected frame is still passed to the rule engine with `accepted=False`
and a list of `reject_reasons`, so the engine can surface *"low quality
signal"* to the user instead of silently dropping data.

---

## 6. Windowing

**Window**: 30 seconds rolling.

**Why 30 s?**
- Long enough to compute stable mean/variance for HR, SpO₂, temp.
- Short enough for sub-minute alerting latency.
- Aligns with common short-term HRV clinical windows.

**Per-signal features** (mean, variance, min, max):

```python
for k in (hr, hrv_rmssd, temp, spo2, acc_mag):
    feats[f"{k}_mean"] = mean(window[k])
    feats[f"{k}_var"]  = var(window[k])
    feats[f"{k}_min"]  = min(window[k])
    feats[f"{k}_max"]  = max(window[k])
feats["window_n"] = len(window)
```

**Streaming, not batched**: `RollingWindow.add()` pops out-of-range
samples based on timestamp, so the window is always current with O(1)
amortized cost. Trivially real-time.

Reference: `RollingWindow` class.

---

## 7. Output Format — the contract with the rule engine

One `AnomalyInputBuilder.step(raw)` call returns:

```python
{
    "sample": {
        "timestamp": int,
        "sequence":  int,
        "hr":        float | None,
        "hrv_rmssd": float | None,
        "temp":      float | None,
        "spo2":      float | None,
        "acc_mag":   float,
        "acc_x":     float, "acc_y": float, "acc_z": float,
        "finger_on": bool,
        "sqi": {"hr", "hrv", "temp", "acc", "spo2", "overall"},  # all 0..1
        "accepted":       bool,
        "reject_reasons": list[str],
    },

    "zscores": {
        "hr":        float,   # 0.0 during cold-start
        "hrv_rmssd": float,
        "temp":      float,
        "spo2":      float,
        "acc_mag":   float,
    },

    "baseline_ready": bool,   # False => deviation rules should be skipped

    "window": {
        "window_n":            int,
        "hr_mean",  "hr_var",  "hr_min",  "hr_max":           float,
        "hrv_rmssd_mean", ..., "hrv_rmssd_max":               float,
        "temp_mean", ..., "temp_max":                         float,
        "spo2_mean", ..., "spo2_max":                         float,
        "acc_mag_mean", ..., "acc_mag_max":                   float,
    },
}
```

Rule engine now has everything it needs for:
- **Absolute rules** → `sample.*`
- **Deviation rules** → `zscores.*` (gated on `baseline_ready`)
- **Short-term trend rules** → `window.*_mean / _var`
- **Quality gating** → `sample.accepted`, `sample.sqi.overall`

---

## 8. Constraints — how each is satisfied

| Constraint                | How                                                             |
|---------------------------|-----------------------------------------------------------------|
| < 700 ms latency          | O(1) per frame, pure Python, no I/O — typically < 1 ms          |
| Interpretable             | Every number is a mean / variance / z-score — inspectable       |
| Robust to noisy data      | Range gates + despiker + hold-last-good + SQI rejection         |
| Reliability over complexity | No ML, no DSP, no external deps beyond the stdlib              |

---

## 9. Deliverables (what's in this repo)

| File                         | Purpose                                        |
|------------------------------|------------------------------------------------|
| `preprocessing_pipeline.py`  | Self-contained, runnable reference pipeline    |
| `PIPELINE_DESIGN.md` (this)  | Stage-by-stage design rationale                |
| `app/services/preprocess.py` | Production version wired into the FastAPI app |
| `app/services/baseline.py`   | Production personal baseline                   |
| `app/services/rules.py`      | Rule engine consuming the contract in §7       |

**Run the demo**:

```bash
python3 preprocessing_pipeline.py
```

You'll see 8 synthetic frames streamed through the full pipeline with a
deliberate HR spike at frame 5 to demonstrate the despiker.

---

## Design decisions worth re-visiting

1. **HRV proxy vs. real IBI** — when firmware exposes inter-beat intervals,
   replace `_RRBuffer` and lift the `sqi.hrv ≤ 0.6` cap.
2. **EMA α values** — tuned on synthetic data. Tune on real WUALT captures.
3. **Baseline window = 300 s** — trade-off between "adapts to the user" and
   "doesn't adapt *out of* an anomaly". For sleep / long events you may want
   to freeze the baseline during elevated z-scores.
4. **Window size = 30 s** — if rule accuracy is limited by sample count,
   increase to 60 s; latency cost is zero (streaming window).
5. **`accel_mag` not EMA-filtered** — intentional, but if false-positive fall
   spikes are a problem, add a very light median-of-3 on `acc_mag`.
