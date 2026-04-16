# WUALT — Rule-Based Physiological Distress Detection Engine

## Complete Technical Documentation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why Rule-Based, Not Machine Learning](#2-why-rule-based-not-machine-learning)
3. [System Architecture](#3-system-architecture)
4. [Input Contract](#4-input-contract)
5. [Output Contract](#5-output-contract)
6. [The Seven-Stage Evaluation Pipeline](#6-the-seven-stage-evaluation-pipeline)
   - 6.1 [Stage 1: Quality Gate](#61-stage-1-quality-gate)
   - 6.2 [Stage 2: Motion Classification](#62-stage-2-motion-classification)
   - 6.3 [Stage 3: Signal Flagging](#63-stage-3-signal-flagging)
   - 6.4 [Stage 4: Motion-Aware Suppression](#64-stage-4-motion-aware-suppression)
   - 6.5 [Stage 5: Raw State Determination](#65-stage-5-raw-state-determination)
   - 6.6 [Stage 6: Persistence Enforcement](#66-stage-6-persistence-enforcement)
   - 6.7 [Stage 7: Confidence Scoring](#67-stage-7-confidence-scoring)
7. [The Three-State Model](#7-the-three-state-model)
8. [Signal Channels Deep Dive](#8-signal-channels-deep-dive)
   - 8.1 [Heart Rate (HR)](#81-heart-rate-hr)
   - 8.2 [Blood Oxygen Saturation (SpO2)](#82-blood-oxygen-saturation-spo2)
   - 8.3 [Heart Rate Variability (HRV RMSSD)](#83-heart-rate-variability-hrv-rmssd)
   - 8.4 [Skin Temperature](#84-skin-temperature)
9. [Dual-Threshold Architecture](#9-dual-threshold-architecture)
10. [Clinical Flag Integration](#10-clinical-flag-integration)
11. [Motion-Aware Exercise Suppression](#11-motion-aware-exercise-suppression)
12. [Persistence Tracker](#12-persistence-tracker)
13. [Emergency Bypass Logic](#13-emergency-bypass-logic)
14. [Confidence Scoring Model](#14-confidence-scoring-model)
15. [STAR-Principle Alert System](#15-star-principle-alert-system)
16. [Cold-Start Handling](#16-cold-start-handling)
17. [Synthetic Training Dataset](#17-synthetic-training-dataset)
18. [Validation Results](#18-validation-results)
19. [Known Limitations and Calibration Notes](#19-known-limitations-and-calibration-notes)
20. [File Reference](#20-file-reference)

---

## 1. Executive Summary

The WUALT Distress Detection Engine is a **standalone, stateful, rule-based system** that evaluates physiological signals from a wearable smart ring to detect stress and distress in real time.

| Property | Value |
|----------|-------|
| **Architecture** | Rule-based, deterministic, no ML |
| **Latency** | < 100 ms per evaluation (O(1) complexity) |
| **Input** | Preprocessed pipeline output (4 signals + metadata) |
| **Output** | State (normal/stress/distress) + confidence + alert |
| **Signals** | Heart Rate, SpO2, HRV (RMSSD), Skin Temperature |
| **Statefulness** | Persistence tracking, baseline warmup, alert rotation |
| **Dependencies** | Python stdlib only — zero external packages |
| **File** | `distress_engine.py` (991 lines, fully self-contained) |

The engine sits downstream of the preprocessing pipeline (`preprocessing_pipeline.py`), which handles raw sensor ingestion, EMA smoothing, Signal Quality Index (SQI) computation, clinical flag generation, and z-score baseline tracking. The distress engine consumes that processed output and makes a detection decision.

---

## 2. Why Rule-Based, Not Machine Learning

This is a deliberate architectural choice, not a limitation.

| Criterion | Rule-Based | ML-Based |
|-----------|-----------|----------|
| **Interpretability** | Every decision is fully traceable to a specific threshold and rule. You can explain *exactly* why a distress alert fired. | Black-box. Even with SHAP/LIME, explanations are approximate. |
| **Auditability** | A clinician or safety reviewer can read the code and verify every threshold against published literature. | Model weights are opaque. Requires extensive validation studies. |
| **Regulatory path** | Simpler regulatory story for a safety device — rules map directly to clinical guidelines (BTS/WHO SpO2 thresholds, Tanaka HR limits). | Requires statistical validation of model performance, drift monitoring, retraining infrastructure. |
| **Data requirements** | Works immediately with zero training data. Thresholds are set from physiology, calibrated with synthetic data. | Requires large, labelled, diverse real-world datasets that don't exist yet. |
| **Failure modes** | Predictable. If a threshold is wrong, you change one number. | Unpredictable. Model can fail in novel ways on unseen distributions. |
| **Edge cases** | Explicitly handled (exercise suppression, COPD baselines, pregnancy). | Must be represented in training data or the model won't learn them. |
| **Latency** | O(1) — constant time, trivially fast. | Depends on model size. Even simple models add overhead. |

**The key insight**: for a safety-critical wearable device in early stage, rule-based detection provides a reliable, auditable, and immediately deployable baseline. ML can be layered on top later for improved sensitivity, but the rule-based engine remains as the safety net.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WUALT Smart Ring                         │
│  (HR, PPG, SpO2, Accelerometer, Thermometer)               │
└──────────────────────┬──────────────────────────────────────┘
                       │ raw sensor frames (1 Hz)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           PREPROCESSING PIPELINE                            │
│  preprocessing_pipeline.py                                  │
│                                                             │
│  • EMA smoothing (α = 0.5)                                  │
│  • Signal Quality Index (SQI) — 7 sub-scores                │
│  • Thermal bias correction                                  │
│  • Dynamic acceleration extraction                          │
│  • Z-score baseline tracking (rolling 300-frame window)     │
│  • Clinical flag generation (advisory)                      │
│  • Frame acceptance/rejection                               │
└──────────────────────┬──────────────────────────────────────┘
                       │ pipeline_output dict
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           DISTRESS DETECTION ENGINE                         │
│  distress_engine.py                                         │
│                                                             │
│  Stage 1: Quality Gate                                      │
│  Stage 2: Motion Classification                             │
│  Stage 3: Signal Flagging (z-score + absolute + clinical)   │
│  Stage 4: Motion-Aware Suppression                          │
│  Stage 5: Raw State Determination                           │
│  Stage 6: Persistence Enforcement (60s sustained)           │
│  Stage 7: Confidence Scoring                                │
│       +   Alert Message Selection                           │
└──────────────────────┬──────────────────────────────────────┘
                       │ detection result
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           USER-FACING OUTPUT                                │
│                                                             │
│  • State: "normal" / "stress" / "distress"                  │
│  • Confidence: 0.0 – 1.0                                    │
│  • Alert: title + message + severity                        │
│  • Debug: flags, scores, persistence, motion                │
└─────────────────────────────────────────────────────────────┘
```

**Critical separation**: The distress engine has **zero imports** from the preprocessing pipeline. It consumes a well-defined dictionary contract. This means:
- Either component can be replaced independently
- The engine can be tested with synthetic data without running preprocessing
- Different preprocessing backends (different ring hardware) can feed the same engine

---

## 4. Input Contract

The engine's `evaluate()` method accepts exactly one argument — a dictionary with this structure:

```python
{
    "sample": {
        "timestamp":      int,       # Unix timestamp
        "sequence":       int,       # Frame sequence number
        "hr":             float,     # Heart rate (bpm), EMA-smoothed
        "hrv_rmssd":      float,     # HRV RMSSD (ms)
        "spo2":           float,     # Blood oxygen saturation (%)
        "temp":           float,     # Skin temperature (°C), bias-corrected
        "temp_raw":       float,     # Raw skin temperature before correction
        "dyn_acc_mag":    float,     # Dynamic acceleration magnitude (g)
        "acc_mag":        float,     # Total acceleration magnitude (g)
        "acc_x":          float,     # Accelerometer X axis (g)
        "acc_y":          float,     # Accelerometer Y axis (g)
        "acc_z":          float,     # Accelerometer Z axis (g)
        "finger_on":      bool,      # Finger-on-sensor detection
        "charging":       bool,      # Device on charger
        "battery_mv":     int,       # Battery voltage (mV)
        "die_temp":       float,     # Internal chip temperature (°C)
        "adc_raw":        int,       # Raw PPG ADC value
        "thermal_bias":   float,     # Computed thermal bias correction
        "sqi": {                     # Signal Quality Index sub-scores
            "hr":      float,        # 0.0–1.0
            "hrv":     float,
            "temp":    float,
            "acc":     float,
            "spo2":    float,
            "ppg":     float,
            "overall": float,        # Composite quality score
        },
        "accepted":       bool,      # True if frame passed quality checks
        "reject_reasons": [str],     # Why frame was rejected (if any)
        "clinical_flags": [str],     # Advisory flags (not rejections)
    },
    "zscores": {
        "hr":        float,          # Z-score: (current - baseline_mean) / baseline_std
        "hrv_rmssd": float,
        "spo2":      float,
        "temp":      float,
        "acc_mag":   float,
    },
    "baseline_ready": bool,          # True after sufficient warmup frames
    "window": {
        "window_n":       int,       # Number of frames in rolling window
        "hr_mean":        float,     # Rolling mean HR
        "hr_var":         float,     # Rolling variance
        "hr_min":         float,
        "hr_max":         float,
        "hrv_rmssd_mean": float,
        "hrv_rmssd_var":  float,
        "hrv_rmssd_min":  float,
        "hrv_rmssd_max":  float,
        "temp_mean":      float,
        "temp_var":       float,
        "temp_min":       float,
        "temp_max":       float,
        "spo2_mean":      float,
        "spo2_var":       float,
        "spo2_min":       float,
        "spo2_max":       float,
        "acc_mag_mean":   float,
        "acc_mag_var":    float,
        "acc_mag_min":    float,
        "acc_mag_max":    float,
    },
}
```

**Key fields the engine actually uses**:
- `sample.hr`, `sample.spo2`, `sample.dyn_acc_mag`, `sample.temp` — vital signs
- `sample.accepted` — gate check
- `sample.sqi.overall` — quality gate
- `sample.clinical_flags` — clinical advisory integration
- `zscores.hr`, `zscores.hrv_rmssd`, `zscores.spo2`, `zscores.temp` — personalized deviation
- `baseline_ready` — cold-start handling
- `window.window_n` — confidence scaling

---

## 5. Output Contract

Every call to `engine.evaluate()` returns:

```python
{
    "state":                "normal" | "stress" | "distress",
    "confidence":           float,       # 0.0 – 1.0
    "contributing_signals": ["hr", "spo2", ...],   # which signals triggered
    "alert": {
        "title":    str,                 # Short heading (e.g., "Heart rate is up")
        "message":  str,                 # Full user-facing message
        "severity": "low" | "medium" | "high",
    },
    "debug": {
        "flags": {
            "hr":        bool,
            "hrv_rmssd": bool,
            "spo2":      bool,
            "temp":      bool,
        },
        "scores": {
            "hr":        float,          # 0.0–1.0 severity score per signal
            "hrv_rmssd": float,
            "spo2":      float,
            "temp":      float,
        },
        "persistence_s": float,          # How long this condition has persisted
        "motion_state":  str,            # "still" / "walking" / "active" / "exercise"
        "flag_count":    int,            # Number of signals flagged
        "weighted_score": float,         # Combined weighted score
        "source":        str,            # "zscore" / "absolute" / "clinical_flag" / "none"
        "skipped":       bool,           # Present if frame was skipped
        "skip_reason":   str,            # "frame_rejected" / "low_sqi"
    },
}
```

---

## 6. The Seven-Stage Evaluation Pipeline

Each frame passes through seven sequential stages. The stages are strictly ordered — the output of one feeds the next.

### 6.1 Stage 1: Quality Gate

**Purpose**: Prevent the engine from making decisions on garbage data.

Two checks, in order:

1. **Frame acceptance**: If `sample.accepted == False`, the frame was rejected by the preprocessing pipeline (finger off, sensor failure, charging). The engine returns `state="normal"`, `confidence=0.0`, and sets `debug.skipped=True`, `debug.skip_reason="frame_rejected"`. The persistence tracker is also **reset** — we don't want a finger-off event to count as "stress duration."

2. **SQI threshold**: If `sample.sqi.overall < 0.5` (the `MIN_SQI_FOR_DETECTION` constant), the data quality is too low to trust. Same behavior — return normal with zero confidence, skip reason `"low_sqi"`, and reset persistence.

**Why reset persistence on gate failure**: If a user removes their ring for 30 seconds in the middle of a stress episode, we don't want the persistence counter to keep ticking. When they put it back on, the condition must re-establish itself from zero.

```python
# Configuration
MIN_SQI_FOR_DETECTION = 0.5
```

### 6.2 Stage 2: Motion Classification

**Purpose**: Determine whether the user is exercising, so elevated HR can be suppressed as expected behavior.

The engine classifies motion state from `sample.dyn_acc_mag` (dynamic acceleration magnitude — the component of acceleration beyond gravity, computed by the preprocessing pipeline):

| Motion State | dyn_acc_mag Range | Interpretation |
|-------------|-------------------|----------------|
| `"still"` | < 0.05 g | Sitting, lying, standing still |
| `"walking"` | 0.05 – 0.25 g | Light walking, casual movement |
| `"active"` | 0.25 – 0.50 g | Brisk walking, light sport |
| `"exercise"` | > 0.50 g | Running, HIIT, vigorous activity |

```python
MOTION_THRESHOLDS = {
    "still":    0.05,
    "walking":  0.25,
    "active":   0.50,
}
```

**Why dynamic acceleration, not total**: Total acceleration (`acc_mag`) includes gravity (~1.0 g). A still person has `acc_mag ≈ 0.98 g`, a runner might have `acc_mag ≈ 1.5 g`. Dynamic acceleration isolates the movement component by subtracting the gravitational baseline, making it a much cleaner motion indicator.

### 6.3 Stage 3: Signal Flagging

**Purpose**: Determine which of the four signal channels are currently in an abnormal state.

This is the core detection step. For each signal, three flag sources are checked in priority order:

#### Source 1: Z-Score Thresholds (Personalized)

Only active when `baseline_ready == True`. The preprocessing pipeline computes z-scores as:

```
z = (current_value - baseline_mean) / baseline_std
```

This is **personalized** — a person with resting HR of 55 bpm will have a very different z-score for 95 bpm than someone with resting HR of 85 bpm.

| Signal | Threshold | Direction | Meaning |
|--------|-----------|-----------|---------|
| HR | z > +2.0 | Elevated | Heart rate is 2+ standard deviations above personal baseline |
| HRV (RMSSD) | z < -1.5 | Suppressed | HRV has dropped 1.5+ standard deviations below baseline |
| SpO2 | z < -2.0 | Suppressed | Oxygen saturation is 2+ standard deviations below baseline |
| Temperature | z > +2.5 | Elevated | Skin temp is 2.5+ standard deviations above baseline |

**Direction matters**: HR and temperature flag when they go *up*. HRV and SpO2 flag when they go *down*. This reflects the physiology — stress causes HR to rise and HRV to fall.

**Score computation**: When a z-score flag fires, a severity score (0.0–1.0) is computed:

```python
# HR example (elevated direction)
hr_score = (z_hr - threshold) / 2.0 + 0.5

# HRV example (inverted direction)
hrv_score = abs(z_hrv - threshold) / 2.0 + 0.5

# Temperature (elevated, weak)
temp_score = (z_temp - threshold) / 3.0 + 0.3   # lower base, slower ramp
```

This means:
- A z-score exactly at threshold → score = 0.5
- A z-score 2.0 beyond threshold → score = 1.0 (capped)
- Temperature scores are deliberately dampened (0.3 base, divider of 3.0 instead of 2.0)

#### Source 2: Absolute Thresholds (Universal)

**Always active**, even during cold-start when no baseline exists. These are physiological limits that apply regardless of individual variation:

| Signal | Threshold | Level | Clinical Basis |
|--------|-----------|-------|----------------|
| HR | ≥ 120 bpm | High | Clearly elevated for any resting adult |
| HR | ≥ 150 bpm | Very High | Concerning even during exercise; triggers emergency bypass |
| SpO2 | ≤ 94% | Low | BTS/WHO clinical concern threshold |
| SpO2 | ≤ 90% | Very Low | Hypoxemia — medical emergency; triggers emergency bypass |
| Temp | ≥ 37.8°C (skin) | High | Possible fever (skin is 4–7°C below core temperature) |

**Score assignment for absolute thresholds**:
- `hr_very_high` (≥150): score = 0.9
- `hr_high` (≥120): score = 0.6
- `spo2_very_low` (≤90): score = 0.9
- `spo2_low` (≤94): score = 0.6
- `temp_high` (≥37.8): score = 0.3

Absolute thresholds use `max()` against any existing z-score — they never reduce a score, only elevate it.

#### Source 3: Clinical Flags (From Preprocessing)

The preprocessing pipeline generates clinical advisory flags that are **separate from reject reasons**. The engine integrates three:

| Clinical Flag | Engine Action | Score |
|--------------|---------------|-------|
| `spo2_hypoxemia (XX%)` | Flag SpO2 | 0.85 |
| `spo2_clinical_concern (XX%)` | Flag SpO2 | 0.55 |
| `elevated_skin_temp (XX°C)` | Flag temp | 0.3 |

Clinical flags only fire if the corresponding signal wasn't already flagged by z-score or absolute thresholds (they act as a fallback source).

#### Flag Priority

The `source` field in `SignalFlags` tracks which source triggered:
1. Z-score fires first (most personalized)
2. Absolute thresholds override if they produce a higher score
3. Clinical flags fill in if neither of the above triggered

### 6.4 Stage 4: Motion-Aware Suppression

**Purpose**: Prevent false stress alerts during exercise.

**The problem**: A person jogging at HR=145 bpm would trigger both z-score (likely z > +10) and absolute (≥120 bpm) thresholds. But this is completely normal during exercise.

**The rule**: If motion state is `"active"` or `"exercise"`, **and** only HR is flagged (not SpO2), **and** HR is below the emergency threshold (150 bpm), then:
- `flags.hr = False`
- `flags.hr_score = 0.0`

The HR flag is completely cleared. The user sees a normal state.

**The critical exception**: If SpO2 is ALSO flagged during exercise, HR suppression does NOT occur. A runner with HR=145 and SpO2=91% is potentially in real danger — the exercise doesn't explain the oxygen drop. Both signals remain flagged, and the state escalates to distress.

```python
if motion_state in ("active", "exercise"):
    if flags.hr and not flags.spo2 and hr < 150:
        flags.hr = False
        flags.hr_score = 0.0
```

**Why not suppress HRV during exercise?**: HRV naturally drops during exercise. But unlike HR (which is expected to rise), HRV suppression during exercise can still indicate overtraining or cardiac distress. We let HRV flags stand during exercise as a secondary indicator.

### 6.5 Stage 5: Raw State Determination

**Purpose**: Map the flag count to one of three states.

The logic is simple and deliberate:

```
flags.count >= 2  →  "distress"    (multiple systems affected)
flags.count == 1  →  "stress"     (single signal deviation)
flags.count == 0  →  "normal"     (all clear)
```

**Temperature exception**: If the *only* flag is temperature (`flags.temp == True` and `flags.count == 1`), the state remains `"normal"`. Temperature alone is too unreliable — ambient heat, post-meal warmth, clothing can all cause skin temp elevation. Temperature only contributes when it accompanies another primary signal.

This produces a **raw state** — the final state may differ after persistence enforcement.

### 6.6 Stage 6: Persistence Enforcement

**Purpose**: Prevent single-frame noise from triggering alerts. A condition must persist for a sustained duration before the engine promotes the state.

#### Configuration

```python
PERSISTENCE_STRESS_S   = 60    # 60 seconds of sustained stress before alerting
PERSISTENCE_DISTRESS_S = 60    # 60 seconds of sustained distress before alerting
```

#### How It Works

The `PersistenceTracker` class maintains two independent timers:
- **Stress timer**: starts when `raw_state in ("stress", "distress")`, resets when the condition clears
- **Distress timer**: starts when `raw_state == "distress"`, resets when the condition clears

**Grace period**: The tracker has a 2-frame grace window. If the condition drops for 1–2 frames (sensor noise, brief artifact), the timer does NOT reset. It only resets after 3+ consecutive clear frames. This prevents jitter at the boundary from constantly restarting the timer.

```python
class PersistenceTracker:
    def __init__(self, grace_frames=2):
        # Stress and distress have independent timers
        # Grace period of 2 frames absorbs sensor noise
```

#### State Promotion Logic

```
If raw_state == "distress":
    If distress_duration >= 60s → final_state = "distress"
    Elif stress_duration >= 60s → final_state = "stress"     (demoted)
    Else                        → final_state = "normal"     (building up)

If raw_state == "stress":
    If stress_duration >= 60s   → final_state = "stress"
    Else                        → final_state = "normal"     (building up)
```

This means a new stress episode takes 60 seconds to become visible to the user. During that time, the engine returns `"normal"` but the `debug.persistence_s` field shows the build-up progress.

**Why 60 seconds**: Physiological stress responses are sustained phenomena. A single-frame HR spike from coughing, sneezing, or a startle reflex should not trigger an alert. 60 seconds filters out transient events while still being responsive enough for genuine episodes.

#### Emergency Bypass

Persistence is **completely bypassed** for emergency signals (see Section 13).

### 6.7 Stage 7: Confidence Scoring

**Purpose**: Quantify how confident the engine is in its current detection.

The confidence model differs by state:

#### Normal State

```
confidence = sqi_overall × 0.6 + window_factor × 0.4
```

Where `window_factor = min(1.0, window_n / 20.0)`. This means:
- High SQI + full window → confidence ≈ 0.93 (very confident it's normal)
- Low SQI or sparse window → lower confidence in normality
- After 20 frames (~20 seconds at 1 Hz), window factor is maxed

#### Stress / Distress State

```
confidence = signal_score × 0.5 + persist_factor × 0.3 + sqi_factor × 0.2
```

Components:
1. **Signal score** (50%): The weighted sum of flagged signal scores (Section 8 weights)
2. **Persistence factor** (30%): Ramps from 0.3 at onset to 1.0 after 120 seconds
3. **SQI factor** (20%): `min(1.0, sqi_overall / 0.8)` — higher quality data → higher confidence

```python
persist_factor = min(1.0, 0.3 + (persistence_s / 120.0) * 0.7)
sqi_factor     = min(1.0, sqi_overall / 0.8)
```

**Why persistence affects confidence**: A stress signal that has persisted for 2 minutes is much more reliable than one that just started (might be noise). The 120-second ramp reflects this.

---

## 7. The Three-State Model

```
                                    SpO2 < 90% (bypass)
                                    HR > 150 at rest (bypass)
                                           │
    ┌──────────┐    1 signal     ┌──────────▼──┐    2+ signals    ┌────────────┐
    │          │    sustained    │             │    sustained     │            │
    │  NORMAL  │───────────────▶│   STRESS    │────────────────▶│  DISTRESS  │
    │          │    60 sec      │             │    60 sec       │            │
    └────▲─────┘                └──────┬──────┘                 └──────┬─────┘
         │                             │                               │
         │         signals clear       │         signals clear         │
         └─────────────────────────────┘───────────────────────────────┘
```

### Normal
- **Definition**: Zero primary signals flagged, OR only temperature flagged alone
- **Severity**: Low
- **Action**: No intervention needed
- **Confidence meaning**: "How sure we are that everything is fine"

### Stress
- **Definition**: Exactly one primary signal (HR, SpO2, or HRV) deviating beyond threshold
- **Severity**: Low to Medium (SpO2-related stress is Medium)
- **Action**: Advisory — suggest breathing, relaxation
- **Escalation**: If a second signal joins, promotes to distress

### Distress
- **Definition**: Two or more primary signals simultaneously deviating
- **Severity**: High
- **Action**: Urgent — suggest reaching out for help, contacting someone
- **Emergency**: SpO2 < 90% or HR > 150 at rest bypass persistence and immediately escalate

**Why 2+ signals for distress**: A single elevated signal could have many benign explanations (coffee for HR, altitude for SpO2, fatigue for HRV). When multiple physiological systems are simultaneously disrupted, the probability of genuine distress is much higher.

---

## 8. Signal Channels Deep Dive

### 8.1 Heart Rate (HR)

| Property | Value |
|----------|-------|
| **Weight** | 0.35 (highest) |
| **Direction** | Elevated = stress |
| **Z-score threshold** | z > +2.0 |
| **Absolute thresholds** | ≥120 bpm (high), ≥150 bpm (emergency) |
| **Exercise suppression** | Yes — HR flag cleared during exercise unless SpO2 also drops |

**Physiological basis**: Heart rate is the most responsive and reliable indicator of acute stress. Sympathetic nervous system activation directly elevates HR within seconds. The 0.35 weight reflects its primacy as a stress indicator.

**Score formula**:
- Z-score source: `score = (z - 2.0) / 2.0 + 0.5` → range [0.5, 1.0]
- Absolute source: 0.6 for ≥120 bpm, 0.9 for ≥150 bpm

### 8.2 Blood Oxygen Saturation (SpO2)

| Property | Value |
|----------|-------|
| **Weight** | 0.30 (second highest) |
| **Direction** | Suppressed = distress |
| **Z-score threshold** | z < -2.0 |
| **Absolute thresholds** | ≤94% (clinical concern, BTS/WHO), ≤90% (hypoxemia, emergency) |
| **Exercise suppression** | Never — SpO2 drop during exercise is always concerning |
| **Emergency bypass** | SpO2 ≤ 90% bypasses persistence entirely |

**Physiological basis**: Blood oxygen saturation is a critical safety signal. Normal SpO2 is 95–100%. Below 94% indicates clinical concern per British Thoracic Society and WHO guidelines. Below 90% is hypoxemia — a medical emergency requiring immediate action.

**Why 0.30 weight, not higher**: SpO2 is slightly less responsive than HR to psychological stress. Its primary role is detecting respiratory distress, hypoxia, and medical emergencies rather than emotional stress. However, when SpO2 drops, it's almost always clinically significant.

**Score formula**:
- Z-score source: `score = abs(z - (-2.0)) / 2.0 + 0.5`
- Absolute source: 0.6 for ≤94%, 0.9 for ≤90%
- Clinical flag source: 0.55 for concern, 0.85 for hypoxemia

### 8.3 Heart Rate Variability (HRV RMSSD)

| Property | Value |
|----------|-------|
| **Weight** | 0.25 |
| **Direction** | Suppressed = stress (inverted) |
| **Z-score threshold** | z < -1.5 |
| **Absolute threshold** | None (HRV is too individual for universal limits) |
| **Exercise suppression** | Not suppressed — low HRV during exercise can indicate overtraining |

**Physiological basis**: HRV (specifically RMSSD — Root Mean Square of Successive Differences) reflects parasympathetic nervous system tone. High HRV indicates healthy autonomic function and relaxation. Suppressed HRV indicates sympathetic dominance — the stress response.

**Why no absolute threshold**: Unlike HR and SpO2, HRV varies enormously between individuals. A healthy athlete might have resting HRV of 65 ms while a sedentary older adult has 18 ms. There is no universal "bad HRV" number. Only personalized z-scores work for HRV.

**Why lower z-score threshold (-1.5 vs -2.0)**: HRV responds to chronic stress, not just acute events. A more sensitive threshold (-1.5) catches subtle ANS dysregulation that a -2.0 threshold would miss.

**Score formula**:
- Z-score source: `score = abs(z - (-1.5)) / 2.0 + 0.5`

### 8.4 Skin Temperature

| Property | Value |
|----------|-------|
| **Weight** | 0.10 (weakest — deliberately) |
| **Direction** | Elevated = supporting signal |
| **Z-score threshold** | z > +2.5 (highest threshold) |
| **Absolute threshold** | ≥ 37.8°C skin |
| **Special rule** | Temperature ALONE never triggers stress |

**Physiological basis**: The ring measures **skin** temperature, which is 4–7°C below core temperature. Skin temp is affected by ambient temperature, clothing, post-meal vasodilation, physical activity, and many non-stress factors. It's an unreliable standalone indicator.

**Why keep it at all**: When combined with other signals, elevated skin temp supports the detection. For example: HR elevated + skin temp elevated → more likely genuine fever/infection than just anxiety. Temperature as a supporting signal increases confidence without generating false positives.

**Why the highest z-score threshold (2.5)**: Because skin temp varies so much with environment, we require a very large deviation before considering it meaningful at all.

**Why temp alone ≠ stress**: This is an explicit rule in the state determination:
```python
if flags.temp and flags.count == 1:
    raw_state = "normal"    # temp alone is advisory only
```

**Score formula**:
- Z-score source: `score = (z - 2.5) / 3.0 + 0.3` → range [0.3, ~0.6] — deliberately dampened
- Absolute source: 0.3

---

## 9. Dual-Threshold Architecture

The engine uses two complementary threshold systems:

### Z-Score Thresholds (Personalized)

```python
ZSCORE_THRESHOLDS = {
    "hr":        2.0,     # ↑ elevated
    "hrv_rmssd": -1.5,    # ↓ suppressed
    "spo2":      -2.0,    # ↓ desaturation
    "temp":      2.5,     # ↑ elevated
}
```

**Advantages**: Personalized to the individual. An athlete with resting HR 48 bpm who suddenly goes to 80 bpm (z ≈ +8) is flagged even though 80 bpm is "normal" for most people. Conversely, a sedentary person with resting HR 88 bpm at 95 bpm (z ≈ +1.2) is NOT flagged — it's within their normal range.

**Requirement**: Only active after `baseline_ready == True` (the preprocessing pipeline has accumulated enough frames to compute stable mean and standard deviation).

### Absolute Thresholds (Universal)

```python
ABSOLUTE_THRESHOLDS = {
    "hr_high":      120.0,    # bpm
    "hr_very_high": 150.0,    # bpm — emergency
    "spo2_low":      94.0,    # % — BTS/WHO
    "spo2_very_low": 90.0,    # % — hypoxemia, emergency
    "temp_high":     37.8,    # °C skin
}
```

**Advantages**: Always active, even during cold-start. No matter who the user is, HR ≥ 150 at rest or SpO2 ≤ 90% is dangerous. These are the **safety net** — they catch emergencies even before the baseline is established.

**Clinical sources**:
- HR 120 bpm: widely accepted clinical threshold for resting tachycardia
- HR 150 bpm: emergency threshold based on Tanaka 2001 MPHR formula margins
- SpO2 94%: British Thoracic Society (BTS) guideline for clinical concern
- SpO2 90%: WHO definition of hypoxemia
- Skin temp 37.8°C: corresponds to approximately 42–45°C core temperature range, indicating possible fever

---

## 10. Clinical Flag Integration

The preprocessing pipeline generates **clinical advisory flags** that are separate from frame rejection reasons. This is a critical architectural distinction:

| Concept | Reject Reasons | Clinical Flags |
|---------|---------------|----------------|
| **Purpose** | Data quality failure | Health advisory |
| **Effect on frame** | `accepted = False` — frame is discarded | `accepted` unchanged — frame is still analyzed |
| **Examples** | `finger_off`, `low_ppg_variance`, `hr_out_of_range` | `spo2_clinical_concern (93%)`, `elevated_skin_temp (37.9°C)`, `spo2_hypoxemia (88%)` |
| **Engine behavior** | Frame skipped at gate | Flags integrated into signal flagging |

**Why this matters**: A person with a fever (skin temp 37.9°C) and clean sensor data should still have their data analyzed. If we treated clinical concerns as quality rejections, we'd stop monitoring precisely when monitoring is most important.

The distress engine checks three clinical flags:

1. **`spo2_hypoxemia (XX%)`**: SpO2 ≤ 90% — flags SpO2 with score 0.85
2. **`spo2_clinical_concern (XX%)`**: SpO2 ≤ 94% — flags SpO2 with score 0.55
3. **`elevated_skin_temp (XX°C)`**: Skin temp ≥ 37.5°C — flags temp with score 0.3

Clinical flags only apply when the signal wasn't already flagged by z-score or absolute thresholds (they don't double-flag).

---

## 11. Motion-Aware Exercise Suppression

This is one of the most important features of the engine. Without it, every exercise session would generate continuous stress/distress alerts.

### The Problem

A person running has:
- HR = 145 bpm → triggers z-score (z ≈ +15) AND absolute (≥120) thresholds
- HRV = 12 ms → triggers z-score (z ≈ -4)
- That's 2 signals → raw state = "distress"

But this is perfectly normal during exercise. The engine must distinguish between:
- **Expected exercise response**: elevated HR, suppressed HRV, high motion
- **Real distress during exercise**: elevated HR, suppressed HRV, high motion, AND dropping SpO2

### The Rules

```
IF motion is "active" or "exercise":
    IF HR is flagged AND SpO2 is NOT flagged AND HR < 150 bpm:
        CLEAR HR flag (set to False, score to 0.0)
    
    IF SpO2 IS ALSO flagged:
        DO NOT suppress anything — this could be real danger
    
    IF HR ≥ 150 bpm:
        DO NOT suppress — even for exercise, this is concerning
```

### Examples

| Scenario | HR | SpO2 | Motion | HR Flag | Result |
|----------|-----|------|--------|---------|--------|
| Normal jog | 140 | 97% | exercise | Cleared | Normal |
| Sprint | 165 | 97% | exercise | Kept (≥150) | Stress |
| Jog + altitude sickness | 140 | 91% | exercise | Kept (SpO2 also flagged) | Distress |
| Resting stress | 120 | 97% | still | Kept | Stress |

---

## 12. Persistence Tracker

### Purpose

A single-frame anomaly should not generate an alert. Sensor noise, a cough, a sudden movement — all can produce momentary spikes. The persistence tracker ensures conditions must be **sustained** before the engine escalates its state.

### Mechanism

```python
class PersistenceTracker:
    grace_frames = 2        # frames of tolerance before resetting
    stress_start  = None    # timestamp when stress condition began
    distress_start = None   # timestamp when distress condition began
```

**On each frame**:
1. If condition is active → record start time (if not already started), reset grace counter
2. If condition clears → increment grace counter
3. If grace counter exceeds 2 → reset timer (condition truly ended)
4. Return duration = `now - start_time`

**The grace period prevents**:
- A 1-frame sensor dropout from restarting a 58-second stress timer
- Boundary jitter (values oscillating around threshold) from never reaching 60 seconds

### Timer Independence

Stress and distress have **separate, independent timers**. This allows:
- Distress timer can start while stress timer is already running
- If distress duration isn't met but stress duration is, state can be "stress" (demotion)
- Both timers reset independently when their respective conditions clear

### Reset Conditions

The persistence tracker fully resets on:
- Frame rejection (finger off, low SQI)
- This prevents phantom persistence from building up during gaps in data

---

## 13. Emergency Bypass Logic

Certain conditions are so dangerous that waiting 60 seconds is unacceptable. The engine has two emergency bypass rules that skip persistence entirely:

### SpO2 ≤ 90% → Immediate Distress

```python
if spo2 <= 90.0:
    final_state = "distress"
    persistence_s = max(persistence_s, 1.0)
```

**Rationale**: SpO2 below 90% is hypoxemia by WHO definition. Every second matters. Waiting 60 seconds to confirm could be dangerous.

### HR ≥ 150 bpm at Rest → Immediate Stress (minimum)

```python
if hr >= 150.0 and motion not in ("active", "exercise"):
    final_state = max(final_state, "stress")
    persistence_s = max(persistence_s, 1.0)
```

**Rationale**: HR above 150 bpm while sitting/standing still is abnormal tachycardia. However, note this promotes to "stress" at minimum — it doesn't force "distress" by itself because a single signal (even an extreme one) follows the single-signal-to-stress rule. If SpO2 is also dropping, the multi-signal rule promotes it to distress.

**Critical**: The HR emergency bypass does NOT fire during exercise. HR 160 bpm while sprinting is normal; HR 160 bpm while sitting is not.

---

## 14. Confidence Scoring Model

Confidence is a 0.0–1.0 score that represents how certain the engine is about its current detection.

### For Normal State

```
confidence = SQI × 0.6 + window_factor × 0.4
```

- **SQI (60%)**: Higher signal quality → more confident that normal is truly normal
- **Window factor (40%)**: `min(1.0, window_n / 20)` — after ~20 frames of consistent data, window factor maxes out

Typical normal confidence: **0.88–0.93**

### For Stress / Distress States

```
confidence = signal_score × 0.5 + persist_factor × 0.3 + sqi_factor × 0.2
```

- **Signal score (50%)**: Weighted sum of flagged signals (using the 0.35/0.30/0.25/0.10 weights)
- **Persist factor (30%)**: Ramps from 0.3 → 1.0 over 120 seconds — longer conditions are more reliable
- **SQI factor (20%)**: `min(1.0, sqi / 0.8)` — poor data quality reduces detection confidence

**Example calculations**:

| Scenario | Signal Score | Persistence | SQI | Confidence |
|----------|-------------|-------------|-----|-----------|
| HR only, just started | 0.35 × 0.6 = 0.21 | 0.3 | 1.0 | 0.21×0.5 + 0.3×0.3 + 1.0×0.2 = **0.40** |
| HR + SpO2, 90 seconds | 0.35×0.8 + 0.30×0.7 = 0.49 | 0.83 | 1.0 | 0.49×0.5 + 0.83×0.3 + 1.0×0.2 = **0.69** |
| HR + SpO2, 120+ seconds | 0.35×0.9 + 0.30×0.9 = 0.59 | 1.0 | 1.0 | 0.59×0.5 + 1.0×0.3 + 1.0×0.2 = **0.80** |
| Emergency SpO2=88% | 0.35×0.5 + 0.30×0.9 = 0.45 | 0.3 | 1.0 | 0.45×0.5 + 0.3×0.3 + 1.0×0.2 = **0.52** |

---

## 15. STAR-Principle Alert System

All user-facing alerts follow the STAR principle (Situation, Task, Action, Result) internally, but the message is natural and conversational — no medical jargon, no alarming language.

### Alert Categories

| Category | Severity | When | Example Message |
|----------|----------|------|-----------------|
| **NORMAL** | Low | All clear | "Everything looks steady. You're doing great." |
| **WARMING UP** | Low | Baseline not ready | "We're still learning your baseline. Alerts will become more personalized shortly." |
| **TEMP ELEVATED** | Low | Temp only (advisory) | "Your skin temperature is a bit warmer than usual, but there's no need to worry right now." |
| **STRESS (HR)** | Low | HR is the primary flag | "Your heart rate is a bit higher than usual. If you're not exercising, try to relax and take it easy." |
| **STRESS (SpO2)** | Medium | SpO2 is the primary flag | "Your oxygen level dipped a little. Try some slow, deep breaths — it usually helps." |
| **STRESS (General)** | Low | Multiple signals but only 1 primary | "Your body seems a bit stressed right now. Try taking a few slow, deep breaths." |
| **EXERCISE** | Low | HR elevated during exercise | "Your readings are elevated, but you seem to be moving — this is likely from physical activity." |
| **DISTRESS (General)** | High | 2+ signals | "We're noticing signs of distress. If you're not feeling safe, please reach out to someone you trust." |
| **DISTRESS (SpO2)** | High | SpO2 score ≥ 0.8 | "Your oxygen level is lower than expected. If you're feeling lightheaded or short of breath, please seek help." |

### Alert Selection Priority

```
1. Baseline not ready + normal        → WARMING UP
2. Normal + temp only elevated        → TEMP ELEVATED
3. Normal                             → NORMAL
4. Exercise + HR only (no SpO2)       → EXERCISE
5. Stress + SpO2 dominant             → STRESS (SpO2)
6. Stress + HR dominant               → STRESS (HR)
7. Stress (mixed)                     → STRESS (General)
8. Distress + SpO2 score ≥ 0.8       → DISTRESS (SpO2)
9. Distress (any)                     → DISTRESS (General)
```

### Message Rotation

Several categories have multiple message variants (stress has 3, distress has 3). The engine cycles through them using a deterministic counter (`cycle % len(pool)`) to avoid showing the same message repeatedly during sustained conditions.

### Design Principles

- **No medical jargon**: "oxygen level" not "SpO2 saturation", "heart rate" not "tachycardia"
- **No alarming language**: "seems a bit stressed" not "WARNING: abnormal vitals detected"
- **Actionable suggestions**: "Try some slow, deep breaths", "Take a moment", "Reach out to someone you trust"
- **Calm tone**: Even high-severity alerts are empathetic and supportive, not clinical
- **No diagnosis**: "signs of distress" not "you may be having a panic attack"

---

## 16. Cold-Start Handling

When the device is first put on, or after a long gap, the preprocessing pipeline needs time to establish a stable baseline. During this period:

- `baseline_ready = False`
- Z-scores are all 0.0 (no baseline to compare against)

**Engine behavior during cold-start**:
1. Z-score thresholds are **disabled** — only absolute thresholds are active
2. This means only extreme deviations (HR ≥ 120, SpO2 ≤ 94, etc.) are caught
3. The alert message says: "We're still learning your baseline. Alerts will become more personalized shortly."
4. Confidence is lower because the window factor is small

**Why not skip detection entirely during cold-start**: Safety. If someone puts on the ring and immediately has SpO2 at 88%, we need to catch that even without a baseline. Absolute thresholds serve as the universal safety net.

**Typical warmup duration**: 30–60 seconds at 1 Hz sampling. The preprocessing pipeline requires approximately 20–30 frames to compute stable rolling statistics.

---

## 17. Synthetic Training Dataset

The engine ships with a comprehensive synthetic dataset generator (`generate_synthetic_dataset.py`) that produces physiologically realistic labelled data for threshold calibration and validation.

### 15 User Profiles

| ID | Description | HR | HRV | SpO2 | Temp | Notes |
|----|-------------|-----|-----|------|------|-------|
| H01 | Healthy young male, 25M | 72 | 35 | 98.2 | 36.5 | |
| H02 | Healthy young female, 28F | 76 | 32 | 98.0 | 36.6 | |
| H03 | Athletic male, 22M | 58 | 52 | 98.8 | 36.3 | |
| H04 | Athletic female, 24F | 62 | 48 | 98.6 | 36.4 | |
| H05 | Sedentary male, 35M | 82 | 22 | 97.5 | 36.7 | |
| H06 | Sedentary female, 40F | 80 | 24 | 97.8 | 36.8 | |
| H07 | Older male, 60M | 70 | 20 | 97.0 | 36.4 | |
| H08 | Older female, 65F | 74 | 18 | 96.8 | 36.5 | |
| H09 | Teen male, 16M | 78 | 38 | 98.5 | 36.5 | |
| H10 | Teen female, 17F | 80 | 36 | 98.3 | 36.6 | |
| S01 | Anxiety disorder, 30F | 85 | 20 | 97.8 | 36.7 | Naturally elevated HR |
| S02 | COPD patient, 55M | 78 | 19 | 94.5 | 36.6 | Chronically low SpO2 |
| S03 | Obese/hypertensive, 45M | 88 | 16 | 97.0 | 37.0 | Higher baseline HR/temp |
| S04 | Pregnant (32wk), 29F | 90 | 22 | 97.5 | 36.9 | Elevated resting HR |
| S05 | Ultra-fit athlete, 26M | 48 | 65 | 99.0 | 36.2 | Very low resting HR |

### 37 Scenario Templates

**Normal (7)**: resting calm, coffee/standup, sleeping, light walking, deep breathing, post-meal, warm environment

**Exercise (4)**: moderate exercise, vigorous exercise, warmup, cooldown

**Stress (6)**: mild HR elevation, moderate HR, absolute HR threshold, SpO2 dip, HRV suppression, panic onset

**Distress (8)**: HR+SpO2, HR+HRV, SpO2+HRV, triple signal, fever combo, SpO2 critical emergency, HR extreme emergency, combined emergency

**Edge Cases (10)**: finger off, charging, low SQI, transient spike, temp only elevated, exercise+SpO2 drop, COPD baseline, anxiety baseline, recovery after distress, cold start

**Transitions (3)**: normal→stress, stress→distress, distress→normal

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total frames | 9,926 |
| Scenario instances | 2,121 |
| Columns per frame | 53 |
| Normal frames | 5,401 (54.4%) |
| Stress frames | 2,128 (21.4%) |
| Distress frames | 2,188 (22.0%) |
| Rejected frames | 209 (2.1%) |

---

## 18. Validation Results

Running every frame in the synthetic dataset through the engine (with persistence disabled for per-frame evaluation) produces:

### Overall Accuracy: 65.8%

### Per-Class Metrics

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal | 94.6% | 60.9% | 74.1% |
| Stress | 49.0% | 45.9% | 47.4% |
| Distress | 48.3% | 93.9% | 63.8% |
| Rejected | 100.0% | 100.0% | 100.0% |

### Interpretation

**What works well**:
- **Distress recall is 93.9%** — the engine catches almost all dangerous conditions. This is the most important metric for a safety device.
- **Normal precision is 94.6%** — when the engine says "normal," it's right 95% of the time.
- **Rejected handling is perfect** — 100% on both axes.

**Where calibration is needed**:
- **Exercise scenarios**: Vigorous exercise (HR~140-160, very high motion) sometimes falls below the 0.50g exercise threshold due to noise, causing false stress alerts. The motion threshold needs tuning.
- **Stress boundary**: Some single-signal stress scenarios (HR+HRV) produce 2 flags because HRV naturally drops with elevated HR, pushing into distress. This is an inherent coupling in the physiology — may need HRV-HR correlation adjustment.
- **Post-meal / slight HR elevation**: Small HR bumps from coffee or standing can push over z-score thresholds for some profiles (especially the athlete with baseline 48 bpm).

**This is by design** — the synthetic dataset exists precisely to reveal these calibration gaps. Adjusting thresholds and re-running validation is the calibration workflow.

---

## 19. Known Limitations and Calibration Notes

### Limitations

1. **Skin vs core temperature**: The ring measures skin temperature (4–7°C below core). This makes fever detection inherently imprecise. Temperature is kept as a weak supporting signal for this reason.

2. **HRV quality from wrist/ring PPG**: Ring-based HRV is less accurate than chest-strap ECG. The preprocessing pipeline caps HRV SQI accordingly (Esco & Flatt 2014, cap at 0.7), and the engine treats HRV as a secondary indicator.

3. **Motion artifacts**: During vigorous exercise, all signals become noisier. The SQI gate catches severe artifacts, but borderline-quality data during exercise is a known challenge.

4. **Individual variation not fully captured**: While z-scores personalize detection, the absolute thresholds are population-wide. A COPD patient with baseline SpO2 of 92% would trigger absolute thresholds constantly. The preprocessing pipeline's baseline should ideally shift their z-score center, but the 94% absolute threshold remains.

5. **No circadian adjustment**: Heart rate and temperature have natural circadian rhythms (lower during sleep, higher in afternoon). The current engine doesn't account for time-of-day effects.

6. **Single-ring limitation**: The engine only has access to one measurement site (finger). Cross-body comparisons (e.g., bilateral SpO2) are not possible.

### Calibration Levers

| Parameter | Current Value | Effect of Increasing | Effect of Decreasing |
|-----------|---------------|---------------------|---------------------|
| `ZSCORE_THRESHOLDS["hr"]` | 2.0 | Fewer HR stress detections (more specific) | More HR stress detections (more sensitive) |
| `ZSCORE_THRESHOLDS["hrv_rmssd"]` | -1.5 | More HRV stress detections | Fewer HRV stress detections |
| `ZSCORE_THRESHOLDS["spo2"]` | -2.0 | More SpO2 detections | Fewer SpO2 detections |
| `ZSCORE_THRESHOLDS["temp"]` | 2.5 | Fewer temp flags | More temp flags |
| `ABSOLUTE_THRESHOLDS["hr_high"]` | 120 | Fewer absolute HR flags | More absolute HR flags |
| `ABSOLUTE_THRESHOLDS["spo2_low"]` | 94 | Fewer SpO2 flags | More SpO2 flags |
| `PERSISTENCE_STRESS_S` | 60s | Slower to alert (fewer false positives) | Faster to alert (more responsive) |
| `PERSISTENCE_DISTRESS_S` | 60s | Slower to escalate | Faster to escalate |
| `MIN_SQI_FOR_DETECTION` | 0.5 | More frames skipped (stricter quality) | More frames evaluated (more permissive) |
| `MOTION_THRESHOLDS["active"]` | 0.50 | More exercise classified as "active" | More movement classified as "exercise" |
| `SIGNAL_WEIGHTS["hr"]` | 0.35 | HR has more influence on confidence | HR has less influence |
| `SIGNAL_WEIGHTS["temp"]` | 0.10 | Temp has more influence | Temp has less influence |

### Recommended Calibration Workflow

1. Run `python generate_synthetic_dataset.py` to produce labelled data
2. Examine the per-scenario accuracy breakdown
3. Identify scenarios with poor accuracy
4. Adjust the relevant thresholds in `distress_engine.py`
5. Re-run validation to see the impact
6. Repeat until the desired sensitivity/specificity balance is achieved
7. When real ring data is available, replace synthetic profiles with actual user baselines

---

## 20. File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `distress_engine.py` | 991 | The complete standalone distress detection engine |
| `generate_synthetic_dataset.py` | ~650 | Generates 9,926 labelled frames + runs validation |
| `synthetic_dataset.csv` | 9,927 | Flat CSV with 53 columns — every frame labelled |
| `synthetic_scenarios.csv` | 2,122 | Scenario-level summaries with mean vitals |
| `synthetic_dataset_summary.json` | — | Full metadata, profiles, validation results |
| `preprocessing_pipeline.py` | — | Upstream: raw sensor → processed output (separate system) |

### How to Run

```bash
# Run the 13-scenario demo
python distress_engine.py

# Generate synthetic dataset and validate
python generate_synthetic_dataset.py

# Use in your code
from distress_engine import DistressEngine
engine = DistressEngine()
result = engine.evaluate(pipeline_output)
print(result["state"], result["alert"]["message"])
```

---

*Document version: 1.0 — April 2026*
*Engine version: distress_engine.py commit 854167a*
