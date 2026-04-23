"""
WUALT — Hardware → Rule-Engine Preprocessing Pipeline
======================================================

A single, self-contained, readable file that takes a raw `latest_sensor_data`
frame from the WUALT ring firmware and turns it into a clean, normalized,
SQI-scored, windowed, z-scored dict that is directly feedable into the
rule-based anomaly detection engine.

Design goals:
    - Real-time:    < 700 ms per frame (trivially satisfied, O(1) per frame)
    - Interpretable: no black-box models, every number is traceable
    - Robust:       range-gated, despiked, finger-off aware, cold-start safe
    - Lightweight:  pure Python, no numpy/scipy/pandas dependency

Run it directly to see a demo:

    python preprocessing_pipeline.py

---------------------------------------------------------------------------
INPUT  (exactly what your firmware already emits)
---------------------------------------------------------------------------
    {
        "accel_x": float,   "accel_y": float,   "accel_z": float,  # g
        "adc_raw": int,                                             # raw PPG
        "heart_rate": float,                                        # bpm
        "spo2": float,                                              # %
        "temperature": float,                                       # °C
        "vbat_mv": int,                                             # mV
        "die_temp": float,                                          # °C
        "charger_stat": str,                                        # "idle"/"charging"/...
        "vbus_present": bool,
        "finger_on": bool,
        "sequence": int,
        "status": str,
    }

---------------------------------------------------------------------------
OUTPUT (ready for the rule engine)
---------------------------------------------------------------------------
    {
        "timestamp":  int,            # seconds since epoch
        "sequence":   int,            # copied from device
        "hr":         float | None,   # bpm, despiked + EMA-smoothed
        "hr_stability_score":  float | None,   # ms, HR-stability proxy (NOT clinical HRV)
        "temp":       float | None,   # °C skin
        "spo2":       float | None,   # %
        "acc_mag":    float,          # g, sqrt(ax² + ay² + az²)
        "acc_x":      float,
        "acc_y":      float,
        "acc_z":      float,
        "finger_on":  bool,
        "sqi": {                      # per-signal quality 0..1
            "hr":   float,
            "hrv":  float,
            "temp": float,
            "acc":  float,
            "spo2": float,
            "overall": float,
        },
        "accepted":       bool,       # False => rule engine should skip / warm-up
        "reject_reasons": list[str],  # quality failures (sensor/contact/battery)
        "clinical_flags": list[str],  # advisory (elevated skin temp, low SpO2…)
    }

---------------------------------------------------------------------------
PIPELINE STAGES
---------------------------------------------------------------------------
    raw frame
       │
       ▼
    1. VALIDATE   type-coerce, null-safe, track dropped sequences
       │
       ▼
    2. CLEAN      median-of-3 despike + hold-last-good on HR/SpO2/temp
       │
       ▼
    3. NORMALIZE  compute acc_mag, EMA-smooth slow signals,
       │          derive HRV-proxy from rolling HR buffer
       ▼
    4. SQI        per-signal 0..1 quality, finger-off / range gating,
       │          collect reject_reasons
       ▼
    ProcessedFrame  →  .to_engine_sample()  →  rule engine
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# ===========================================================================
# 0. CONSTANTS
# ===========================================================================

# Hard plausibility ranges. Anything outside is treated as sensor garbage and
# replaced with the last-good value (if any).
#
# Scientific rationale (per peer-reviewed clinical standards):
#
#   hr   30..220 bpm — 30 bpm is the His–Purkinje failsafe pacemaker rate;
#                      some elite endurance athletes reach 28–40 bpm at rest.
#                      220 bpm is the classic max-HR ceiling (Fox 1971).
#
#   spo2 60..100%   — NOT a clinical threshold. 60% is the lower reliable
#                      measurement range of reflectance PPG oximetry; values
#                      below this are rejected as sensor noise. Clinical
#                      hypoxemia alerts (<95% / <90% / <80%) live in the
#                      downstream rule engine, not here. (WHO Pulse Oximetry
#                      Training Manual; BTS Guidelines PMC5531304)
#
#   temp 25..42 °C  — SKIN (peripheral) temperature at the finger, NOT core
#                      body temp. Finger skin at thermoneutral rest runs
#                      32–35 °C (Brajkovic 2001). 25 °C floors cold ambient /
#                      reduced peripheral perfusion (Raynaud's episodes can
#                      drop to 18–25 °C — slightly beyond this gate).
#                      42 °C is a hard plausibility ceiling — peripheral
#                      readings at this level imply systemic emergency.
#                      Note: skin runs 4–7 °C below core, so skin 37.5 °C is
#                      NOT equivalent to core 37.5 °C — the rule engine
#                      must account for this.
RANGE = {
    "hr":   (30.0, 220.0),   # bpm
    "spo2": (60.0, 100.0),   # % (sensor plausibility, NOT clinical threshold)
    "temp": (25.0,  42.0),   # °C SKIN (peripheral), not core body temp
}

# Clinical downstream-alert thresholds. These DO NOT reject the frame — they
# are surfaced as advisory reasons for the rule engine to act on. Keeping
# clinical logic explicit and separate from sensor-quality logic.
CLINICAL_SKIN_TEMP_ELEVATED = 37.5   # °C skin; warrants downstream fever check
CLINICAL_SPO2_CONCERN       = 94.0   # %; <94% = clinical concern (BTS / WHO)
CLINICAL_SPO2_HYPOXEMIA     = 90.0   # %; <90% = hypoxemia / medical emergency

# Maximum physically reasonable change per 1-second frame. Jumps bigger than
# this trigger the despiker (median-of-3 substitution).
MAX_JUMP = {
    "hr":   25.0,   # bpm/s
    "spo2":  5.0,   # %/s
    "temp":  0.6,   # °C/s
}

# EMA smoothing factors. Higher alpha = more responsive, less smoothing.
EMA_ALPHA = {"hr": 0.7, "spo2": 0.3, "temp": 0.05}

# SQI weights for the overall quality score.
# PPG quality is gathered from adc_raw and feeds into the hr/spo2 scores.
SQI_WEIGHTS = {
    "hr": 0.30, "hrv": 0.10, "temp": 0.15,
    "acc": 0.15, "spo2": 0.15, "ppg": 0.15,
}

# ADC plausibility and saturation thresholds (MAX30102-class PPG, 18-bit).
ADC_MAX           = 262143       # 2^18 - 1
ADC_SATURATED_HI  = 250000       # near-rail → photodiode saturated
ADC_SATURATED_LO  = 2000         # near-zero → no signal / finger off
ADC_MIN_VARIANCE  = 500.0        # rolling σ below this = no visible pulse

# Thermal compensation: when die_temp (chip) runs hotter than its rolling
# baseline, some of that leaks into the skin temperature reading. We estimate
# the bias as k · (die_temp − die_baseline) and subtract it from `temperature`.
THERMAL_COMP_K     = 0.25        # °C skin bias per °C chip rise (empirical)
THERMAL_BIAS_ALERT = 0.4         # °C correction above which we lower temp SQI

# Battery gates — below this voltage the ADC noise floor rises.
VBAT_LOW_MV        = 3500        # flag SQI
VBAT_CRIT_MV       = 3300        # reject frame outright

# Gravity estimator alpha (low-pass → estimates gravity vector per axis).
GRAVITY_ALPHA      = 0.05


# ===========================================================================
# 1. OUTPUT DATACLASSES
# ===========================================================================

@dataclass
class SignalQuality:
    hr:      float = 0.0
    hrv:     float = 0.0
    temp:    float = 0.0
    acc:     float = 0.0
    spo2:    float = 0.0
    ppg:     float = 0.0   # from adc_raw (saturation + pulsatile variance)
    overall: float = 0.0


@dataclass
class ProcessedFrame:
    """Strict, normalized frame consumed by the rule engine."""
    timestamp: int
    sequence:  int

    hr:            Optional[float]
    hr_stability_score:     Optional[float]
    temp:          Optional[float]    # skin temp AFTER thermal compensation
    temp_raw:      Optional[float]    # skin temp BEFORE compensation (debug)
    spo2:          Optional[float]
    acc_mag:       float              # total |a| (incl. gravity)
    dyn_acc_mag:   float              # dynamic |a| (gravity removed)
    acc_x:         float
    acc_y:         float
    acc_z:         float

    finger_on:     bool
    charging:      bool               # charger_stat != idle OR vbus_present
    sqi:           SignalQuality

    accepted:       bool
    reject_reasons: List[str] = field(default_factory=list)
    clinical_flags: List[str] = field(default_factory=list)

    # Diagnostics — not required by the rule engine but useful for debugging.
    raw_status:     Optional[str] = None
    battery_mv:     Optional[int] = None
    die_temp:       Optional[float] = None
    adc_raw:        Optional[float] = None
    thermal_bias:   float = 0.0       # °C subtracted from raw skin temp
    cleaned_fields: List[str]     = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["sqi"] = asdict(self.sqi)
        return d

    def to_engine_sample(self) -> Dict:
        """Flat dict that the rule engine consumes directly."""
        return {
            "timestamp":    self.timestamp,
            "sequence":     self.sequence,
            "hr":           self.hr,
            "hr_stability_score":    self.hr_stability_score,
            "temp":         self.temp,
            "temp_raw":     self.temp_raw,
            "spo2":         self.spo2,
            "acc_mag":      self.acc_mag,
            "dyn_acc_mag":  self.dyn_acc_mag,
            "acc_x":        self.acc_x,
            "acc_y":        self.acc_y,
            "acc_z":        self.acc_z,
            "finger_on":    self.finger_on,
            "charging":     self.charging,
            "battery_mv":   self.battery_mv,
            "die_temp":     self.die_temp,
            "adc_raw":      self.adc_raw,
            "thermal_bias": self.thermal_bias,
            "sqi": {
                "hr":      self.sqi.hr,
                "hrv":     self.sqi.hrv,
                "temp":    self.sqi.temp,
                "acc":     self.sqi.acc,
                "spo2":    self.sqi.spo2,
                "ppg":     self.sqi.ppg,
                "overall": self.sqi.overall,
            },
            "accepted":       self.accepted,
            "reject_reasons": list(self.reject_reasons),
            "clinical_flags": list(self.clinical_flags),
        }


# ===========================================================================
# 2. TINY STATEFUL HELPERS (despiker, EMA, RR buffer, accel jitter)
# ===========================================================================

def _coerce(value, default=None) -> Optional[float]:
    """Best-effort float coercion, None-safe."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median3(values: List[float]) -> float:
    return sorted(values)[len(values) // 2]


class _Despiker:
    """
    Median-of-3 despiker with hold-last-good fallback.

    Rules:
      - None or out-of-range → return last_good (nothing new published)
      - Value within range but jumps more than `max_jump` from last_good
        → replace with median of last 3 (or last_good if buffer not full)
      - Otherwise → pass through and remember as last_good
    """

    def __init__(self, max_jump: float, lo: float, hi: float):
        self.buf: Deque[float] = deque(maxlen=3)
        self.last_good: Optional[float] = None
        self.max_jump = max_jump
        self.lo, self.hi = lo, hi

    def step(self, value: Optional[float]) -> Tuple[Optional[float], bool]:
        """Returns (cleaned_value, was_cleaned_or_held)."""
        if value is None or value < self.lo or value > self.hi:
            return self.last_good, self.last_good is not None

        cleaned = False
        if (
            self.last_good is not None
            and abs(value - self.last_good) > self.max_jump
        ):
            if len(self.buf) == 3:
                value = _median3(list(self.buf) + [value])
            else:
                value = self.last_good
            cleaned = True

        self.buf.append(value)
        self.last_good = value
        return value, cleaned


class _EMA:
    """Exponential moving average: y = α·x + (1-α)·y_prev."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def step(self, x: Optional[float]) -> Optional[float]:
        if x is None:
            return self.value
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class _RRBuffer:
    """
    Rolling buffer used to produce an "HR stability score" from instantaneous
    HR values.

    IMPORTANT — this is NOT clinical RMSSD / HRV.

    True clinical RMSSD is computed from consecutive R-R intervals at
    millisecond resolution from an ECG (or a dedicated PPG beat detector).
    Converting 1 Hz averaged HR values to pseudo-RR intervals via
    `RR ≈ 60000/HR` fundamentally cannot capture the high-frequency HRV
    components (0.15–0.4 Hz, respiratory sinus arrhythmia) that constitute
    the clinically meaningful RMSSD signal — the 1 Hz Nyquist ceiling is
    0.5 Hz and HR is already pre-averaged by the firmware.

    What this metric DOES capture: short-window HR stability. Useful for
    relative trend detection against a personal baseline (stress onset,
    exercise recovery). NOT useful as an absolute clinical HRV measurement.

    Reference: Task Force of ESC / NASPE, 1996, Circulation 93:1043-1065.
    Ultra-short window validity: Esco & Flatt, 2014, J Sports Sci Med.

    The output field is named `hr_stability_score` to clearly indicate this
    is NOT a clinical RMSSD / HRV measurement.
    SQI.hrv is capped at 0.7 to reflect this honestly.
    """

    def __init__(self, maxlen: int = 30):
        self.rr: Deque[float] = deque(maxlen=maxlen)

    def update(self, hr: Optional[float]) -> Optional[float]:
        if hr is None or hr <= 0:
            return None
        self.rr.append(60000.0 / hr)
        if len(self.rr) < 4:
            return None
        diffs = [self.rr[i + 1] - self.rr[i] for i in range(len(self.rr) - 1)]
        if len(diffs) < 2:
            return None
        return math.sqrt(sum(d * d for d in diffs) / (len(diffs) - 1))


class _AccelJitter:
    """
    Tracks short-window dynamic-acc variance to build an acceleration SQI
    score. Low jitter → high quality (still, PPG is clean). High jitter →
    low quality (movement artifacts contaminate HR/SpO2).

    Feed this with `dyn_acc_mag` (gravity already removed), not raw |a|.
    """

    def __init__(self, maxlen: int = 10):
        self.buf: Deque[float] = deque(maxlen=maxlen)

    def update(self, dyn_acc_mag: float) -> float:
        self.buf.append(dyn_acc_mag)
        if len(self.buf) < 3:
            return 0.95
        n = len(self.buf)
        mean = sum(self.buf) / n
        var = sum((x - mean) ** 2 for x in self.buf) / (n - 1)
        return max(0.0, min(1.0, 1.0 - min(var * 80.0, 1.0)))


class _GravityEstimator:
    """
    Low-pass-filters the accelerometer per axis to estimate the current
    gravity vector. The dynamic motion signal is then:

        dyn = raw − gravity

    This is the standard trick used by iOS CoreMotion / Android. It gives
    us a motion signal that's zero when the user is still, regardless of
    how the ring is oriented, which is *exactly* what we want for PPG
    motion-artifact gating and fall detection.
    """

    def __init__(self, alpha: float = GRAVITY_ALPHA):
        self.alpha = alpha
        self.gx = self.gy = self.gz = None

    def step(self, ax: float, ay: float, az: float) -> Tuple[float, float, float]:
        if self.gx is None:
            self.gx, self.gy, self.gz = ax, ay, az
        else:
            self.gx = self.alpha * ax + (1 - self.alpha) * self.gx
            self.gy = self.alpha * ay + (1 - self.alpha) * self.gy
            self.gz = self.alpha * az + (1 - self.alpha) * self.gz
        return ax - self.gx, ay - self.gy, az - self.gz


class _PPGQuality:
    """
    Scores PPG quality directly from the `adc_raw` photodiode reading.

    A valid pulsatile signal looks like:
      - not rail-saturated (adc well below ADC_MAX)
      - not near zero (finger pressure / off-wrist)
      - has visible AC component across a short window (pulsatile variance
        above ADC_MIN_VARIANCE)

    An HR value that's reported while the ADC is flat is *fabricated* by
    the sensor's internal tracking loop and should be rejected.
    """

    def __init__(self, maxlen: int = 8):
        self.buf: Deque[float] = deque(maxlen=maxlen)

    def step(self, adc: Optional[float]) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if adc is None:
            return 0.0, ["ppg_missing"]

        # Saturation / flat-line gates.
        if adc >= ADC_SATURATED_HI:
            return 0.1, ["ppg_saturated_high"]
        if adc <= ADC_SATURATED_LO:
            return 0.1, ["ppg_saturated_low"]

        self.buf.append(adc)
        if len(self.buf) < 3:
            return 0.75, []   # tentative while warming up

        n = len(self.buf)
        mean = sum(self.buf) / n
        var  = sum((x - mean) ** 2 for x in self.buf) / (n - 1)
        sigma = math.sqrt(var)

        if sigma < ADC_MIN_VARIANCE:
            return 0.2, ["ppg_no_pulse"]

        # Map sigma to 0..1 — cap at a generous upper bound.
        score = min(1.0, sigma / (ADC_MIN_VARIANCE * 6.0))
        return max(0.5, score), []


class _ThermalCompensator:
    """
    Estimates a slow baseline for `die_temp` (chip temperature) and returns
    a correction to subtract from the reported skin `temperature`.

    When the chip self-heats — charging, direct sun, handheld measurement —
    the skin temperature sensor sees some of that heat and over-reports.
    Empirically about 0.25 °C of skin bias per 1 °C of chip rise above the
    rolling baseline. A heavy low-pass on die_temp gives us the baseline.
    """

    def __init__(self, alpha: float = 0.02):
        self.alpha = alpha
        self.baseline: Optional[float] = None

    def step(self, die_temp: Optional[float]) -> float:
        if die_temp is None:
            return 0.0
        if self.baseline is None:
            self.baseline = die_temp
            return 0.0
        self.baseline = self.alpha * die_temp + (1 - self.alpha) * self.baseline
        delta = die_temp - self.baseline
        if delta <= 0:
            return 0.0
        return THERMAL_COMP_K * delta


# ===========================================================================
# 3. THE PIPELINE
# ===========================================================================

class Preprocessor:
    """
    Stateful per-device preprocessing pipeline.

    Usage:
        pp = Preprocessor()
        for raw_frame in device_stream:
            processed = pp.process(raw_frame)          # ProcessedFrame
            engine_input = processed.to_engine_sample() # dict → rule engine
    """

    def __init__(self):
        # Stage-2 despikers
        self.dsp_hr   = _Despiker(MAX_JUMP["hr"],   *RANGE["hr"])
        self.dsp_spo2 = _Despiker(MAX_JUMP["spo2"], *RANGE["spo2"])
        self.dsp_temp = _Despiker(MAX_JUMP["temp"], *RANGE["temp"])

        # Stage-3 smoothers
        self.ema_hr   = _EMA(EMA_ALPHA["hr"])
        self.ema_spo2 = _EMA(EMA_ALPHA["spo2"])
        self.ema_temp = _EMA(EMA_ALPHA["temp"])

        self.rr_buffer = _RRBuffer()
        self.jitter    = _AccelJitter()
        self.gravity   = _GravityEstimator()
        self.ppg       = _PPGQuality()
        self.thermal   = _ThermalCompensator()

        # Rolling stats
        self.frames_in       = 0
        self.frames_accepted = 0
        self.frames_rejected = 0
        self.last_seq: Optional[int] = None
        self.dropped_seq     = 0

    # ---------- Stage 1: validate ----------
    def _validate(self, raw: Dict) -> Dict:
        """Type-coerce everything; anything unparseable becomes None."""
        charger = raw.get("charger_stat")
        vbus    = bool(raw.get("vbus_present", False))
        charging = vbus or (
            isinstance(charger, str)
            and charger.lower() not in ("idle", "none", "off", "")
        )

        return {
            "accel_x":    _coerce(raw.get("accel_x"), 0.0),
            "accel_y":    _coerce(raw.get("accel_y"), 0.0),
            "accel_z":    _coerce(raw.get("accel_z"), 0.0),
            "hr":         _coerce(raw.get("heart_rate")),
            "spo2":       _coerce(raw.get("spo2")),
            "temp":       _coerce(raw.get("temperature")),
            "adc_raw":    _coerce(raw.get("adc_raw")),
            "die_temp":   _coerce(raw.get("die_temp")),
            "vbat_mv":    _coerce(raw.get("vbat_mv")),
            "finger_on":  bool(raw.get("finger_on", False)),
            "charging":   charging,
            "charger_stat": charger,
            "vbus_present": vbus,
            "sequence":   int(raw.get("sequence") or 0),
            "status":     raw.get("status"),
        }

    # ---------- Stage 2: clean ----------
    def _clean(self, v: Dict) -> Tuple[Dict, List[str]]:
        cleaned: List[str] = []
        v["hr"],   c = self.dsp_hr.step(v["hr"])
        if c: cleaned.append("hr")
        v["spo2"], c = self.dsp_spo2.step(v["spo2"])
        if c: cleaned.append("spo2")
        v["temp"], c = self.dsp_temp.step(v["temp"])
        if c: cleaned.append("temp")
        return v, cleaned

    # ---------- Stage 3: normalize ----------
    def _normalize(self, v: Dict) -> Dict:
        # Total |a| (gravity included) and dynamic |a| (gravity removed).
        ax, ay, az = v["accel_x"], v["accel_y"], v["accel_z"]
        v["acc_mag"] = math.sqrt(ax * ax + ay * ay + az * az)
        dx, dy, dz = self.gravity.step(ax, ay, az)
        v["dyn_acc_mag"] = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Thermal compensation: correct skin temp using die_temp drift.
        v["temp_raw"] = v["temp"]
        bias = self.thermal.step(v["die_temp"])
        if v["temp"] is not None:
            v["temp"] = v["temp"] - bias
        v["thermal_bias"] = bias

        # Save raw (despiked but unsmoothed) HR for RR interval computation.
        raw_hr = v["hr"]

        # Light smoothing on slow-varying signals.
        v["hr"]   = self.ema_hr.step(v["hr"])
        v["spo2"] = self.ema_spo2.step(v["spo2"])
        v["temp"] = self.ema_temp.step(v["temp"])

        # HR stability score derived from the RAW (despiked, unsmoothed) HR
        # stream. Using unsmoothed HR preserves the inter-beat variability
        # that EMA would attenuate, giving a more honest stability metric.
        v["hr_stability_score"] = self.rr_buffer.update(raw_hr)
        return v

    # ---------- Stage 4: signal quality check ----------
    def _sqi(self, v: Dict) -> Tuple[SignalQuality, List[str], List[str]]:
        """Returns (sqi, reject_reasons, clinical_flags).

        `reject_reasons` are QUALITY failures that should drop the frame
        from baseline learning and downstream rule evaluation.

        `clinical_flags` are advisory markers (elevated skin temp, SpO2
        clinical concern) that DO NOT reject the frame — they're passed
        to the rule engine as additional context. Quality and clinical
        concern are deliberately kept separate so a high-quality reading
        of a sick user still gets analyzed.
        """
        reasons: List[str] = []
        clinical: List[str] = []
        sq = SignalQuality()

        # --- Contact / charging / battery gates (hard) ---
        if not v["finger_on"]:
            reasons.append("finger_off")

        if v["charging"]:
            reasons.append("charging")

        if v["vbat_mv"] is not None:
            if v["vbat_mv"] < VBAT_CRIT_MV:
                reasons.append(f"battery_critical ({int(v['vbat_mv'])}mV)")
            elif v["vbat_mv"] < VBAT_LOW_MV:
                reasons.append(f"battery_low ({int(v['vbat_mv'])}mV)")

        # --- PPG quality from adc_raw (drives HR + SpO2 ceilings) ---
        sq.ppg, ppg_reasons = self.ppg.step(v["adc_raw"])
        reasons.extend(ppg_reasons)

        # --- Heart rate ---
        # Upper SQI bound = 200 bpm (was 180 bpm).
        # Scientific correction: maximum predicted HR (Tanaka 2001, JACC 37)
        # = 208 − 0.7·age. A 20-year-old's MPHR is ~194 bpm; 25-year-old's
        # ~190 bpm. Young adults routinely reach 185–200 bpm during vigorous
        # exercise, so 180 bpm was incorrectly rejecting valid readings.
        if v["hr"] is None:
            sq.hr = 0.0
            reasons.append("hr_missing")
        elif not (40.0 <= v["hr"] <= 200.0):
            sq.hr = 0.2
            reasons.append(f"hr_out_of_range ({v['hr']:.0f})")
        else:
            base = 0.95 if v["finger_on"] else 0.3
            # HR quality is bounded above by the PPG quality — an HR value
            # with no pulsatile signal behind it is untrustworthy.
            sq.hr = min(base, 0.2 + 0.8 * sq.ppg)

        # --- HR-stability score (kept under the `hrv` field name for
        # backward compatibility with the rule engine) ---
        # Cap raised from 0.6 → 0.7 per scientific review: for relative
        # trend detection against a personal baseline (the WUALT use case)
        # the ultra-short 1 Hz proxy retains more validity than 0.6 implied.
        # Still below the 1.0 ceiling to flag that this is NOT a clinical
        # RMSSD measurement.  (Esco & Flatt, 2014, J Sports Sci Med.)
        sq.hrv = min(0.7, sq.hr)
        if v["hr_stability_score"] is None:
            sq.hrv = 0.0

        # --- Temperature (uses compensated value) ---
        # Note: this is SKIN (peripheral) temperature, not core body temp.
        # Finger skin normally runs 32–35 °C at thermoneutral rest; skin
        # runs ~4–7 °C below core. Quality gate (30–40.5 °C) is separate
        # from clinical concern threshold (>37.5 °C skin — downstream flag
        # for possible fever, not a quality failure).
        if v["temp"] is None:
            sq.temp = 0.0
            reasons.append("temp_missing")
        elif 30.0 <= v["temp"] <= 40.5:
            sq.temp = 0.9
            # Penalize when thermal compensation had to subtract a lot.
            if v["thermal_bias"] > THERMAL_BIAS_ALERT:
                sq.temp = max(0.5, sq.temp - v["thermal_bias"])
                reasons.append(f"thermal_bias ({v['thermal_bias']:.2f}C)")
            # Advisory-only: clinical elevated-skin-temp marker for the
            # downstream rule engine. Does NOT reduce SQI or reject the
            # frame — the reading is still high quality, just clinically
            # worth flagging. Skin 37.5 °C is NOT equivalent to core
            # 37.5 °C (skin runs 4–7 °C below core), so this is a
            # "worth investigating" marker, not a fever diagnosis.
            if v["temp"] > CLINICAL_SKIN_TEMP_ELEVATED:
                clinical.append(
                    f"elevated_skin_temp ({v['temp']:.2f}C)"
                )
        else:
            sq.temp = 0.3

        # --- SpO2 (also bounded by PPG quality) ---
        # Quality threshold (70%) is intentionally distinct from the
        # plausibility threshold (60%, in RANGE). Clinical concern tiers
        # (94% / 90%) are surfaced as advisory reasons but do NOT penalize
        # SQI — a 92% reading can still be a high-quality measurement.
        if v["spo2"] is None:
            sq.spo2 = 0.0
        elif 70.0 <= v["spo2"] <= 100.0:
            base = 0.9 if v["finger_on"] else 0.3
            sq.spo2 = min(base, 0.2 + 0.8 * sq.ppg)
            if v["spo2"] < CLINICAL_SPO2_HYPOXEMIA:
                clinical.append(f"spo2_hypoxemia ({v['spo2']:.1f}%)")
            elif v["spo2"] < CLINICAL_SPO2_CONCERN:
                clinical.append(f"spo2_clinical_concern ({v['spo2']:.1f}%)")
        else:
            sq.spo2 = 0.2

        # --- Accelerometer SQI from dynamic motion (gravity removed) ---
        # Motion does NOT reject the frame outright — instead it degrades
        # confidence in PPG-derived signals (HR, SpO2). This keeps motion
        # frames in the pipeline (important for fall detection and activity
        # tracking) while honestly reporting reduced PPG reliability.
        sq.acc = self.jitter.update(v["dyn_acc_mag"])
        if sq.acc < 0.7:
            # Downgrade HR and SpO2 quality by motion penalty instead of
            # rejecting the frame. The penalty is proportional to acc SQI.
            sq.hr   *= sq.acc
            sq.spo2 *= sq.acc

        # --- Hard gate on HR quality ---
        if sq.hr < 0.7:
            reasons.append(f"low_hr_sqi ({sq.hr:.2f})")

        # --- Weighted overall ---
        total_w, total_v = 0.0, 0.0
        for name, w in SQI_WEIGHTS.items():
            val = getattr(sq, name)
            if val > 0:
                total_w += w
                total_v += val * w
        sq.overall = (total_v / total_w) if total_w > 0 else 0.0

        return sq, reasons, clinical

    # ---------- Public ----------
    def process(self, raw: Dict) -> ProcessedFrame:
        """Run all 4 stages on one raw frame and return a ProcessedFrame."""
        self.frames_in += 1

        v = self._validate(raw)

        # Track dropped sequence numbers for diagnostics.
        if self.last_seq is not None and v["sequence"] > self.last_seq + 1:
            self.dropped_seq += v["sequence"] - self.last_seq - 1
        self.last_seq = v["sequence"]

        v, cleaned_fields     = self._clean(v)
        v                     = self._normalize(v)
        sq, reasons, clinical = self._sqi(v)

        accepted = len(reasons) == 0
        if accepted:
            self.frames_accepted += 1
        else:
            self.frames_rejected += 1

        return ProcessedFrame(
            timestamp      = int(time.time()),
            sequence       = v["sequence"],
            hr             = v["hr"],
            hr_stability_score      = v["hr_stability_score"],
            temp           = v["temp"],
            temp_raw       = v.get("temp_raw"),
            spo2           = v["spo2"],
            acc_mag        = v["acc_mag"],
            dyn_acc_mag    = v["dyn_acc_mag"],
            acc_x          = v["accel_x"],
            acc_y          = v["accel_y"],
            acc_z          = v["accel_z"],
            finger_on      = v["finger_on"],
            charging       = v["charging"],
            sqi            = sq,
            accepted       = accepted,
            reject_reasons = reasons,
            clinical_flags = clinical,
            raw_status     = v.get("status"),
            battery_mv     = int(v["vbat_mv"]) if v.get("vbat_mv") is not None else None,
            die_temp       = v.get("die_temp"),
            adc_raw        = v.get("adc_raw"),
            thermal_bias   = v.get("thermal_bias", 0.0),
            cleaned_fields = cleaned_fields,
        )

    def stats(self) -> Dict:
        return {
            "frames_in":         self.frames_in,
            "frames_accepted":   self.frames_accepted,
            "frames_rejected":   self.frames_rejected,
            "dropped_sequences": self.dropped_seq,
            "accept_rate":       (
                self.frames_accepted / self.frames_in if self.frames_in else 0.0
            ),
        }


# ===========================================================================
# 4. PERSONAL BASELINE (z-score normalization with cold-start handling)
# ===========================================================================

class PersonalBaseline:
    """
    Rolling per-user baseline for z-score normalization.

    Why personal baseline instead of population norms?
        A resting HR of 62 is normal for one person and dangerously low for
        another. Rule-based anomaly detection needs deviations relative to
        *this* user, not a textbook.

    Cold-start problem:
        The first N frames don't have enough history to compute a meaningful
        mean / std. Strategy:
            - frames < warmup_size     → ready=False, z-score = 0.0,
                                          rule engine should operate in
                                          "warming up" mode (absolute rules
                                          only, no deviation rules)
            - frames >= warmup_size    → ready=True, emit real z-scores
        Only update the baseline with frames that are SQI-accepted AND
        not during an active critical event (caller's responsibility).
    """

    def __init__(self, warmup_size: int = 60, window_size: int = 1800):
        self.warmup_size = warmup_size
        self.window_size = window_size
        self.buffers: Dict[str, Deque[float]] = {
            "hr":        deque(maxlen=window_size),
            "hr_stability_score": deque(maxlen=window_size),
            "temp":      deque(maxlen=window_size),
            "spo2":      deque(maxlen=window_size),
            "acc_mag":   deque(maxlen=window_size),
        }

    def ready(self) -> bool:
        return len(self.buffers["hr"]) >= self.warmup_size

    def update(self, sample: Dict) -> None:
        """Unconditional update — kept for backward compatibility."""
        for key in self.buffers:
            val = sample.get(key)
            if val is not None:
                self.buffers[key].append(float(val))

    def gated_update(self, sample: Dict, engine_state: Optional[str]) -> None:
        """
        Preferred entry point. Only learns from 'normal' engine states so
        that distress / alarm events don't pollute the personal baseline.

        During warmup (engine_state is None — engine not yet running), we
        apply simple outlier rejection: reject any value more than 3 SD
        from the running mean. This prevents early spikes from biasing
        the baseline before the engine has enough data to classify state.
        """
        for key in self.buffers:
            val = sample.get(key)
            if val is None:
                continue
            val = float(val)

            if engine_state is not None:
                # Engine is running — only update on normal state.
                if engine_state == "normal":
                    self.buffers[key].append(val)
            else:
                # Warmup: outlier rejection (|val - mean| > 3 * SD).
                buf = self.buffers[key]
                if len(buf) < 2:
                    # Not enough data for stats — accept unconditionally.
                    buf.append(val)
                else:
                    mean = sum(buf) / len(buf)
                    var = sum((x - mean) ** 2 for x in buf) / (len(buf) - 1)
                    sd = math.sqrt(var) if var > 1e-9 else 1.0
                    if abs(val - mean) <= 3.0 * sd:
                        buf.append(val)
                    # else: outlier rejected during warmup

    def _stats(self, key: str) -> Tuple[float, float]:
        buf = self.buffers[key]
        if len(buf) < 2:
            return 0.0, 1.0
        mean = sum(buf) / len(buf)
        var  = sum((x - mean) ** 2 for x in buf) / (len(buf) - 1)
        std  = math.sqrt(var) if var > 1e-9 else 1.0
        return mean, std

    def zscores(self, sample: Dict) -> Dict[str, float]:
        """
        Return {signal: z-score} for the current sample. During cold-start
        (ready() == False) every z-score is 0.0 — meaning "no signal".
        """
        out: Dict[str, float] = {}
        if not self.ready():
            return {k: 0.0 for k in self.buffers}
        for key in self.buffers:
            val = sample.get(key)
            if val is None:
                out[key] = 0.0
                continue
            mean, std = self._stats(key)
            out[key] = (val - mean) / std
        return out


# ===========================================================================
# 5. WINDOWING (30-second rolling aggregation for the rule engine)
# ===========================================================================

class RollingWindow:
    """
    Fixed-length rolling window over ProcessedFrames.

    Why 30 s?
        - Short enough for real-time alerting (< 1-min reaction time)
        - Long enough to compute stable mean / variance of HR and acc
        - Aligns with short-term HRV windows used clinically

    Per call you get streaming statistics — no re-scan of the buffer, O(1)
    update cost, so it's trivially real-time.
    """

    def __init__(self, window_seconds: int = 30):
        self.window_seconds = window_seconds
        self.samples: Deque[Dict] = deque()

    def add(self, sample: Dict) -> None:
        self.samples.append(sample)
        cutoff = sample["timestamp"] - self.window_seconds
        while self.samples and self.samples[0]["timestamp"] < cutoff:
            self.samples.popleft()

    def features(self) -> Dict[str, float]:
        """Mean + variance + min/max per signal across the window."""
        feats: Dict[str, float] = {"window_n": len(self.samples)}
        keys = ("hr", "hr_stability_score", "temp", "spo2", "acc_mag")
        for k in keys:
            vals = [s[k] for s in self.samples if s.get(k) is not None]
            if not vals:
                feats[f"{k}_mean"] = 0.0
                feats[f"{k}_var"]  = 0.0
                feats[f"{k}_min"]  = 0.0
                feats[f"{k}_max"]  = 0.0
                continue
            mean = sum(vals) / len(vals)
            var  = (
                sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                if len(vals) > 1 else 0.0
            )
            feats[f"{k}_mean"] = mean
            feats[f"{k}_var"]  = var
            feats[f"{k}_min"]  = min(vals)
            feats[f"{k}_max"]  = max(vals)
        return feats


# ===========================================================================
# 6. TOP-LEVEL ORCHESTRATOR — one call, complete output
# ===========================================================================

class AnomalyInputBuilder:
    """
    High-level wrapper that ties everything together:

        raw frame
           │
           ▼
        Preprocessor   (validate → clean → normalize → SQI)
           │
           ├─► PersonalBaseline.update()   (if accepted)
           │
           ├─► RollingWindow.add()
           │
           ▼
        {sample, zscores, window, sqi, accepted} → rule engine
    """

    def __init__(
        self,
        window_seconds: int = 30,
        baseline_warmup: int = 60,
    ):
        self.pre = Preprocessor()
        self.baseline = PersonalBaseline(warmup_size=baseline_warmup)
        self.window = RollingWindow(window_seconds=window_seconds)
        self._engine_state: Optional[str] = None

    def set_engine_state(self, state: str) -> None:
        """
        Receive feedback from the distress engine so the baseline can
        gate its updates. Call this after each engine evaluation with
        the current engine state (e.g. 'normal', 'elevated', 'distress').
        """
        self._engine_state = state

    def step(self, raw: Dict) -> Dict:
        pf = self.pre.process(raw)
        sample = pf.to_engine_sample()

        # Only learn from accepted frames, and gate on engine state so
        # distress events don't pollute the personal baseline.
        if pf.accepted:
            self.baseline.gated_update(sample, self._engine_state)
            self.window.add(sample)

        return {
            "sample":         sample,
            "zscores":        self.baseline.zscores(sample),
            "baseline_ready": self.baseline.ready(),
            "window":         self.window.features(),
        }


# ===========================================================================
# 7. DEMO  (run:  python preprocessing_pipeline.py)
# ===========================================================================

if __name__ == "__main__":
    import json
    import random

    builder = AnomalyInputBuilder(window_seconds=30, baseline_warmup=5)

    print("=" * 72)
    print(" DEMO: feeding 8 synthetic frames through the full pipeline")
    print("=" * 72)

    for seq in range(1, 9):
        # Simulate a realistic resting frame with tiny jitter.
        raw = {
            "accel_x":      random.gauss(0.02, 0.01),
            "accel_y":      random.gauss(0.02, 0.01),
            "accel_z":      random.gauss(0.98, 0.01),
            "adc_raw":      random.gauss(110000, 4000),   # pulsatile variance ~4000
            "heart_rate":   random.gauss(72, 2),
            "spo2":         random.gauss(98.5, 0.4),
            "temperature":  random.gauss(36.6, 0.1),
            "vbat_mv":      3850,
            "die_temp":     random.gauss(35.0, 0.3),
            "charger_stat": "idle",
            "vbus_present": False,
            "finger_on":    True,
            "sequence":     seq,
            "status":       "ok",
        }

        # Inject a garbage HR spike at seq=5 to show the despiker working.
        if seq == 5:
            raw["heart_rate"] = 240.0

        out = builder.step(raw)

        print(f"\n--- frame {seq} ---")
        print("raw   :", {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in raw.items()})
        print("engine:", json.dumps(out, indent=2, default=str))

    print("\n" + "=" * 72)
    print(" final pipeline stats")
    print("=" * 72)
    print(json.dumps(builder.pre.stats(), indent=2))
