"""
WUALT — Multi-Layer Safety Intelligence Engine
====================================================================

A standalone safety intelligence engine with five detection capabilities:
  1. Physiological distress from body signals (HR, HRV, SpO2, temp)
  2. Fall detection via 3-stage accelerometer model
  3. Geospatial safety risk (location, movement, time, familiarity)
  4. Audio intelligence (environmental sound classification, vocal stress,
     noise monitoring) — optional, requires ML dependencies
  +  EDA/GSR processing — optional sensor for future hardware

Layers 1-3 are deterministic rule-based (no ML dependencies).
Layer 4 uses YAMNet (Google) + AST (MIT) for sound classification
and Praat/parselmouth for vocal stress analysis. Layer 4 is fully
optional — the engine works without it.

Answers the question:
    "Given the user's body signals, environment, and context,
     how concerning is the situation right now?"

Design goals:
    - Real-time:       < 700 ms per evaluation (Layers 1-3, O(1))
    - Interpretable:   every decision is traceable
    - Robust:          handles noisy wearable data, cold-start, motion artifacts
    - Human-first:     alerts are calm, clear, actionable — no medical jargon
    - Privacy-first:   no raw GPS stored, only zone classification
    - Women's safety:  tuned for real-world safety scenarios
    - Graceful degradation: works with any subset of sensors/layers

---------------------------------------------------------------------------
INPUT — Physiological (from AnomalyInputBuilder.step())
---------------------------------------------------------------------------
    {
        "sample":         { ... },    # processed frame from preprocessing
        "zscores":        { ... },    # personal baseline deviations
        "baseline_ready": bool,
        "window":         { ... },    # 30-sec rolling stats
    }

---------------------------------------------------------------------------
INPUT — Geospatial Context (from phone GPS / companion app)
---------------------------------------------------------------------------
    {
        "latitude":              float,
        "longitude":             float,
        "timestamp":             int,
        "speed_kmph":            float,
        "heading":               float,
        "is_home_zone":          bool,
        "is_work_zone":          bool,
        "is_known_area":         bool,
        "is_unfamiliar_area":    bool,
        "distance_from_home_km": float,
        "hour_of_day":           int,
        "is_night":              bool,
        "is_stationary":         bool,
        "is_walking":            bool,
        "is_vehicle_like_motion": bool,
        "sudden_route_change":   bool,
        "sudden_stop":           bool,
        "phone_connected":       bool,
    }

---------------------------------------------------------------------------
OUTPUT — Unified Safety Result
---------------------------------------------------------------------------
    {
        # --- Physiological layer ---
        "state":       "normal" | "stress" | "distress",
        "confidence":  float 0.0..1.0,
        "contributing_signals": [str, ...],

        # --- Fall detection layer ---
        "fall_detected": {
            "detected":    bool,
            "stage":       str,
            "confidence":  float,
            "description": str,
        },

        # --- Geospatial safety layer ---
        "safety": {
            "risk_level":  "normal" | "low_risk" | "moderate_risk"
                           | "high_risk" | "critical",
            "risk_score":  float 0.0..1.0,
            "reasoning":   [str, ...],
            "recommended_action": str,
            "alert": { ... },
        },

        # --- Audio intelligence layer (None if unavailable) ---
        "audio_intelligence": {
            "audio_risk_score": float 0.0..1.0,
            "audio_risk_level": "normal" | "elevated" | "danger",
            "env_classification": { ... },
            "vocal_stress":      { ... },
            "noise_level":       { ... },
            "alert": { ... },
        } | None,

        # --- EDA/GSR (None if unavailable) ---
        "eda": {
            "scl_mean":    float,
            "scr_per_min": float,
            "eda_score":   float 0.0..1.0,
            "eda_status":  "calm" | "moderate" | "high_arousal",
        } | None,

        # --- Alert (highest priority across all layers) ---
        "alert": { "title": str, "message": str, "severity": str },

        # --- Debug ---
        "debug": { ... },
    }

---------------------------------------------------------------------------
DETECTION LOGIC — DISTRESS (physiological)
---------------------------------------------------------------------------
    1. GATE:    skip if frame rejected or baseline not ready (absolute only)
    2. MOTION:  detect exercise to avoid false stress alerts
    3. FLAG:    per-signal boolean flags from z-scores + absolute thresholds
    4. SCORE:   weighted signal scoring (temp is weak/supporting)
    5. PERSIST: require condition to hold for >= 60s before promoting state
    6. DECIDE:  normal / stress / distress based on flag count + persistence
    7. ALERT:   generate calm, actionable, STAR-principle user message

---------------------------------------------------------------------------
DETECTION LOGIC — FALL (3-stage)
---------------------------------------------------------------------------
    Stage 1 — IMPACT DETECTION (threshold-based)
        • Free-fall:  acc_mag drops below 0.5g for 3 consecutive samples
        • Impact:     acc_mag spikes above 5.0g (sudden deceleration)

    Stage 2 — ORIENTATION CHANGE
        • Check if body orientation changed by ~90° from vertical
        • Estimated from accelerometer axis ratios (no gyro needed)

    Stage 3 — POST-FALL THREE-OUTCOME MONITORING
        • Stillness >= 3s (dyn_acc < 0.05g) → CONFIRMED (unconscious)
        • Vigorous activity (dyn_acc > 0.5g sustained 3s+) → CANCELLED
        • Weak/moderate movement 10s+ → CONFIRMED_UNCERTAIN

---------------------------------------------------------------------------
DETECTION LOGIC — GEOSPATIAL SAFETY (context layer)
---------------------------------------------------------------------------
    The geospatial model does NOT trigger alerts by itself.
    It amplifies or suppresses the physiological state:

    1. CONTEXT FACTORS:  score each dimension (time, location, movement)
    2. RISK AMPLIFY:     unfamiliar + night + alone → amplify distress
    3. RISK SUPPRESS:    home + daytime + known area → suppress stress
    4. COMBINE:          physio_state × context_multiplier → risk_level
    5. ALERT:            context-aware, privacy-respecting, actionable

    Risk levels:
        normal        → physio normal + safe context
        low_risk      → mild stress OR normal in slightly unusual context
        moderate_risk → stress in concerning context OR distress in safe context
        high_risk     → distress in concerning context
        critical      → distress + multiple high-risk context factors
                         OR fall confirmed in any context

Run directly for a demo:
    python distress_engine.py
"""

from __future__ import annotations

import math
import time
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Optional heavy dependencies — audio intelligence (Layer 4) & EDA
# NOT required for core distress/fall/geo detection.
# ---------------------------------------------------------------------------
_AUDIO_AVAILABLE: bool = False
_EDA_AVAILABLE:   bool = False

_yamnet_model   = None
_yamnet_classes = None
_ast_model      = None
_ast_extractor  = None


def _ensure_audio_deps() -> bool:
    """Lazy-load TensorFlow, PyTorch, Transformers, librosa, parselmouth."""
    global _AUDIO_AVAILABLE
    if _AUDIO_AVAILABLE:
        return True
    try:
        import tensorflow       # noqa: F401
        import tensorflow_hub   # noqa: F401
        import torch            # noqa: F401
        from transformers import AutoFeatureExtractor, ASTForAudioClassification  # noqa: F401
        import librosa          # noqa: F401
        import parselmouth      # noqa: F401
        _AUDIO_AVAILABLE = True
        return True
    except ImportError:
        return False


def _ensure_eda_deps() -> bool:
    """Lazy-load scipy for EDA/GSR processing."""
    global _EDA_AVAILABLE
    if _EDA_AVAILABLE:
        return True
    try:
        from scipy.signal import find_peaks, butter, filtfilt  # noqa: F401
        _EDA_AVAILABLE = True
        return True
    except ImportError:
        return False


# ===========================================================================
# 0. CONFIGURATION
# ===========================================================================

# --- Z-score thresholds for flagging ---
# Direction matters: HR ↑ = stress, HRV ↓ = stress, SpO2 ↓ = stress
ZSCORE_THRESHOLDS = {
    "hr":                 2.0,    # z > +2.0 → flag (elevated)
    "hr_stability_score": -1.5,   # z < -1.5 → flag (suppressed — note: inverted)
    "spo2":               -2.0,   # z < -2.0 → flag (desaturation)
    "temp":               2.5,    # z > +2.5 → flag (weak signal, high threshold)
    "temp_low":           -1.5,   # z < -1.5 → flag (vasoconstriction / distress)
}

# --- Absolute thresholds (always active, even during cold start) ---
# These fire regardless of baseline — universal physiological limits.
ABSOLUTE_THRESHOLDS = {
    "hr_high":      120.0,   # bpm — clearly elevated for a resting person
    "hr_very_high": 150.0,   # bpm — concerning even during exercise
    "spo2_low":      94.0,   # % — BTS/WHO clinical-concern boundary (matches CLINICAL_SPO2_CONCERN)
    "spo2_very_low": 90.0,   # % — hypoxemia / medical emergency
    "temp_high":     37.8,   # °C skin — possible fever (skin is 4-7°C below core)
}

# --- Signal weights for confidence scoring ---
# Temperature is deliberately weak — it's a supporting signal, not primary.
SIGNAL_WEIGHTS = {
    "hr":                 0.35,
    "hr_stability_score": 0.25,
    "spo2":               0.30,
    "temp":               0.10,
}

# --- Motion detection ---
# Dynamic acceleration magnitude thresholds for activity classification.
MOTION_THRESHOLDS = {
    "still":    0.05,   # dyn_acc_mag < 0.05 → user is still
    "walking":  0.25,   # 0.05 .. 0.25 → light motion
    "active":   0.50,   # 0.25 .. 0.50 → moderate activity
    # > 0.50 → vigorous exercise
}

# --- Fall detection thresholds (3-stage model) ---
FALL_FREEFALL_G         = 0.5    # acc_mag < 0.5g → free-fall detected
FALL_IMPACT_G           = 5.0    # acc_mag > 5.0g → impact spike detected
FALL_ORIENTATION_DEG    = 60.0   # body angle change > 60° → posture changed
FALL_INACTIVITY_G       = 0.05   # dyn_acc_mag < 0.05g → stillness
FALL_INACTIVITY_S       = 3.0    # seconds of post-impact stillness to confirm
FALL_WINDOW_S           = 1.0    # max seconds between free-fall and impact
FALL_CONFIRM_TIMEOUT_S  = 10.0   # max seconds to wait for inactivity confirmation
FALL_ORIENTATION_DELAY_S = 1.5   # seconds post-impact to wait before orientation check (A.1)
FALL_PHYSIO_WINDOW_S    = 30.0   # seconds post-impact to monitor physio corroboration (A.7)
FALL_PHYSIO_HR_SURGE    = 10.0   # bpm increase within 10s → sympathetic response (A.7)
FALL_PHYSIO_HR_DROP     = -15.0  # bpm decrease within 30s → vasovagal/syncope (A.7)
FALL_PHYSIO_TEMP_Z      = -1.0   # temp z-score below this → vasoconstriction (A.7)
FALL_PHYSIO_STILLNESS_S = 3.0    # seconds of post-impact stillness for corroboration (A.7)
FALL_PHYSIO_NO_CHANGE_S = 60.0   # seconds with no physio change → likely false alarm (A.7)

# --- Persistence ---
PERSISTENCE_STRESS_S   = 60    # seconds before promoting to "stress"
PERSISTENCE_DISTRESS_S = 60    # seconds before promoting to "distress"

# --- SQI gate ---
MIN_SQI_FOR_DETECTION = 0.5    # overall SQI below this → skip detection

# --- Audio intelligence (Layer 4) ---
AUDIO_DANGER_THRESHOLD      = 0.25   # fused YAMNet+AST score → danger flag
AUDIO_VOCAL_STRESS_THRESHOLD = 0.6   # Praat stress score → high stress flag
AUDIO_NOISE_DANGEROUS_DB    = 85.0   # dB level considered dangerous
AUDIO_PERSISTENCE_S         = 10     # seconds of sustained danger sound before alert

# Danger sound classes (AudioSet ontology)
DANGER_SOUNDS_ENV: List[str] = [
    "Screaming", "Glass breaking", "Gunshot, gunfire",
    "Explosion", "Crash", "Alarm", "Siren",
    "Fire alarm", "Smoke detector",
]
DANGER_SOUNDS_PHYSIO: List[str] = [
    "Crying, sobbing", "Groan", "Gasp",
]

# Sub-component weights within audio layer
AUDIO_WEIGHTS = {
    "env_danger":   0.50,
    "vocal_stress": 0.30,
    "noise_level":  0.20,
}

# --- EDA/GSR (optional sensor, future hardware) ---
EDA_HIGH_AROUSAL_THRESHOLD = 0.6
EDA_MODERATE_THRESHOLD     = 0.3


# ===========================================================================
# 1. ALERT MESSAGES — STAR principle, human-friendly
# ===========================================================================

# Each category has multiple variants to avoid repetitive messaging.
# Internally structured as STAR (Situation, Task, Action, Result) but
# the user-facing message is natural and conversational.

ALERTS_NORMAL: List[Dict] = [
    {
        "title": "All good",
        "message": "Everything looks steady. You're doing great.",
        "severity": "low",
    },
]

ALERTS_STRESS: List[Dict] = [
    {
        "title": "Take a moment",
        "message": "Your body seems a bit stressed right now. "
                   "Try taking a few slow, deep breaths.",
        "severity": "low",
    },
    {
        "title": "Check in with yourself",
        "message": "You might be feeling a little tense. "
                   "A short pause could help you reset.",
        "severity": "low",
    },
    {
        "title": "Breathe",
        "message": "We're picking up some signs of stress. "
                   "Take a moment — breathe in for 4, out for 6.",
        "severity": "low",
    },
]

ALERTS_STRESS_HR: List[Dict] = [
    {
        "title": "Heart rate is up",
        "message": "Your heart rate is a bit higher than usual. "
                   "If you're not exercising, try to relax and take it easy.",
        "severity": "low",
    },
]

ALERTS_STRESS_SPO2: List[Dict] = [
    {
        "title": "Oxygen dipped slightly",
        "message": "Your oxygen level dipped a little. "
                   "Try some slow, deep breaths — it usually helps.",
        "severity": "medium",
    },
]

ALERTS_DISTRESS: List[Dict] = [
    {
        "title": "We're here for you",
        "message": "We're noticing signs of distress. If you're not feeling "
                   "safe, please reach out to someone you trust.",
        "severity": "high",
    },
    {
        "title": "Something seems off",
        "message": "Multiple signals suggest your body is under strain. "
                   "If you need help, don't hesitate to ask.",
        "severity": "high",
    },
    {
        "title": "Please check in",
        "message": "Your body is showing signs of significant stress. "
                   "Take a moment — and if you're in danger, call for help.",
        "severity": "high",
    },
]

ALERTS_DISTRESS_SPO2: List[Dict] = [
    {
        "title": "Oxygen is low",
        "message": "Your oxygen level is lower than expected. "
                   "If you're feeling lightheaded or short of breath, "
                   "please seek help.",
        "severity": "high",
    },
]

ALERTS_EXERCISE_ELEVATED: List[Dict] = [
    {
        "title": "Active and elevated",
        "message": "Your readings are elevated, but you seem to be moving — "
                   "this is likely from physical activity.",
        "severity": "low",
    },
]

ALERTS_WARMING_UP: List[Dict] = [
    {
        "title": "Getting to know you",
        "message": "We're still learning your baseline. "
                   "Alerts will become more personalized shortly.",
        "severity": "low",
    },
]

ALERTS_TEMP_ELEVATED: List[Dict] = [
    {
        "title": "Slightly warm",
        "message": "Your skin temperature is a bit warmer than usual, "
                   "but there's no need to worry right now.",
        "severity": "low",
    },
]

ALERTS_FALL_DETECTED: List[Dict] = [
    {
        "title": "Possible fall detected",
        "message": "It looks like you may have fallen. "
                   "Stay still — we're checking to make sure you're okay.",
        "severity": "high",
    },
]

ALERTS_FALL_CONFIRMED: List[Dict] = [
    {
        "title": "Fall confirmed",
        "message": "A fall has been detected and you haven't moved. "
                   "If you need help, please call out or press the alert button.",
        "severity": "high",
    },
    {
        "title": "Are you okay?",
        "message": "We detected a fall followed by a period of stillness. "
                   "If you're hurt or need assistance, help is available.",
        "severity": "high",
    },
]

ALERTS_FALL_UNCERTAIN: List[Dict] = [
    {
        "title": "Are you alright?",
        "message": "We detected a possible fall but you seem to be moving a little. "
                   "If you need help, press the alert button or call out.",
        "severity": "medium",
    },
]

ALERTS_FALL_CANCELLED: List[Dict] = [
    {
        "title": "Glad you're okay",
        "message": "We noticed a sudden movement, but you seem to be moving "
                   "normally now. No action needed.",
        "severity": "low",
    },
]

# ---- Audio intelligence alerts (Layer 4) ----

ALERTS_AUDIO_DANGER: List[Dict] = [
    {
        "title": "Dangerous sound detected",
        "message": "We're picking up sounds that may indicate danger nearby. "
                   "Stay alert and move to a safe area if possible.",
        "severity": "high",
    },
    {
        "title": "Loud disturbance detected",
        "message": "We've detected alarming sounds in your environment. "
                   "If you feel unsafe, reach out to someone you trust.",
        "severity": "high",
    },
]

ALERTS_AUDIO_STRESS: List[Dict] = [
    {
        "title": "Voice stress noticed",
        "message": "Your voice patterns suggest you may be feeling stressed. "
                   "Take a moment to breathe if you can.",
        "severity": "medium",
    },
]

ALERTS_AUDIO_NOISE: List[Dict] = [
    {
        "title": "High noise level",
        "message": "The noise around you is quite loud. "
                   "Consider moving to a quieter area or using hearing protection.",
        "severity": "low",
    },
]

ALERTS_AUDIO_NORMAL: List[Dict] = [
    {
        "title": "Audio clear",
        "message": "No concerning sounds detected in your environment.",
        "severity": "low",
    },
]


# ===========================================================================
# 2. MOTION CLASSIFIER
# ===========================================================================

def classify_motion(sample: Dict) -> str:
    """Classify user's motion state from dynamic acceleration magnitude.

    Returns one of: "still", "walking", "active", "exercise"
    """
    dyn = sample.get("dyn_acc_mag", 0.0) or 0.0
    if dyn < MOTION_THRESHOLDS["still"]:
        return "still"
    elif dyn < MOTION_THRESHOLDS["walking"]:
        return "walking"
    elif dyn < MOTION_THRESHOLDS["active"]:
        return "active"
    else:
        return "exercise"


# ===========================================================================
# 3. SIGNAL FLAGGER
# ===========================================================================

@dataclass
class SignalFlags:
    """Per-signal boolean flags + the score that triggered them."""
    hr:                 bool = False
    hr_stability_score: bool = False
    spo2:               bool = False
    temp:               bool = False
    temp_drop:          bool = False   # vasoconstriction / distress (stronger than temp elevation)

    hr_score:                 float = 0.0
    hr_stability_score_score: float = 0.0
    spo2_score:               float = 0.0
    temp_score:               float = 0.0
    temp_drop_score:          float = 0.0

    source: str = "none"   # "zscore", "absolute", "clinical_flag"

    @property
    def count(self) -> int:
        return sum([self.hr, self.hr_stability_score, self.spo2,
                    self.temp, self.temp_drop])

    @property
    def triggered(self) -> List[str]:
        out = []
        if self.hr:                 out.append("hr")
        if self.hr_stability_score: out.append("hr_stability_score")
        if self.spo2:               out.append("spo2")
        if self.temp:               out.append("temp")
        if self.temp_drop:          out.append("temp_drop")
        return out

    def weighted_score(self) -> float:
        """Weighted confidence score from triggered signals.

        temp_drop contributes with 1.5x the normal temp weight
        (vasoconstriction is a stronger distress signal than fever).
        """
        total = 0.0
        if self.hr:
            total += SIGNAL_WEIGHTS["hr"] * min(1.0, self.hr_score)
        if self.hr_stability_score:
            total += SIGNAL_WEIGHTS["hr_stability_score"] * min(1.0, self.hr_stability_score_score)
        if self.spo2:
            total += SIGNAL_WEIGHTS["spo2"] * min(1.0, self.spo2_score)
        if self.temp:
            total += SIGNAL_WEIGHTS["temp"] * min(1.0, self.temp_score)
        if self.temp_drop:
            # temp_drop is a stronger distress signal — 1.5x temp weight
            total += SIGNAL_WEIGHTS["temp"] * 1.5 * min(1.0, self.temp_drop_score)
        return total


def flag_signals(pipeline_output: Dict) -> SignalFlags:
    """Evaluate z-scores + absolute thresholds to produce per-signal flags.

    Strategy:
        1. If baseline_ready → use z-score thresholds (personalized)
        2. Always check absolute thresholds (universal safety net)
        3. Check clinical_flags from the preprocessing pipeline
        4. Temperature is treated as a weak/supporting signal
    """
    sample  = pipeline_output["sample"]
    zscores = pipeline_output["zscores"]
    baseline_ready = pipeline_output["baseline_ready"]

    flags = SignalFlags()

    # --- Z-score based flags (only when baseline is ready) ---
    if baseline_ready:
        flags.source = "zscore"

        # HR: elevated z-score → stress
        z_hr = zscores.get("hr", 0.0)
        if z_hr > ZSCORE_THRESHOLDS["hr"]:
            flags.hr = True
            flags.hr_score = (z_hr - ZSCORE_THRESHOLDS["hr"]) / 2.0 + 0.5

        # HR stability: suppressed z-score → stress (INVERTED direction)
        z_hrv = zscores.get("hr_stability_score", 0.0)
        if z_hrv < ZSCORE_THRESHOLDS["hr_stability_score"]:
            flags.hr_stability_score = True
            flags.hr_stability_score_score = (
                abs(z_hrv - ZSCORE_THRESHOLDS["hr_stability_score"]) / 2.0 + 0.5
            )

        # SpO2: suppressed z-score → distress (INVERTED direction)
        # Clinical floor guard: only flag z-score deviation when SpO2 is
        # actually approaching clinical concern (< 96%).  Values 96-100%
        # are solidly normal and should not alarm the user even if they
        # deviate from a high personal baseline.
        z_spo2 = zscores.get("spo2", 0.0)
        spo2_val_for_guard = sample.get("spo2", 100.0)
        if z_spo2 < ZSCORE_THRESHOLDS["spo2"] and spo2_val_for_guard < 96.0:
            flags.spo2 = True
            flags.spo2_score = (
                abs(z_spo2 - ZSCORE_THRESHOLDS["spo2"]) / 2.0 + 0.5
            )

        # Temp: elevated z-score → weak supporting signal
        z_temp = zscores.get("temp", 0.0)
        if z_temp > ZSCORE_THRESHOLDS["temp"]:
            flags.temp = True
            flags.temp_score = (z_temp - ZSCORE_THRESHOLDS["temp"]) / 3.0 + 0.3

        # Temp drop: vasoconstriction / distress (stronger than elevation)
        if z_temp < ZSCORE_THRESHOLDS["temp_low"]:
            flags.temp_drop = True
            flags.temp_drop_score = (
                abs(z_temp - ZSCORE_THRESHOLDS["temp_low"]) / 2.0 + 0.5
            )

    # --- Absolute threshold flags (always active) ---
    hr = sample.get("hr")
    if hr is not None:
        if hr >= ABSOLUTE_THRESHOLDS["hr_very_high"]:
            flags.hr = True
            flags.hr_score = max(flags.hr_score, 0.9)
            flags.source = "absolute"
        elif hr >= ABSOLUTE_THRESHOLDS["hr_high"] and not flags.hr:
            flags.hr = True
            flags.hr_score = max(flags.hr_score, 0.6)
            flags.source = "absolute"

    spo2 = sample.get("spo2")
    if spo2 is not None:
        if spo2 <= ABSOLUTE_THRESHOLDS["spo2_very_low"]:
            flags.spo2 = True
            flags.spo2_score = max(flags.spo2_score, 0.9)
            flags.source = "absolute"
        elif spo2 <= ABSOLUTE_THRESHOLDS["spo2_low"] and not flags.spo2:
            flags.spo2 = True
            flags.spo2_score = max(flags.spo2_score, 0.6)
            flags.source = "absolute"

    temp = sample.get("temp")
    if temp is not None:
        if temp >= ABSOLUTE_THRESHOLDS["temp_high"] and not flags.temp:
            flags.temp = True
            flags.temp_score = max(flags.temp_score, 0.3)

    # --- Clinical flags from preprocessing pipeline ---
    # Note: spo2_clinical_concern is no longer handled here because
    # it is redundant with the absolute threshold check (spo2_low = 94.0
    # matches CLINICAL_SPO2_CONCERN in the preprocessing pipeline).
    clinical = sample.get("clinical_flags", [])
    for cf in clinical:
        if "spo2_hypoxemia" in cf and not flags.spo2:
            flags.spo2 = True
            flags.spo2_score = max(flags.spo2_score, 0.85)
            flags.source = "clinical_flag"
        elif "elevated_skin_temp" in cf and not flags.temp:
            flags.temp = True
            flags.temp_score = max(flags.temp_score, 0.3)

    return flags


# ===========================================================================
# 4. PERSISTENCE TRACKER
# ===========================================================================

class PersistenceTracker:
    """
    Tracks how long a condition has been continuously active.

    A single-frame spike should NOT trigger an alert. The condition must
    persist for a configurable duration before we promote the state.

    Strategy:
        - Each evaluation, if flags are active → increment duration
        - If flags clear → reset to zero (with a 1-frame grace period
          to handle intermittent sensor noise)
        - Report the current persistence duration in seconds
    """

    def __init__(self, grace_frames: int = 2):
        self.stress_start: Optional[float] = None
        self.distress_start: Optional[float] = None
        self.grace_frames = grace_frames
        self._stress_grace = 0
        self._distress_grace = 0

    def update_stress(self, active: bool, now: float) -> float:
        """Returns seconds of continuous stress. 0 if not active."""
        if active:
            if self.stress_start is None:
                self.stress_start = now
            self._stress_grace = 0
            return now - self.stress_start
        else:
            self._stress_grace += 1
            if self._stress_grace > self.grace_frames:
                self.stress_start = None
            return (now - self.stress_start) if self.stress_start else 0.0

    def update_distress(self, active: bool, now: float) -> float:
        """Returns seconds of continuous distress. 0 if not active."""
        if active:
            if self.distress_start is None:
                self.distress_start = now
            self._distress_grace = 0
            return now - self.distress_start
        else:
            self._distress_grace += 1
            if self._distress_grace > self.grace_frames:
                self.distress_start = None
            return (now - self.distress_start) if self.distress_start else 0.0

    def reset(self):
        self.stress_start = None
        self.distress_start = None
        self._stress_grace = 0
        self._distress_grace = 0


# ===========================================================================
# 5. FALL DETECTOR — 3-stage state machine
# ===========================================================================

class FallDetector:
    """
    Three-stage fall detection model using accelerometer data only.

    State machine:
        IDLE → FREEFALL_DETECTED → IMPACT_DETECTED → CONFIRMED / CANCELLED

    Stage 1 — Impact Detection (threshold-based)
        • Free-fall:  acc_mag drops below 0.5g for 3 consecutive samples
        • Impact:     acc_mag spikes above 5.0g (sudden deceleration on landing)
        • Both must occur within a 1-second window

    Stage 2 — Orientation Change
        • After impact, check if body orientation changed significantly
        • Estimated from accelerometer axis ratios (gravity vector shift)
        • A ~60°+ change from the pre-fall orientation → likely fell down

    Stage 3 — Post-Fall Three-Outcome Monitoring
        • Stillness >= 3s (dyn_acc < 0.05g) → CONFIRMED (unconscious)
        • Vigorous activity (dyn_acc > 0.5g sustained 3s+) → CANCELLED
        • Weak/moderate movement 10s+ → CONFIRMED_UNCERTAIN

    All thresholds are configurable via module-level constants.
    """

    # States
    IDLE              = "idle"
    FREEFALL_DETECTED = "freefall_detected"
    IMPACT_DETECTED   = "impact_detected"
    CONFIRMED         = "confirmed"
    CANCELLED         = "cancelled"

    # Additional post-impact states for three-outcome logic (A.4)
    CONFIRMED_UNCERTAIN = "confirmed_uncertain"

    def __init__(self):
        self.state: str = self.IDLE
        self._freefall_time: Optional[float] = None
        self._impact_time: Optional[float] = None
        self._pre_fall_orientation: Optional[Tuple[float, float, float]] = None
        self._pre_fall_gravity: Optional[Tuple[float, float, float]] = None  # A.1: gravity vector before fall
        self._orientation_checked: bool = False  # A.1: deferred orientation check flag
        self._post_impact_still_start: Optional[float] = None
        self._post_impact_vigorous_start: Optional[float] = None  # A.4: vigorous activity timer
        self._post_impact_weak_start: Optional[float] = None      # A.4: weak/moderate movement timer
        self._last_orientation: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        self._last_gravity: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # A.1: track last gravity vector
        self._result_hold_until: Optional[float] = None  # hold confirmed/cancelled
        self._freefall_consecutive: int = 0  # A.3: consecutive sub-0.5g samples

        # A.7: Physiological corroboration state
        self._impact_hr: Optional[float] = None       # HR at time of impact
        self._physio_window_end: Optional[float] = None  # end of corroboration window
        self._physio_confidence_boost: float = 0.0     # accumulated confidence boost
        self._physio_stillness_start: Optional[float] = None  # post-impact stillness tracker
        self._physio_no_change_start: Optional[float] = None  # no-change tracker
        self._physio_last_hr: Optional[float] = None
        self._physio_last_temp_z: Optional[float] = None
        self._physio_corroboration_reasons: List[str] = []

    def update(self, sample: Dict, now: float,
              gravity_x: float = 0.0, gravity_y: float = 0.0,
              gravity_z: float = 1.0) -> Dict:
        """
        Process one frame and return fall detection status.

        Args:
            sample:    preprocessed frame with acc_mag, dyn_acc_mag, acc_x/y/z
            now:       current timestamp (seconds)
            gravity_x: low-pass gravity estimate x (from ProcessedFrame)
            gravity_y: low-pass gravity estimate y (from ProcessedFrame)
            gravity_z: low-pass gravity estimate z (from ProcessedFrame)

        Returns:
            {
                "detected":    bool,   # True if fall confirmed
                "stage":       str,    # current stage name
                "confidence":  float,  # 0.0..1.0
                "description": str,    # human-readable status
            }
        """
        acc_mag = sample.get("acc_mag", 1.0) or 1.0
        dyn_acc = sample.get("dyn_acc_mag", 0.0) or 0.0
        acc_x = sample.get("acc_x", 0.0) or 0.0
        acc_y = sample.get("acc_y", 0.0) or 0.0
        acc_z = sample.get("acc_z", 1.0) or 1.0
        current_orientation = (acc_x, acc_y, acc_z)
        current_gravity = (gravity_x, gravity_y, gravity_z)

        # --- Hold confirmed/cancelled/uncertain result briefly, then reset ---
        if self.state in (self.CONFIRMED, self.CANCELLED, self.CONFIRMED_UNCERTAIN):
            if self._result_hold_until and now < self._result_hold_until:
                if self.state == self.CONFIRMED:
                    return self._make_result(True, self.CONFIRMED, 0.95,
                                             "Fall confirmed — post-fall inactivity detected")
                elif self.state == self.CONFIRMED_UNCERTAIN:
                    return self._make_result(True, self.CONFIRMED_UNCERTAIN, 0.55,
                                             "Fall detected — weak movement, uncertain severity")
                else:
                    return self._make_result(False, self.CANCELLED, 0.0,
                                             "Movement resumed — false alarm cleared")
            # Hold expired, reset
            self._reset()

        # === STAGE 1a: Detect free-fall (requires 3 consecutive sub-0.5g samples) ===
        if self.state == self.IDLE:
            if acc_mag < FALL_FREEFALL_G:
                self._freefall_consecutive += 1
                if self._freefall_consecutive >= 3:
                    # Save orientation BEFORE the fall (from the previous frame)
                    self._pre_fall_orientation = self._last_orientation
                    # A.1: Save pre-fall GRAVITY vector (not raw acc)
                    self._pre_fall_gravity = self._last_gravity
                    self.state = self.FREEFALL_DETECTED
                    self._freefall_time = now
                    self._freefall_consecutive = 0
                    return self._make_result(False, "freefall", 0.2,
                                             "Free-fall detected — watching for impact")
            else:
                self._freefall_consecutive = 0
            # Only update last orientation for non-freefall frames
            self._last_orientation = current_orientation
            self._last_gravity = current_gravity

        # === STAGE 1b: Detect impact after free-fall ===
        elif self.state == self.FREEFALL_DETECTED:
            # Timeout: if no impact within window, cancel
            if now - self._freefall_time > FALL_WINDOW_S:
                self._reset()
                return self._make_result(False, "none", 0.0,
                                         "Free-fall timeout — no impact detected")

            if acc_mag > FALL_IMPACT_G:
                self._impact_time = now
                # A.1: At impact, raw acc is dominated by dynamic force (3-5g),
                # NOT gravity. Defer orientation check until dynamic motion decays
                # (1-2 seconds post-impact). Move directly to IMPACT_DETECTED.
                self.state = self.IMPACT_DETECTED
                self._orientation_checked = False
                self._post_impact_still_start = None
                # A.7: Record HR at time of impact for physio corroboration
                self._impact_hr = sample.get("hr")
                self._physio_window_end = now + FALL_PHYSIO_WINDOW_S
                self._physio_confidence_boost = 0.0
                self._physio_corroboration_reasons = []
                self._physio_no_change_start = now
                self._physio_last_hr = self._impact_hr
                self._physio_last_temp_z = None
                return self._make_result(False, "impact", 0.5,
                                         "Impact detected — deferring orientation check "
                                         "until dynamic motion decays")

        # === STAGE 3: Post-fall three-outcome monitoring ===
        # A.1: Deferred orientation check — wait 1-2s post-impact for dynamic
        # motion to decay, then compare pre-fall gravity vector with current
        # gravity vector (not raw acc which is dominated by impact force).
        # (a) stillness >= 3s (dyn_acc < 0.05g) → CONFIRMED (unconscious)
        # (b) vigorous activity (dyn_acc > 0.5g sustained 3s+) → CANCELLED
        # (c) weak/moderate movement for 10s+ → CONFIRMED_UNCERTAIN
        elif self.state == self.IMPACT_DETECTED:
            # A.1: Deferred orientation check using gravity vectors
            if not self._orientation_checked:
                time_since_impact = now - self._impact_time
                if time_since_impact >= FALL_ORIENTATION_DELAY_S:
                    # Dynamic motion has decayed — now compare gravity vectors
                    angle_change = self._compute_angle_change(
                        self._pre_fall_gravity, current_gravity
                    )
                    self._orientation_checked = True
                    if angle_change < FALL_ORIENTATION_DEG:
                        # No significant orientation change — likely a bump, not a fall
                        self._reset()
                        return self._make_result(False, "none", 0.0,
                                                 f"Post-impact gravity orientation unchanged "
                                                 f"({angle_change:.0f}°) — not a fall")
                    # Orientation changed — continue monitoring
                else:
                    # Still waiting for dynamic motion to decay
                    return self._make_result(False, "impact", 0.5,
                                             f"Impact detected — waiting {time_since_impact:.1f}s/"
                                             f"{FALL_ORIENTATION_DELAY_S}s for orientation check")

            # Timeout: if too long since impact without confirmation, cancel
            if now - self._impact_time > FALL_CONFIRM_TIMEOUT_S:
                # If there was weak movement the whole time, mark uncertain
                if (self._post_impact_weak_start is not None
                        and now - self._post_impact_weak_start >= 10.0):
                    self.state = self.CONFIRMED_UNCERTAIN
                    self._result_hold_until = now + 10.0
                    return self._make_result(True, self.CONFIRMED_UNCERTAIN, 0.55,
                                             "Fall detected — weak movement only, uncertain severity")
                self.state = self.CANCELLED
                self._result_hold_until = now + 3.0
                return self._make_result(False, self.CANCELLED, 0.0,
                                         "Confirmation timeout — movement detected")

            if dyn_acc < FALL_INACTIVITY_G:
                # (a) User is still — track stillness
                if self._post_impact_still_start is None:
                    self._post_impact_still_start = now
                still_duration = now - self._post_impact_still_start
                # Reset vigorous timer (no longer vigorous)
                self._post_impact_vigorous_start = None

                if still_duration >= FALL_INACTIVITY_S:
                    # === CONFIRMED FALL (unconscious/incapacitated) ===
                    self.state = self.CONFIRMED
                    self._result_hold_until = now + 10.0
                    return self._make_result(True, self.CONFIRMED, 0.95,
                                             f"Fall confirmed — {still_duration:.1f}s of post-impact inactivity")
                else:
                    return self._make_result(False, "inactivity_check", 0.7,
                                             f"Post-impact stillness: {still_duration:.1f}s / {FALL_INACTIVITY_S}s")

            elif dyn_acc > 0.5:
                # (b) Vigorous activity — person got up, clearly fine
                self._post_impact_still_start = None
                self._post_impact_weak_start = None
                if self._post_impact_vigorous_start is None:
                    self._post_impact_vigorous_start = now
                vigorous_duration = now - self._post_impact_vigorous_start

                if vigorous_duration >= 3.0:
                    # Sustained vigorous activity → CANCELLED
                    self.state = self.CANCELLED
                    self._result_hold_until = now + 3.0
                    return self._make_result(False, self.CANCELLED, 0.0,
                                             "Vigorous movement after impact — person recovered")
                else:
                    return self._make_result(False, "impact", 0.5,
                                             f"Vigorous movement post-impact: {vigorous_duration:.1f}s / 3.0s")

            else:
                # (c) Weak/moderate movement (0.05g <= dyn_acc <= 0.5g)
                self._post_impact_still_start = None
                self._post_impact_vigorous_start = None
                if self._post_impact_weak_start is None:
                    self._post_impact_weak_start = now
                weak_duration = now - self._post_impact_weak_start

                if weak_duration >= 10.0:
                    # Sustained weak movement → CONFIRMED_UNCERTAIN
                    self.state = self.CONFIRMED_UNCERTAIN
                    self._result_hold_until = now + 10.0
                    return self._make_result(True, self.CONFIRMED_UNCERTAIN, 0.55,
                                             f"Fall detected — {weak_duration:.1f}s of weak movement, uncertain severity")
                else:
                    return self._make_result(False, "impact", 0.5,
                                             f"Weak movement post-impact: {weak_duration:.1f}s — still monitoring")

        # --- Default: IDLE, no fall activity ---
        self._last_orientation = current_orientation
        self._last_gravity = current_gravity
        return self._make_result(False, "none", 0.0, "No fall activity")

    def check_physio_corroboration(
        self,
        hr: Optional[float],
        temp_z: float,
        spo2_z: float,
        dyn_acc: float,
        now: float,
    ) -> Tuple[float, List[str]]:
        """
        A.7: Physiological corroboration after an accelerometer-flagged impact.

        Called each frame during the 30-second post-impact corroboration window.
        Accumulates confidence adjustments based on physiological signals that
        corroborate (or contradict) a real fall.

        Args:
            hr:       current heart rate (bpm)
            temp_z:   current temperature z-score
            spo2_z:   current SpO2 z-score
            dyn_acc:  current dynamic acceleration magnitude
            now:      current timestamp

        Returns:
            (confidence_adjustment, list_of_reasons)
            Positive adjustment = more confident it's a real fall.
            Negative adjustment = less confident (likely false alarm).
        """
        if self._impact_time is None or self._physio_window_end is None:
            return 0.0, []

        # Outside corroboration window
        if now > self._physio_window_end:
            return self._physio_confidence_boost, list(self._physio_corroboration_reasons)

        time_since_impact = now - self._impact_time
        reasons = []
        boost = 0.0

        # --- HR surge (sympathetic response): HR delta >= +10 bpm within 10s ---
        if (hr is not None and self._impact_hr is not None
                and time_since_impact <= 10.0):
            hr_delta = hr - self._impact_hr
            if hr_delta >= FALL_PHYSIO_HR_SURGE:
                boost += 0.10
                reasons.append(f"HR surge +{hr_delta:.0f} bpm (sympathetic response)")

        # --- HR drop (vasovagal/syncope): HR delta <= -15 bpm within 30s ---
        if (hr is not None and self._impact_hr is not None
                and time_since_impact <= 30.0):
            hr_delta = hr - self._impact_hr
            if hr_delta <= FALL_PHYSIO_HR_DROP:
                boost += 0.15
                reasons.append(f"HR drop {hr_delta:.0f} bpm (vasovagal/syncope)")

        # --- Skin temp drop (vasoconstriction): temp z < -1.0 ---
        if temp_z < FALL_PHYSIO_TEMP_Z:
            boost += 0.05
            reasons.append(f"Skin temp drop (z={temp_z:.1f}, vasoconstriction)")

        # --- Post-impact stillness >= 3s ---
        if dyn_acc < FALL_INACTIVITY_G:
            if self._physio_stillness_start is None:
                self._physio_stillness_start = now
            still_duration = now - self._physio_stillness_start
            if still_duration >= FALL_PHYSIO_STILLNESS_S:
                boost += 0.10
                reasons.append(f"Post-impact stillness {still_duration:.1f}s")
        else:
            self._physio_stillness_start = None

        # --- No change in anything over 60s → likely false alarm ---
        if hr is not None and self._physio_last_hr is not None:
            hr_changed = abs(hr - self._physio_last_hr) > 3.0
            temp_changed = (self._physio_last_temp_z is not None
                            and abs(temp_z - self._physio_last_temp_z) > 0.5)
            if hr_changed or temp_changed:
                self._physio_no_change_start = now

        if (self._physio_no_change_start is not None
                and now - self._physio_no_change_start >= FALL_PHYSIO_NO_CHANGE_S):
            boost -= 0.20
            reasons.append("No physiological change for 60s — likely false alarm")

        # Track last values for change detection
        if hr is not None:
            self._physio_last_hr = hr
        self._physio_last_temp_z = temp_z

        # Accumulate (don't let it go below -0.3 or above +0.3)
        self._physio_confidence_boost = max(-0.3, min(0.3, self._physio_confidence_boost + boost))
        self._physio_corroboration_reasons = reasons

        return self._physio_confidence_boost, reasons

    def _compute_angle_change(
        self,
        before: Optional[Tuple[float, float, float]],
        after: Tuple[float, float, float],
    ) -> float:
        """Compute angle between two orientation vectors (degrees).

        Uses the gravity vector from accelerometer axes to estimate
        body orientation change. The dot product of normalized vectors
        gives cos(theta).
        """
        import math

        if before is None:
            return 0.0

        # Normalize both vectors
        def _norm(v):
            mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            if mag < 1e-6:
                return (0.0, 0.0, 1.0)
            return (v[0]/mag, v[1]/mag, v[2]/mag)

        b = _norm(before)
        a = _norm(after)

        # Dot product → angle
        dot = b[0]*a[0] + b[1]*a[1] + b[2]*a[2]
        dot = max(-1.0, min(1.0, dot))  # clamp for numerical stability
        angle_rad = math.acos(dot)
        return math.degrees(angle_rad)

    def _make_result(self, detected: bool, stage: str,
                     confidence: float, description: str) -> Dict:
        return {
            "detected": detected,
            "stage": stage,
            "confidence": round(confidence, 3),
            "description": description,
        }

    def _reset(self):
        self.state = self.IDLE
        self._freefall_time = None
        self._impact_time = None
        self._pre_fall_orientation = None
        self._pre_fall_gravity = None
        self._orientation_checked = False
        self._post_impact_still_start = None
        self._post_impact_vigorous_start = None
        self._post_impact_weak_start = None
        self._result_hold_until = None
        self._freefall_consecutive = 0
        # A.7: Reset physio corroboration state
        self._impact_hr = None
        self._physio_window_end = None
        self._physio_confidence_boost = 0.0
        self._physio_stillness_start = None
        self._physio_no_change_start = None
        self._physio_last_hr = None
        self._physio_last_temp_z = None
        self._physio_corroboration_reasons = []


# ===========================================================================
# 6. ALERT SELECTOR
# ===========================================================================

def _pick_alert(pool: List[Dict], seed: int = 0) -> Dict:
    """Deterministically pick from the pool to avoid repeating the same message."""
    return pool[seed % len(pool)]


def select_alert(
    state: str,
    flags: SignalFlags,
    motion_state: str,
    baseline_ready: bool,
    persistence_s: float,
    cycle: int = 0,
) -> Dict:
    """Choose the most appropriate alert for the current state.

    Prioritizes specificity:
        1. If exercise is detected and HR is elevated → exercise-specific
        2. If SpO2 is the dominant signal → SpO2-specific
        3. Otherwise → general stress/distress pool
    """
    # --- Not ready yet ---
    if not baseline_ready and state == "normal":
        return _pick_alert(ALERTS_WARMING_UP)

    # --- Normal ---
    if state == "normal":
        # Check if temp alone is elevated (advisory, not stress)
        if flags.temp and flags.count == 1 and not flags.temp_drop:
            return _pick_alert(ALERTS_TEMP_ELEVATED)
        return _pick_alert(ALERTS_NORMAL)

    # --- Exercise override ---
    # If the user is clearly exercising, elevated HR is expected — don't
    # alarm them. Only escalate if SpO2 is ALSO flagged.
    if motion_state in ("active", "exercise"):
        if flags.hr and not flags.spo2 and flags.count <= 2:
            return _pick_alert(ALERTS_EXERCISE_ELEVATED)

    # --- Stress ---
    if state == "stress":
        if flags.spo2 and not flags.hr:
            return _pick_alert(ALERTS_STRESS_SPO2, cycle)
        if flags.hr and not flags.spo2:
            return _pick_alert(ALERTS_STRESS_HR, cycle)
        return _pick_alert(ALERTS_STRESS, cycle)

    # --- Distress ---
    if state == "distress":
        if flags.spo2 and flags.spo2_score >= 0.8:
            return _pick_alert(ALERTS_DISTRESS_SPO2, cycle)
        return _pick_alert(ALERTS_DISTRESS, cycle)

    return _pick_alert(ALERTS_NORMAL)


# ===========================================================================
# 6. THE ENGINE — stateful evaluator
# ===========================================================================

class DistressEngine:
    """
    Stateful rule-based physiological distress detection engine.

    Usage:
        engine = DistressEngine()
        for pipeline_output in stream:
            result = engine.evaluate(pipeline_output)
            show_to_user(result["alert"])

    Decision flow:
        1. Check frame acceptance + SQI gate
        2. Classify motion (still / walking / active / exercise)
        3. Flag signals (z-score + absolute + clinical)
        4. Apply motion context (suppress HR-only flags during exercise)
        5. Determine raw state: normal / stress / distress
        6. Apply persistence requirement (60s sustained before promoting)
        7. Compute confidence score
        8. Select appropriate alert message
    """

    def __init__(self):
        self.persistence = PersistenceTracker(grace_frames=2)
        self.fall_detector = FallDetector()
        self._cycle = 0   # for alert message rotation
        self._last_state = "normal"
        self._state_history: Deque[str] = deque(maxlen=120)  # ~2 min at 1 Hz
        self._spo2_emergency_count: int = 0  # sustain gate for SpO2 <= 90% bypass

    def evaluate(self, pipeline_output: Dict) -> Dict:
        """Evaluate one preprocessed frame and return a detection result."""
        self._cycle += 1
        now = time.time()

        sample   = pipeline_output["sample"]
        zscores  = pipeline_output["zscores"]
        baseline_ready = pipeline_output["baseline_ready"]
        window   = pipeline_output["window"]

        # ------------------------------------------------------------------
        # Gate: rejected frames or very low SQI → skip detection
        # ------------------------------------------------------------------
        if not sample.get("accepted", False):
            self.persistence.reset()
            return self._build_result(
                state="normal",
                confidence=0.0,
                signals=[],
                alert=_pick_alert(ALERTS_NORMAL),
                flags=SignalFlags(),
                persistence_s=0,
                motion_state="unknown",
                skipped=True,
                skip_reason="frame_rejected",
            )

        sqi_overall = sample.get("sqi", {}).get("overall", 0.0)
        if sqi_overall < MIN_SQI_FOR_DETECTION:
            self.persistence.reset()
            return self._build_result(
                state="normal",
                confidence=0.0,
                signals=[],
                alert=_pick_alert(ALERTS_NORMAL),
                flags=SignalFlags(),
                persistence_s=0,
                motion_state="unknown",
                skipped=True,
                skip_reason="low_sqi",
            )

        # ------------------------------------------------------------------
        # Step 1: Motion classification
        # ------------------------------------------------------------------
        motion_state = classify_motion(sample)

        # ------------------------------------------------------------------
        # Step 1b: Fall detection (3-stage model)
        # A.1: Pass gravity vector components from ProcessedFrame so the
        # orientation check uses the low-pass gravity estimate (not raw
        # acc which is dominated by dynamic force at impact).
        # ------------------------------------------------------------------
        gravity_x = sample.get("gravity_x", 0.0) or 0.0
        gravity_y = sample.get("gravity_y", 0.0) or 0.0
        gravity_z = sample.get("gravity_z", 1.0) or 1.0
        fall_result = self.fall_detector.update(
            sample, now,
            gravity_x=gravity_x,
            gravity_y=gravity_y,
            gravity_z=gravity_z,
        )

        # A.7: Physiological corroboration during post-impact window
        if (self.fall_detector.state in (
                FallDetector.IMPACT_DETECTED,
                FallDetector.CONFIRMED,
                FallDetector.CONFIRMED_UNCERTAIN)
                and self.fall_detector._physio_window_end is not None
                and now <= self.fall_detector._physio_window_end):
            temp_z = zscores.get("temp", 0.0)
            spo2_z = zscores.get("spo2", 0.0)
            dyn_acc = sample.get("dyn_acc_mag", 0.0) or 0.0
            physio_boost, physio_reasons = self.fall_detector.check_physio_corroboration(
                hr=sample.get("hr"),
                temp_z=temp_z,
                spo2_z=spo2_z,
                dyn_acc=dyn_acc,
                now=now,
            )
            # Apply physio corroboration boost to fall confidence
            if physio_boost != 0.0:
                fall_result["confidence"] = round(
                    max(0.0, min(1.0, fall_result["confidence"] + physio_boost)), 3
                )
            if physio_reasons:
                fall_result["description"] += " | Physio: " + "; ".join(physio_reasons)

        # ------------------------------------------------------------------
        # Step 2: Flag signals
        # ------------------------------------------------------------------
        flags = flag_signals(pipeline_output)

        # ------------------------------------------------------------------
        # Step 3: Motion-aware suppression
        # ------------------------------------------------------------------
        # During exercise, elevated HR alone is expected and NOT stressful.
        # Only flag HR during exercise if it's accompanied by SpO2 drop
        # or the HR is extremely high (> hr_very_high).
        if motion_state in ("active", "exercise"):
            hr = sample.get("hr")
            if (
                flags.hr
                and not flags.spo2
                and (hr is not None and hr < ABSOLUTE_THRESHOLDS["hr_very_high"])
            ):
                flags.hr = False
                flags.hr_score = 0.0

        # ------------------------------------------------------------------
        # Step 4: Raw state determination (weighted-score gating)
        # ------------------------------------------------------------------
        # Use weighted_score instead of simple flag count to account for
        # signal importance (HR=0.35, SpO2=0.30, HRV=0.25, Temp=0.10).
        w_score = flags.weighted_score()
        raw_state = "normal"
        if w_score > 0.40:
            raw_state = "distress"
        elif w_score > 0.15:
            # Temp elevation alone is a "supporting" signal — not enough for stress
            if flags.temp and flags.count == 1 and not flags.temp_drop:
                raw_state = "normal"
            else:
                raw_state = "stress"

        # ------------------------------------------------------------------
        # Step 5: Persistence
        # ------------------------------------------------------------------
        stress_duration   = self.persistence.update_stress(
            raw_state in ("stress", "distress"), now
        )
        distress_duration = self.persistence.update_distress(
            raw_state == "distress", now
        )

        # Promote state only after sufficient persistence
        final_state = "normal"
        persistence_s = 0.0

        if raw_state == "distress":
            if distress_duration >= PERSISTENCE_DISTRESS_S:
                final_state = "distress"
                persistence_s = distress_duration
            elif stress_duration >= PERSISTENCE_STRESS_S:
                # Not enough distress persistence, but stress has persisted
                final_state = "stress"
                persistence_s = stress_duration
            else:
                # Still building up — show as "normal" but with debug info
                final_state = "normal"
                persistence_s = distress_duration

        elif raw_state == "stress":
            if stress_duration >= PERSISTENCE_STRESS_S:
                final_state = "stress"
                persistence_s = stress_duration
            else:
                final_state = "normal"
                persistence_s = stress_duration

        # Exception: absolute emergency signals bypass persistence
        # SpO2 <= 90% for 10 consecutive frames → emergency escalation
        # HR > 150 → immediate escalation (non-exercise)
        spo2_val = sample.get("spo2")
        hr_val   = sample.get("hr")
        if spo2_val is not None and spo2_val <= ABSOLUTE_THRESHOLDS["spo2_very_low"]:
            self._spo2_emergency_count += 1
            if self._spo2_emergency_count >= 10:
                final_state = "distress"
                persistence_s = max(persistence_s, 1.0)
        else:
            self._spo2_emergency_count = 0
        if hr_val is not None and hr_val >= ABSOLUTE_THRESHOLDS["hr_very_high"]:
            if motion_state not in ("active", "exercise"):
                final_state = max(final_state, "stress", key=_state_rank)
                persistence_s = max(persistence_s, 1.0)

        # ------------------------------------------------------------------
        # Step 6: Confidence scoring
        # ------------------------------------------------------------------
        confidence = self._compute_confidence(
            flags, final_state, persistence_s, sqi_overall, window
        )

        # ------------------------------------------------------------------
        # Step 7: Alert selection
        # ------------------------------------------------------------------
        alert = select_alert(
            final_state, flags, motion_state,
            baseline_ready, persistence_s, self._cycle
        )

        # ------------------------------------------------------------------
        # Step 8: Fall detection alert override
        # ------------------------------------------------------------------
        # If a fall is confirmed, override the alert and escalate state
        if fall_result["detected"]:
            if fall_result["stage"] == "confirmed_uncertain":
                # Uncertain fall — flag but defer to physiological corroboration
                alert = _pick_alert(ALERTS_FALL_UNCERTAIN, self._cycle)
                confidence = max(confidence, 0.55)
                # Only escalate if physiology already shows stress/distress
                if final_state in ("stress", "distress"):
                    final_state = "distress"
            else:
                alert = _pick_alert(ALERTS_FALL_CONFIRMED, self._cycle)
                final_state = "distress"
                confidence = max(confidence, 0.9)
        elif fall_result["stage"] in ("freefall", "impact", "inactivity_check"):
            alert = _pick_alert(ALERTS_FALL_DETECTED)
        elif fall_result["stage"] == "cancelled":
            alert = _pick_alert(ALERTS_FALL_CANCELLED)

        # Track history
        self._state_history.append(final_state)
        self._last_state = final_state

        return self._build_result(
            state=final_state,
            confidence=confidence,
            signals=flags.triggered,
            alert=alert,
            flags=flags,
            persistence_s=persistence_s,
            motion_state=motion_state,
            fall_result=fall_result,
        )

    def _compute_confidence(
        self,
        flags: SignalFlags,
        state: str,
        persistence_s: float,
        sqi_overall: float,
        window: Dict,
    ) -> float:
        """
        Confidence = f(signal_score, persistence, sqi, window_size).

        Components:
            1. Signal strength:  weighted score from flagged signals (0..1)
            2. Persistence:      longer duration → higher confidence
            3. SQI:              higher quality → higher confidence
            4. Window coverage:  more frames → more stable statistics

        For "normal" state, confidence represents how confident we are
        that everything IS normal (high = definitely normal).
        """
        if state == "normal":
            # Confidence in normality: high SQI + no flags + adequate window
            window_n = window.get("window_n", 0)
            window_factor = min(1.0, window_n / 20.0)  # confident after ~20 frames
            return min(1.0, sqi_overall * 0.6 + window_factor * 0.4)

        # For stress/distress: confidence in the detection
        signal_score = flags.weighted_score()

        # Persistence factor: ramps from 0.3 → 1.0 over 120 seconds
        persist_factor = min(1.0, 0.3 + (persistence_s / 120.0) * 0.7)

        # SQI factor: lower quality → lower confidence in the detection
        sqi_factor = min(1.0, sqi_overall / 0.8)

        # Combine
        confidence = signal_score * 0.5 + persist_factor * 0.3 + sqi_factor * 0.2
        return round(min(1.0, confidence), 3)

    def _build_result(
        self,
        state: str,
        confidence: float,
        signals: List[str],
        alert: Dict,
        flags: SignalFlags,
        persistence_s: float,
        motion_state: str,
        skipped: bool = False,
        skip_reason: str = "",
        fall_result: Optional[Dict] = None,
    ) -> Dict:
        result = {
            "state":                state,
            "confidence":           round(confidence, 3),
            "contributing_signals": signals,
            "alert": {
                "title":    alert["title"],
                "message":  alert["message"],
                "severity": alert["severity"],
            },
            "fall_detected": fall_result or {
                "detected": False, "stage": "none",
                "confidence": 0.0, "description": "No fall activity",
            },
            "debug": {
                "flags": {
                    "hr":                 flags.hr,
                    "hr_stability_score": flags.hr_stability_score,
                    "spo2":               flags.spo2,
                    "temp":               flags.temp,
                    "temp_drop":          flags.temp_drop,
                },
                "scores": {
                    "hr":                 round(flags.hr_score, 3),
                    "hr_stability_score": round(flags.hr_stability_score_score, 3),
                    "spo2":               round(flags.spo2_score, 3),
                    "temp":               round(flags.temp_score, 3),
                    "temp_drop":          round(flags.temp_drop_score, 3),
                },
                "persistence_s": round(persistence_s, 1),
                "motion_state":  motion_state,
                "flag_count":    flags.count,
                "weighted_score": round(flags.weighted_score(), 3),
                "source":        flags.source,
            },
        }
        if skipped:
            result["debug"]["skipped"] = True
            result["debug"]["skip_reason"] = skip_reason
        return result

    def stats(self) -> Dict:
        """Return a summary of recent detection history."""
        history = list(self._state_history)
        total = len(history)
        if total == 0:
            return {"total_evaluations": 0}
        return {
            "total_evaluations": total,
            "normal_pct":   round(history.count("normal") / total * 100, 1),
            "stress_pct":   round(history.count("stress") / total * 100, 1),
            "distress_pct": round(history.count("distress") / total * 100, 1),
            "current_state": self._last_state,
        }


def _state_rank(s: str) -> int:
    return {"normal": 0, "stress": 1, "distress": 2}.get(s, 0)


# ===========================================================================
# 8. GEOSPATIAL CONTEXT — configuration
# ===========================================================================

# --- Context risk weights ---
# Each factor produces a score 0..1; these weights control how much each
# dimension contributes to the overall context risk multiplier.
GEO_WEIGHTS = {
    "time":        0.25,   # night / late-night risk
    "location":    0.30,   # unfamiliar / far from home
    "movement":    0.25,   # sudden stop, route deviation
    "connectivity": 0.10,  # phone disconnected
    "familiarity": 0.10,   # known vs unknown zone
}

# --- Time risk scoring ---
# Late night (23:00–05:00) is highest risk; evening (20:00–23:00) moderate
TIME_RISK = {
    "late_night":  0.9,    # 23:00 – 04:59
    "night":       0.6,    # 20:00 – 22:59
    "early_morning": 0.4,  # 05:00 – 06:59
    "daytime":     0.1,    # 07:00 – 19:59
}

# --- Distance risk tiers (km from home) ---
DISTANCE_RISK_TIERS = [
    (50.0,  0.8),   # > 50 km
    (20.0,  0.6),   # > 20 km
    (5.0,   0.3),   # > 5 km
    (0.0,   0.0),   # close to home
]

# --- Risk level thresholds (final combined score) ---
RISK_THRESHOLDS = {
    "critical":      0.80,
    "high_risk":     0.60,
    "moderate_risk": 0.40,
    "low_risk":      0.20,
    # below 0.20 → "normal"
}

# --- Context suppression ---
# When ALL of these are true, suppress stress (likely safe context like gym)
SAFE_CONTEXT_KEYS = ["is_known_area", "is_home_zone", "is_work_zone"]


# ===========================================================================
# 9. GEOSPATIAL ALERT MESSAGES
# ===========================================================================

ALERTS_GEO_NORMAL: List[Dict] = [
    {
        "title": "All clear",
        "message": "You seem to be in a familiar place and everything "
                   "looks fine. We're here if you need us.",
        "severity": "low",
    },
]

ALERTS_GEO_LOW: List[Dict] = [
    {
        "title": "Keeping an eye out",
        "message": "Things look mostly fine, but we noticed a small change. "
                   "We'll keep monitoring quietly.",
        "severity": "low",
    },
]

ALERTS_GEO_MODERATE: List[Dict] = [
    {
        "title": "Stay aware",
        "message": "We're picking up some signs worth noting. "
                   "If you feel uneasy, consider reaching out to someone you trust.",
        "severity": "medium",
    },
    {
        "title": "Check in when you can",
        "message": "Your body signals and surroundings suggest being a bit "
                   "more cautious right now. You've got this.",
        "severity": "medium",
    },
]

ALERTS_GEO_HIGH: List[Dict] = [
    {
        "title": "We're concerned",
        "message": "We noticed signs of distress in an unfamiliar setting. "
                   "Are you okay? Consider contacting someone you trust.",
        "severity": "high",
    },
    {
        "title": "Something seems unusual",
        "message": "Your body is showing signs of stress and your "
                   "surroundings seem unusual. Would you like to check in "
                   "with a trusted contact?",
        "severity": "high",
    },
    {
        "title": "Please be careful",
        "message": "We're detecting concerning signals. If you feel unsafe, "
                   "please reach out to someone or move to a safe place.",
        "severity": "high",
    },
]

ALERTS_GEO_CRITICAL: List[Dict] = [
    {
        "title": "Emergency — are you safe?",
        "message": "Multiple warning signs detected. If you're in danger, "
                   "please call for help or press the emergency button. "
                   "We can alert your trusted contacts.",
        "severity": "high",
    },
    {
        "title": "We need to know you're okay",
        "message": "Serious concern detected. If you can't respond, "
                   "we'll notify your emergency contact shortly.",
        "severity": "high",
    },
]

ALERTS_GEO_ROUTE_DEVIATION: List[Dict] = [
    {
        "title": "Unexpected route change",
        "message": "We noticed you've deviated from your usual path. "
                   "If everything's fine, no action needed — we're just "
                   "keeping watch.",
        "severity": "medium",
    },
]

ALERTS_GEO_PHONE_LOST: List[Dict] = [
    {
        "title": "Phone connection lost",
        "message": "We've lost connection to your phone. If you're in an "
                   "unfamiliar area, try to stay in well-lit, public spaces.",
        "severity": "medium",
    },
]

RECOMMENDED_ACTIONS = {
    "normal":        "No action needed. Continue monitoring.",
    "low_risk":      "Passive monitoring. Log context for pattern analysis.",
    "moderate_risk": "Suggest user check in. Enable quick-access to emergency contacts.",
    "high_risk":     "Prompt user to confirm safety. Prepare to notify trusted contacts.",
    "critical":      "Initiate safety protocol. Notify emergency contacts if no response within 60 seconds.",
}


# ===========================================================================
# 10. GEOSPATIAL RISK SCORER
# ===========================================================================

@dataclass
class GeoContext:
    """Structured geospatial context input.

    All fields have safe defaults so partial context still works.
    Privacy note: raw lat/lon are used for distance calculation only
    and are NEVER stored or transmitted — only zone classifications
    (home/work/known/unfamiliar) are persisted.
    """
    latitude:              float = 0.0
    longitude:             float = 0.0
    timestamp:             int   = 0
    speed_kmph:            float = 0.0
    heading:               float = 0.0

    is_home_zone:          bool = False
    is_work_zone:          bool = False
    is_known_area:         bool = False   # fail-safe: unknown when data missing
    is_unfamiliar_area:    bool = False

    distance_from_home_km: float = 0.0

    hour_of_day:           int  = 12
    is_night:              bool = False

    is_stationary:         bool = True
    is_walking:            bool = False
    is_vehicle_like_motion: bool = False

    sudden_route_change:   bool = False
    sudden_stop:           bool = False

    phone_connected:       bool = True
    phone_disconnect_duration_s: float = 0.0  # how long phone has been disconnected

    @classmethod
    def from_dict(cls, d: Dict) -> "GeoContext":
        """Build from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


def _score_time_risk(ctx: GeoContext) -> Tuple[float, List[str]]:
    """Score time-of-day risk. Returns (score, reasons)."""
    h = ctx.hour_of_day
    reasons = []

    if 23 <= h or h < 5:
        score = TIME_RISK["late_night"]
        reasons.append(f"late night ({h:02d}:00)")
    elif 20 <= h < 23:
        score = TIME_RISK["night"]
        reasons.append(f"nighttime ({h:02d}:00)")
    elif 5 <= h < 7:
        score = TIME_RISK["early_morning"]
        reasons.append(f"early morning ({h:02d}:00)")
    else:
        score = TIME_RISK["daytime"]

    return score, reasons


def _score_location_risk(ctx: GeoContext) -> Tuple[float, List[str]]:
    """Score location familiarity + distance risk. Returns (score, reasons)."""
    score = 0.0
    reasons = []

    # Familiarity scoring
    if ctx.is_home_zone:
        score = 0.0  # safest
    elif ctx.is_work_zone:
        score = 0.05
    elif ctx.is_known_area:
        score = 0.15
    elif ctx.is_unfamiliar_area:
        score = 0.7
        reasons.append("unfamiliar area")

    # Distance from home — additive risk
    for threshold, risk in DISTANCE_RISK_TIERS:
        if ctx.distance_from_home_km >= threshold:
            if risk > 0:
                dist_contribution = risk * 0.3  # weighted down, additive
                score = min(1.0, score + dist_contribution)
                reasons.append(f"{ctx.distance_from_home_km:.1f} km from home")
            break

    return score, reasons


def _score_movement_risk(ctx: GeoContext) -> Tuple[float, List[str]]:
    """Score movement anomaly risk. Returns (score, reasons)."""
    score = 0.0
    reasons = []

    if ctx.sudden_route_change:
        score += 0.5
        reasons.append("sudden route deviation")

    if ctx.sudden_stop:
        score += 0.4
        reasons.append("sudden stop detected")

    # Vehicle at night is riskier than walking in daytime
    if ctx.is_vehicle_like_motion and ctx.is_night:
        score += 0.2
        reasons.append("vehicle motion at night")

    # High speed (> 140 km/h) could indicate danger or erratic driving
    if ctx.speed_kmph > 140:
        score += 0.3
        reasons.append(f"high speed ({ctx.speed_kmph:.0f} km/h)")

    return min(1.0, score), reasons


def _score_connectivity_risk(ctx: GeoContext) -> Tuple[float, List[str]]:
    """Score connectivity risk. Returns (score, reasons).

    Only scores disconnection after 5 minutes (300s) of sustained
    disconnection to avoid false alarms from brief signal drops.
    """
    if not ctx.phone_connected and ctx.phone_disconnect_duration_s >= 300:
        return 0.4, ["phone disconnected (>5 min)"]
    return 0.0, []


def compute_context_score(ctx: GeoContext) -> Tuple[float, List[str]]:
    """
    Compute the overall geospatial context risk score (0..1).

    This score represents how RISKY the current context is,
    independent of physiological state. It's used as a multiplier
    on the physiological state to produce the final risk level.

    Returns:
        (context_score, list_of_reasons)
    """
    all_reasons = []

    time_score, time_reasons = _score_time_risk(ctx)
    loc_score, loc_reasons = _score_location_risk(ctx)
    move_score, move_reasons = _score_movement_risk(ctx)
    conn_score, conn_reasons = _score_connectivity_risk(ctx)

    # Familiarity is a separate light factor
    fam_score = 0.0
    fam_reasons = []
    if ctx.is_unfamiliar_area and not ctx.is_known_area:
        fam_score = 0.6
        # reason already captured in location

    # Weighted combination
    combined = (
        GEO_WEIGHTS["time"]         * time_score +
        GEO_WEIGHTS["location"]     * loc_score +
        GEO_WEIGHTS["movement"]     * move_score +
        GEO_WEIGHTS["connectivity"] * conn_score +
        GEO_WEIGHTS["familiarity"]  * fam_score
    )

    all_reasons.extend(time_reasons)
    all_reasons.extend(loc_reasons)
    all_reasons.extend(move_reasons)
    all_reasons.extend(conn_reasons)

    return round(min(1.0, combined), 3), all_reasons


# ===========================================================================
# 11. GEOSPATIAL SAFETY ENGINE
# ===========================================================================

class SafetyRiskEngine:
    """
    Combines physiological distress output with geospatial context
    to produce a unified safety risk assessment.

    Architecture:
        ┌─────────────┐    ┌─────────────┐
        │ Physiology   │    │ Geospatial   │
        │ (distress    │    │ (location,   │
        │  engine)     │    │  time, move) │
        └──────┬───────┘    └──────┬───────┘
               │                    │
               ▼                    ▼
        ┌──────────────────────────────────┐
        │     RISK COMBINATION MATRIX      │
        │                                  │
        │  physio_base × context_modifier  │
        │  + fall_override                 │
        │  + suppression_rules             │
        └──────────────┬───────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  RISK LEVEL    │
              │  + reasoning   │
              │  + alert       │
              └────────────────┘

    The context is a MODIFIER — it never triggers alone.
    Physiological state is always the primary signal.

    Usage:
        safety = SafetyRiskEngine()
        result = safety.evaluate(distress_result, geo_context_dict)
    """

    def __init__(self):
        self._cycle = 0
        self._risk_history: Deque[str] = deque(maxlen=120)
        self._critical_start: Optional[float] = None
        self._high_risk_start: Optional[float] = None
        self._RISK_PERSISTENCE_S = 30  # seconds before allowing high/critical

    def evaluate(
        self,
        distress_result: Dict,
        geo_context: Optional[Dict] = None,
        fall_result: Optional[Dict] = None,
    ) -> Dict:
        """
        Evaluate safety risk by combining physiology + context.

        Args:
            distress_result: output from DistressEngine.evaluate()
            geo_context:     geospatial context dict (or None for physio-only)
            fall_result:     fall detection result (or None)

        Returns:
            Full safety assessment dict.
        """
        self._cycle += 1
        now = time.time()

        physio_state = distress_result.get("state", "normal")
        physio_confidence = distress_result.get("confidence", 0.0)
        physio_signals = distress_result.get("contributing_signals", [])

        # --- Fall override: confirmed fall → immediate critical ---
        if fall_result and fall_result.get("detected", False):
            return self._build_safety_result(
                risk_level="critical",
                risk_score=0.95,
                reasoning=["fall confirmed", "post-fall inactivity detected"],
                physio_state=physio_state,
                physio_confidence=physio_confidence,
                context_score=0.0,
                geo_context=geo_context,
                now=now,
            )

        # --- No geospatial context → use physio-only mapping ---
        if geo_context is None:
            return self._physio_only_assessment(
                physio_state, physio_confidence, physio_signals, now
            )

        # --- Parse context ---
        ctx = GeoContext.from_dict(geo_context)

        # --- Compute context risk score ---
        context_score, context_reasons = compute_context_score(ctx)

        # --- Build combined reasoning ---
        reasoning = []

        # Add physiological reasons
        if physio_state == "distress":
            reasoning.append("distress state detected")
        elif physio_state == "stress":
            reasoning.append("stress state detected")

        if physio_signals:
            reasoning.append(f"signals: {', '.join(physio_signals)}")

        # Add context reasons
        reasoning.extend(context_reasons)

        # --- Compute combined risk score ---
        risk_score = self._compute_combined_risk(
            physio_state, physio_confidence, context_score, ctx
        )

        # --- Apply suppression rules ---
        risk_score, suppression_note = self._apply_suppression(
            risk_score, physio_state, ctx
        )
        if suppression_note:
            reasoning.append(suppression_note)

        # --- Apply escalation rules ---
        risk_score, escalation_notes = self._apply_escalation(
            risk_score, physio_state, ctx, fall_result, now
        )
        reasoning.extend(escalation_notes)

        # --- Map score to risk level ---
        risk_level = self._score_to_level(risk_score)

        # --- Persistence: high/critical risk must sustain ---
        risk_level = self._apply_risk_persistence(risk_level, now)

        return self._build_safety_result(
            risk_level=risk_level,
            risk_score=risk_score,
            reasoning=reasoning,
            physio_state=physio_state,
            physio_confidence=physio_confidence,
            context_score=context_score,
            geo_context=geo_context,
            now=now,
        )

    def _compute_combined_risk(
        self,
        physio_state: str,
        physio_confidence: float,
        context_score: float,
        ctx: GeoContext,
    ) -> float:
        """
        Combine physiological state with geospatial context score.

        Formula:
            risk = physio_base + (physio_base × context_amplifier)

        The context AMPLIFIES or SUPPRESSES the physio signal.
        Context alone cannot produce risk > 0.15.
        """
        # Map physio state to base risk
        PHYSIO_BASE = {
            "normal":  0.05,
            "stress":  0.35,
            "distress": 0.70,
        }
        physio_base = PHYSIO_BASE.get(physio_state, 0.05)

        # Scale by confidence
        physio_base *= max(0.3, physio_confidence)

        # Context amplifier: context_score (0..1) scales into 0.8..1.5
        # Low context risk → slight suppression (0.8x)
        # High context risk → amplification (up to 1.5x)
        context_amplifier = 0.8 + (context_score * 0.7)

        combined = physio_base * context_amplifier

        # Context-only contribution (capped) — even in "normal" state,
        # very high context risk should produce at least "low_risk"
        if physio_state == "normal" and context_score > 0.6:
            combined = max(combined, 0.15)

        return round(min(1.0, combined), 3)

    def _apply_suppression(
        self,
        risk_score: float,
        physio_state: str,
        ctx: GeoContext,
    ) -> Tuple[float, str]:
        """
        Suppress risk in clearly safe contexts.

        Rules:
            - Home zone + daytime + no movement anomaly → suppress
            - Work zone + daytime → suppress
            - Known area + walking + daytime (gym, park) → suppress
        """
        note = ""

        # Home zone during daytime — very safe
        if (ctx.is_home_zone
            and not ctx.is_night
            and not ctx.sudden_route_change
            and not ctx.sudden_stop):
            if physio_state == "stress":
                risk_score *= 0.85  # mild suppression (home is safer, but not silent)
                note = "suppressed: home zone, daytime"

        # Work zone during typical hours
        elif (ctx.is_work_zone
              and 7 <= ctx.hour_of_day <= 20
              and not ctx.sudden_stop):
            if physio_state == "stress":
                risk_score *= 0.6
                note = "suppressed: work zone, business hours"

        # Known area + walking during day (gym, routine walk)
        elif (ctx.is_known_area
              and ctx.is_walking
              and not ctx.is_night
              and not ctx.sudden_route_change):
            if physio_state == "stress":
                risk_score *= 0.7
                note = "suppressed: known area, walking, daytime"

        return round(risk_score, 3), note

    def _apply_escalation(
        self,
        risk_score: float,
        physio_state: str,
        ctx: GeoContext,
        fall_result: Optional[Dict],
        now: float,
    ) -> Tuple[float, List[str]]:
        """
        Escalate risk in dangerous combinations.

        These are specific pattern-matched scenarios that deserve
        higher risk than the formula alone would produce.
        """
        notes = []

        # --- CRITICAL PATTERNS ---
        # Distress + night + unfamiliar + phone disconnected
        if (physio_state == "distress"
            and ctx.is_night
            and ctx.is_unfamiliar_area
            and not ctx.phone_connected):
            risk_score = max(risk_score, 0.90)
            notes.append("critical pattern: distress + night + unfamiliar + no phone")

        # Distress + sudden stop + unfamiliar
        if (physio_state == "distress"
            and ctx.sudden_stop
            and ctx.is_unfamiliar_area):
            risk_score = max(risk_score, 0.85)
            notes.append("critical pattern: distress + sudden stop + unfamiliar")

        # --- HIGH RISK PATTERNS ---
        # Distress + night + unfamiliar
        if (physio_state == "distress"
            and ctx.is_night
            and ctx.is_unfamiliar_area):
            risk_score = max(risk_score, 0.75)
            notes.append("high risk: distress at night in unfamiliar area")

        # Distress + route deviation
        if (physio_state == "distress"
            and ctx.sudden_route_change):
            risk_score = max(risk_score, 0.70)
            notes.append("high risk: distress with route deviation")

        # Stress + night + unfamiliar + far from home
        if (physio_state == "stress"
            and ctx.is_night
            and ctx.is_unfamiliar_area
            and ctx.distance_from_home_km > 10):
            risk_score = max(risk_score, 0.55)
            notes.append("elevated: stress at night, far from home")

        # Vehicle sudden stop + distress
        if (physio_state in ("stress", "distress")
            and ctx.is_vehicle_like_motion
            and ctx.sudden_stop):
            risk_score = max(risk_score, 0.65)
            notes.append("elevated: physiological concern with vehicle sudden stop")

        # Phone lost in unfamiliar area
        if (not ctx.phone_connected
            and ctx.is_unfamiliar_area
            and ctx.is_night):
            risk_score = max(risk_score, 0.50)
            notes.append("elevated: phone lost at night in unfamiliar area")

        return round(min(1.0, risk_score), 3), notes

    def _score_to_level(self, score: float) -> str:
        """Map a risk score (0..1) to a risk level string."""
        if score >= RISK_THRESHOLDS["critical"]:
            return "critical"
        elif score >= RISK_THRESHOLDS["high_risk"]:
            return "high_risk"
        elif score >= RISK_THRESHOLDS["moderate_risk"]:
            return "moderate_risk"
        elif score >= RISK_THRESHOLDS["low_risk"]:
            return "low_risk"
        else:
            return "normal"

    def _apply_risk_persistence(self, risk_level: str, now: float) -> str:
        """
        Prevent flickering by requiring high_risk and critical to persist.

        Both high_risk and critical must hold for 30 seconds before
        actually outputting at that level — until then, cap one tier lower.
        """
        # --- Critical persistence ---
        if risk_level == "critical":
            if self._critical_start is None:
                self._critical_start = now
            duration = now - self._critical_start
            if duration < self._RISK_PERSISTENCE_S:
                risk_level = "high_risk"  # cap until persistence met
        else:
            self._critical_start = None

        # --- High-risk persistence (applies to both original high_risk
        #     and critical that was just capped to high_risk) ---
        if risk_level == "high_risk":
            if self._high_risk_start is None:
                self._high_risk_start = now
            duration = now - self._high_risk_start
            if duration < self._RISK_PERSISTENCE_S:
                risk_level = "moderate_risk"  # cap until persistence met
        else:
            self._high_risk_start = None

        self._risk_history.append(risk_level)
        return risk_level

    def _physio_only_assessment(
        self,
        physio_state: str,
        physio_confidence: float,
        physio_signals: List[str],
        now: float,
    ) -> Dict:
        """Fallback when no geospatial context is available."""
        PHYSIO_MAP = {
            "normal":  ("normal",        0.05),
            "stress":  ("low_risk",      0.25),
            "distress": ("moderate_risk", 0.55),
        }
        risk_level, risk_score = PHYSIO_MAP.get(physio_state, ("normal", 0.05))
        risk_score *= max(0.3, physio_confidence)

        reasoning = []
        if physio_state != "normal":
            reasoning.append(f"{physio_state} state detected")
        if physio_signals:
            reasoning.append(f"signals: {', '.join(physio_signals)}")
        reasoning.append("no geospatial context available")

        return self._build_safety_result(
            risk_level=risk_level,
            risk_score=risk_score,
            reasoning=reasoning,
            physio_state=physio_state,
            physio_confidence=physio_confidence,
            context_score=0.0,
            geo_context=None,
            now=now,
        )

    def _select_alert(self, risk_level: str, reasoning: List[str]) -> Dict:
        """Choose a context-appropriate alert message."""
        # Check for specific patterns to pick specialized alerts
        has_route = any("route" in r for r in reasoning)
        has_phone = any("phone" in r for r in reasoning)

        if risk_level == "critical":
            return _pick_alert(ALERTS_GEO_CRITICAL, self._cycle)
        elif risk_level == "high_risk":
            return _pick_alert(ALERTS_GEO_HIGH, self._cycle)
        elif risk_level == "moderate_risk":
            if has_route:
                return _pick_alert(ALERTS_GEO_ROUTE_DEVIATION, self._cycle)
            if has_phone:
                return _pick_alert(ALERTS_GEO_PHONE_LOST, self._cycle)
            return _pick_alert(ALERTS_GEO_MODERATE, self._cycle)
        elif risk_level == "low_risk":
            return _pick_alert(ALERTS_GEO_LOW, self._cycle)
        else:
            return _pick_alert(ALERTS_GEO_NORMAL, self._cycle)

    def _build_safety_result(
        self,
        risk_level: str,
        risk_score: float,
        reasoning: List[str],
        physio_state: str,
        physio_confidence: float,
        context_score: float,
        geo_context: Optional[Dict],
        now: float,
    ) -> Dict:
        """Build the final safety assessment output dict."""
        alert = self._select_alert(risk_level, reasoning)
        action = RECOMMENDED_ACTIONS.get(risk_level, RECOMMENDED_ACTIONS["normal"])

        return {
            "risk_level":  risk_level,
            "risk_score":  round(risk_score, 3),
            "reasoning":   reasoning,
            "recommended_action": action,
            "alert": {
                "title":    alert["title"],
                "message":  alert["message"],
                "severity": alert["severity"],
            },
            "debug": {
                "physio_state":      physio_state,
                "physio_confidence": physio_confidence,
                "context_score":     context_score,
                "geo_available":     geo_context is not None,
            },
        }

    def stats(self) -> Dict:
        """Return a summary of recent safety risk history."""
        history = list(self._risk_history)
        total = len(history)
        if total == 0:
            return {"total_evaluations": 0}
        return {
            "total_evaluations": total,
            "normal_pct":        round(history.count("normal") / total * 100, 1),
            "low_risk_pct":      round(history.count("low_risk") / total * 100, 1),
            "moderate_risk_pct": round(history.count("moderate_risk") / total * 100, 1),
            "high_risk_pct":     round(history.count("high_risk") / total * 100, 1),
            "critical_pct":      round(history.count("critical") / total * 100, 1),
        }


# ===========================================================================
# 12. AUDIO INTELLIGENCE ENGINE — Layer 4 (phone microphone)
# ===========================================================================

class AudioIntelligenceEngine:
    """
    Layer 4: Audio-based safety intelligence from phone microphone.

    Sub-components:
        1. Environmental sound classification (YAMNet + AST dual-model fusion)
        2. Vocal stress analysis (Praat pitch/jitter/shimmer/HNR)
        3. Noise level monitoring (RMS dB, TWA occupational dose)

    All ML dependencies are optional — if TensorFlow, PyTorch, or
    parselmouth are not installed, evaluate() returns None and the
    engine continues without audio intelligence.

    Input:
        {
            "audio_array": np.ndarray,     # mono PCM float32, any sample rate
            "sample_rate": int,            # original sample rate (resampled to 16 kHz internally)
            "audio_path":  str | None,     # file path (needed for vocal stress only)
        }

    Output:
        {
            "audio_risk_score": float,     # 0.0..1.0 fused score
            "audio_risk_level": str,       # "normal" | "elevated" | "danger"
            "env_classification": { ... },
            "vocal_stress":      { ... },
            "noise_level":       { ... },
            "alert": { "title": str, "message": str, "severity": str },
        }
    """

    def __init__(self):
        self._models_loaded: bool = False
        self._danger_start: Optional[float] = None
        self._cycle: int = 0

    # --- Model loading (deferred to first evaluate call) ---

    def _load_models(self) -> bool:
        """Load YAMNet, AST, and class map. Returns False if deps missing."""
        global _yamnet_model, _yamnet_classes, _ast_model, _ast_extractor

        if self._models_loaded:
            return True
        if not _ensure_audio_deps():
            return False

        try:
            import tensorflow_hub as hub
            from transformers import (
                AutoFeatureExtractor,
                ASTForAudioClassification,
            )
            import requests
            import csv
            import io

            # YAMNet (Google)
            if _yamnet_model is None:
                _yamnet_model = hub.load(
                    "https://tfhub.dev/google/yamnet/1"
                )

            # YAMNet class map
            if _yamnet_classes is None:
                r = requests.get(
                    "https://raw.githubusercontent.com/tensorflow/models/"
                    "master/research/audioset/yamnet/yamnet_class_map.csv"
                )
                _yamnet_classes = [
                    row[2] for row in csv.reader(io.StringIO(r.text))
                ][1:]

            # AST (MIT)
            if _ast_extractor is None:
                model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
                _ast_extractor = AutoFeatureExtractor.from_pretrained(model_id)
                _ast_model = ASTForAudioClassification.from_pretrained(model_id)
                _ast_model.eval()

            self._models_loaded = True
            return True

        except Exception:
            return False

    # --- Sub-component: Environmental sound classification ---

    def _classify_environment(
        self, audio, sr: int = 16000
    ) -> Dict:
        """
        Run YAMNet + AST on audio, fuse scores for danger sound classes.
        Returns danger_score (0..1), top detections, and status.
        """
        import tensorflow as tf
        import torch
        import numpy as np

        all_danger = DANGER_SOUNDS_ENV + DANGER_SOUNDS_PHYSIO
        audio_tf = tf.constant(audio.astype(np.float32))

        # ── YAMNet ──
        scores, _, _ = _yamnet_model(audio_tf)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        yamnet_hits: Dict[str, float] = {}
        for cls in all_danger:
            for i, yc in enumerate(_yamnet_classes):
                if cls.lower() in yc.lower():
                    yamnet_hits[cls] = float(mean_scores[i])
                    break

        # ── AST ──
        inputs = _ast_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        with torch.no_grad():
            logits = _ast_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        id2label = _ast_model.config.id2label

        ast_hits: Dict[str, float] = {}
        for cls in all_danger:
            for idx, label in id2label.items():
                if cls.lower() in label.lower():
                    ast_hits[cls] = float(probs[idx])
                    break

        # ── Fuse (50/50) ──
        fused: Dict[str, float] = {}
        for cls in all_danger:
            y = yamnet_hits.get(cls, 0.0)
            a = ast_hits.get(cls, 0.0)
            fused[cls] = round((y * 0.5) + (a * 0.5), 4)

        top5 = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:5]
        top_score = top5[0][1] if top5 else 0.0

        return {
            "danger_score":   round(top_score, 4),
            "top_detections": top5,
            "status": "danger" if top_score > AUDIO_DANGER_THRESHOLD else "safe",
        }

    # --- Sub-component: Vocal stress analysis (Praat) ---

    def _compute_vocal_stress(self, audio_path: Optional[str]) -> Dict:
        """
        Extract pitch, jitter, shimmer, HNR from audio via parselmouth.
        Returns stress_score (0..1) and label.
        """
        _no_voice = {
            "stress_score": 0.0, "label": "no_voice",
            "pitch_hz": 0.0, "jitter_pct": 0.0,
            "shimmer_pct": 0.0, "hnr_db": 0.0,
        }
        if not audio_path:
            return _no_voice

        try:
            import parselmouth
            from parselmouth.praat import call

            snd = parselmouth.Sound(audio_path)

            pitch = call(snd, "To Pitch", 0.0, 75, 600)
            pitch_mean = call(pitch, "Get mean", 0, 0, "Hertz")

            point_proc = call(
                snd, "To PointProcess (periodic, cc)", 75, 600
            )
            jitter = call(
                point_proc, "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3,
            )
            shimmer = call(
                [snd, point_proc], "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6,
            )

            harmonicity = call(
                snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
            )
            hnr = call(harmonicity, "Get mean", 0, 0)

            # Stress score: high jitter + high shimmer + low HNR → stress
            stress_score = min(
                1.0,
                (jitter * 10) + (shimmer * 5) + max(0, (20 - hnr) / 20),
            )

            if stress_score > AUDIO_VOCAL_STRESS_THRESHOLD:
                label = "high_stress"
            elif stress_score > 0.3:
                label = "moderate_stress"
            else:
                label = "calm"

            return {
                "stress_score": round(stress_score, 3),
                "label":        label,
                "pitch_hz":     round(pitch_mean, 1),
                "jitter_pct":   round(jitter * 100, 3),
                "shimmer_pct":  round(shimmer * 100, 3),
                "hnr_db":       round(hnr, 2),
            }

        except Exception:
            return _no_voice

    # --- Sub-component: Noise level / TWA ---

    def _compute_noise_level(self, audio, sr: int = 16000) -> Dict:
        """
        Compute RMS dB level and OSHA TWA occupational dose.
        """
        import numpy as np

        rms = float(np.sqrt(np.mean(audio ** 2)))
        db = 20 * math.log10(rms + 1e-9) + 90  # offset to realistic dB

        twa_dose = (db / 90.0) * 100.0

        if db < 70:
            status = "safe"
        elif db < AUDIO_NOISE_DANGEROUS_DB:
            status = "moderate"
        else:
            status = "dangerous"

        noise_score = min(1.0, max(0.0, (db - 60) / 60))

        return {
            "db_level":     round(db, 1),
            "twa_dose_pct": round(twa_dose, 1),
            "noise_score":  round(noise_score, 4),
            "status":       status,
        }

    # --- Main evaluate ---

    def evaluate(self, audio_input: Optional[Dict]) -> Optional[Dict]:
        """
        Evaluate audio data and return Layer 4 result.

        Args:
            audio_input: {
                "audio_array": np.ndarray (mono PCM float32),
                "sample_rate": int,
                "audio_path":  str | None,
            }
            Pass None to skip audio intelligence entirely.

        Returns:
            Audio intelligence result dict, or None if unavailable.
        """
        if audio_input is None:
            return None

        if not self._load_models():
            return None

        self._cycle += 1
        now = time.time()

        import numpy as np
        import librosa

        audio_raw = np.array(audio_input["audio_array"], dtype=np.float32)
        sr_in = int(audio_input.get("sample_rate", 16000))

        # Resample to 16 kHz (YAMNet/AST requirement)
        if sr_in != 16000:
            audio = librosa.resample(audio_raw, orig_sr=sr_in, target_sr=16000)
        else:
            audio = audio_raw

        # Run sub-components
        env   = self._classify_environment(audio, 16000)
        vocal = self._compute_vocal_stress(audio_input.get("audio_path"))
        noise = self._compute_noise_level(audio, 16000)

        # Fuse into a single audio risk score
        audio_risk = round(
            env["danger_score"]    * AUDIO_WEIGHTS["env_danger"]
            + vocal["stress_score"] * AUDIO_WEIGHTS["vocal_stress"]
            + noise["noise_score"]  * AUDIO_WEIGHTS["noise_level"],
            4,
        )

        # Risk level classification
        if audio_risk > 0.50:
            risk_level = "danger"
        elif audio_risk > 0.25:
            risk_level = "elevated"
        else:
            risk_level = "normal"

        # Persistence: require sustained danger before high-severity alert
        if risk_level == "danger":
            if self._danger_start is None:
                self._danger_start = now
            danger_sustained_s = now - self._danger_start
        else:
            self._danger_start = None
            danger_sustained_s = 0.0

        # Select alert
        if (risk_level == "danger"
                and danger_sustained_s >= AUDIO_PERSISTENCE_S):
            alert = _pick_alert(ALERTS_AUDIO_DANGER, self._cycle)
        elif risk_level == "danger":
            # Danger detected but not yet sustained — medium alert
            alert = _pick_alert(ALERTS_AUDIO_STRESS, self._cycle)
        elif vocal["label"] == "high_stress":
            alert = _pick_alert(ALERTS_AUDIO_STRESS, self._cycle)
        elif noise["status"] == "dangerous":
            alert = _pick_alert(ALERTS_AUDIO_NOISE, self._cycle)
        else:
            alert = _pick_alert(ALERTS_AUDIO_NORMAL, self._cycle)

        return {
            "audio_risk_score": audio_risk,
            "audio_risk_level": risk_level,
            "env_classification": env,
            "vocal_stress":      vocal,
            "noise_level":       noise,
            "alert":             alert,
        }


# ===========================================================================
# 13. EDA/GSR PROCESSOR — optional sensor (future hardware)
# ===========================================================================

class EDAProcessor:
    """
    Optional EDA (Electrodermal Activity) / GSR signal processing.

    Separates tonic (SCL — Skin Conductance Level) and phasic
    (SCR — Skin Conductance Response) components. Returns an arousal
    score 0..1 for optional integration into physiological context.

    This is NOT a detection layer — it is a supplementary signal.
    When EDA hardware becomes available on the ring, this score can
    feed into the existing weighted-score gating as a supporting signal.

    Requires scipy. Returns None if unavailable.
    """

    def evaluate(
        self,
        eda_signal: Optional[List[float]],
        fs: int = 4,
    ) -> Optional[Dict]:
        """
        Process raw EDA/GSR signal.

        Args:
            eda_signal: raw EDA values (micro-Siemens), sampled at fs Hz.
            fs:         sampling rate in Hz (default 4 Hz).

        Returns:
            { "scl_mean", "scr_per_min", "eda_score", "eda_status" }
            or None if data is unavailable or scipy is missing.
        """
        if eda_signal is None or len(eda_signal) < fs * 2:
            return None

        if not _ensure_eda_deps():
            return None

        try:
            import numpy as np
            from scipy.signal import find_peaks, butter, filtfilt

            sig = np.array(eda_signal, dtype=float)

            # Tonic (SCL) — low-pass
            b, a = butter(2, 0.05 / (fs / 2), btype="low")
            scl = filtfilt(b, a, sig)

            # Phasic (SCR) — high-pass
            b, a = butter(2, 0.05 / (fs / 2), btype="high")
            scr = filtfilt(b, a, sig)

            peaks, _ = find_peaks(scr, height=0.02, distance=fs * 2)
            duration_min = len(sig) / fs / 60.0
            scr_per_min = round(
                len(peaks) / max(duration_min, 0.01), 2
            )
            scl_mean = round(float(np.mean(scl)), 4)

            # Score: high SCR rate + high SCL → arousal
            scr_score = min(1.0, scr_per_min / 10.0)
            scl_score = min(1.0, scl_mean / 2.0)
            eda_score = round(
                (scr_score * 0.6) + (scl_score * 0.4), 4
            )

            if eda_score > EDA_HIGH_AROUSAL_THRESHOLD:
                status = "high_arousal"
            elif eda_score > EDA_MODERATE_THRESHOLD:
                status = "moderate"
            else:
                status = "calm"

            return {
                "scl_mean":    scl_mean,
                "scr_per_min": scr_per_min,
                "eda_score":   eda_score,
                "eda_status":  status,
            }

        except Exception:
            return None


# ===========================================================================
# 14. UNIFIED ENGINE — combines all layers
# ===========================================================================

class UnifiedSafetyEngine:
    """
    Top-level engine that orchestrates all detection layers:
        1. Physiological distress detection (DistressEngine)
        2. Fall detection (FallDetector, inside DistressEngine)
        3. Geospatial safety risk assessment (SafetyRiskEngine)
        4. Audio intelligence (AudioIntelligenceEngine) — optional
        +  EDA/GSR processing (EDAProcessor) — optional sensor

    Usage:
        engine = UnifiedSafetyEngine()

        # Minimal (Layers 1-3 only):
        result = engine.evaluate(pipeline_output, geo_context=geo_dict)

        # Full (all layers):
        result = engine.evaluate(
            pipeline_output,
            geo_context=geo_dict,
            audio_input={"audio_array": audio, "sample_rate": 16000},
            eda_signal=[1.2, 1.3, ...],
        )

    The returned dict contains all layers plus a top-level alert
    that reflects the highest-priority concern across all layers.
    """

    def __init__(self):
        self.distress = DistressEngine()
        self.safety = SafetyRiskEngine()
        # Layer 4: Audio intelligence (optional — works without ML deps)
        self.audio = AudioIntelligenceEngine()
        # Optional sensor: EDA/GSR (future hardware)
        self.eda = EDAProcessor()

    def evaluate(
        self,
        pipeline_output: Dict,
        geo_context: Optional[Dict] = None,
        audio_input: Optional[Dict] = None,
        eda_signal: Optional[List[float]] = None,
    ) -> Dict:
        """
        Run all detection layers and return a unified result.

        Args:
            pipeline_output: from AnomalyInputBuilder.step()
            geo_context:     geospatial context dict (or None)
            audio_input:     audio data dict (or None to skip Layer 4)
                             {"audio_array": ndarray, "sample_rate": int,
                              "audio_path": str | None}
            eda_signal:      raw EDA/GSR values in micro-Siemens (or None)

        Returns:
            Unified result with physiology, fall, safety, audio, and top alert.
        """
        # Layer 1 + 2: Physiological + Fall detection
        distress_result = self.distress.evaluate(pipeline_output)

        # Optional: EDA/GSR processing (supplementary signal)
        eda_result = self.eda.evaluate(eda_signal)

        # Layer 4: Audio intelligence (optional)
        audio_result = self.audio.evaluate(audio_input)

        # Layer 3: Geospatial safety risk
        fall_result = distress_result.get("fall_detected")
        safety_result = self.safety.evaluate(
            distress_result, geo_context, fall_result
        )

        # Cross-layer escalation: audio danger + physio distress → critical
        if audio_result is not None and geo_context is not None:
            self._apply_audio_escalation(
                audio_result, distress_result, safety_result, geo_context
            )

        # Determine the highest-priority alert across ALL layers
        top_alert = self._select_top_alert(
            distress_result, safety_result, audio_result
        )

        # Build unified output
        return {
            # Physiological state
            "state":                distress_result["state"],
            "confidence":           distress_result["confidence"],
            "contributing_signals": distress_result["contributing_signals"],

            # Fall detection
            "fall_detected": distress_result["fall_detected"],

            # Geospatial safety assessment
            "safety": safety_result,

            # Audio intelligence (Layer 4) — None if unavailable/skipped
            "audio_intelligence": audio_result,

            # EDA/GSR (optional sensor) — None if unavailable/skipped
            "eda": eda_result,

            # Top-level alert (highest priority across all layers)
            "alert": top_alert,

            # Debug info
            "debug": distress_result["debug"],
        }

    def _apply_audio_escalation(
        self,
        audio_result: Dict,
        distress_result: Dict,
        safety_result: Dict,
        geo_context: Dict,
    ) -> None:
        """
        Cross-layer escalation: dangerous audio + other risk factors.

        Modifies safety_result in-place when audio intelligence detects
        danger sounds in combination with physiological distress or
        dangerous geospatial context.
        """
        audio_level = audio_result.get("audio_risk_level", "normal")
        if audio_level != "danger":
            return

        physio_state = distress_result.get("state", "normal")
        is_night = geo_context.get("is_night", False)
        is_unfamiliar = geo_context.get("is_unfamiliar_area", False)

        reasoning = safety_result.get("reasoning", [])
        current_score = safety_result.get("risk_score", 0.0)

        # Danger sound + physiological distress → force critical
        if physio_state == "distress":
            new_score = max(current_score, 0.90)
            if new_score > current_score:
                safety_result["risk_score"] = new_score
                safety_result["risk_level"] = "critical"
                reasoning.append(
                    "critical: danger sound + physiological distress"
                )

        # Danger sound + night + unfamiliar → force high risk
        elif is_night and is_unfamiliar:
            new_score = max(current_score, 0.75)
            if new_score > current_score:
                safety_result["risk_score"] = new_score
                safety_result["risk_level"] = "high_risk"
                reasoning.append(
                    "high risk: danger sound at night in unfamiliar area"
                )

    def _select_top_alert(
        self,
        distress_result: Dict,
        safety_result: Dict,
        audio_result: Optional[Dict] = None,
    ) -> Dict:
        """Pick the highest-severity alert across all layers.

        Selection is purely by MAX SEVERITY so that a physiological
        distress alert is never silently downgraded by a lower-severity
        safety alert.  When both have equal severity the safety alert
        wins (more context-aware).
        """
        SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

        candidates = [
            distress_result.get("alert", {}),
            safety_result.get("alert", {}),
        ]
        if audio_result is not None:
            candidates.append(audio_result.get("alert", {}))

        best = candidates[0]
        best_rank = SEVERITY_RANK.get(best.get("severity", "low"), 0)
        for c in candidates[1:]:
            c_rank = SEVERITY_RANK.get(c.get("severity", "low"), 0)
            if c_rank >= best_rank:
                best = c
                best_rank = c_rank
        return best

    def stats(self) -> Dict:
        """Combined stats from all engines."""
        return {
            "distress":        self.distress.stats(),
            "safety":          self.safety.stats(),
            "audio_available": self.audio._models_loaded,
            "eda_available":   _EDA_AVAILABLE,
        }


# ===========================================================================
# 15. DEMO — run:  python distress_engine.py
# ===========================================================================

if __name__ == "__main__":
    import json

    # Disable persistence for demo so state changes are visible immediately.
    # In production these are 60 seconds.
    PERSISTENCE_STRESS_S   = 0
    PERSISTENCE_DISTRESS_S = 0

    engine = DistressEngine()

    print("=" * 80)
    print(" DEMO: Rule-Based Distress Detection Engine")
    print(" (persistence disabled — states fire immediately)")
    print("=" * 80)
    print()
    print(" This demo feeds synthetic *pipeline output* directly into the")
    print(" distress engine, simulating what the preprocessing pipeline")
    print(" produces in steady state (after warmup, EMA converged, etc.)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Helper: build a synthetic pipeline_output dict
    # ------------------------------------------------------------------
    _ts = int(time.time())

    def _make_output(
        hr, spo2, temp, dyn_acc=0.01, accepted=True,
        z_hr=0.0, z_spo2=0.0, z_temp=0.0, z_hrv=0.0,
        baseline_ready=True, clinical_flags=None, window_n=25,
        sqi_overall=0.88, finger_on=True, charging=False,
    ):
        global _ts
        _ts += 1
        return {
            "sample": {
                "timestamp": _ts, "sequence": _ts,
                "hr": hr, "hr_stability_score": 28.0, "temp": temp, "temp_raw": temp,
                "spo2": spo2, "acc_mag": 0.98, "dyn_acc_mag": dyn_acc,
                "acc_x": 0.02, "acc_y": -0.01, "acc_z": 0.98,
                "gravity_x": 0.02, "gravity_y": -0.01, "gravity_z": 0.98,
                "finger_on": finger_on, "charging": charging,
                "battery_mv": 3850, "die_temp": 34.5, "adc_raw": 112000,
                "thermal_bias": 0.0,
                "sqi": {
                    "hr": 0.92, "hrv": 0.65, "temp": 0.90,
                    "acc": 0.95, "spo2": 0.90, "ppg": 0.85,
                    "overall": sqi_overall,
                },
                "accepted": accepted,
                "reject_reasons": [] if accepted else ["finger_off"],
                "clinical_flags": clinical_flags or [],
            },
            "zscores": {
                "hr": z_hr, "hr_stability_score": z_hrv,
                "temp": z_temp, "spo2": z_spo2, "acc_mag": 0.0,
            },
            "baseline_ready": baseline_ready,
            "window": {
                "window_n": window_n,
                "hr_mean": hr, "hr_var": 2.0, "hr_min": hr - 2, "hr_max": hr + 2,
                "hr_stability_score_mean": 28.0, "hr_stability_score_var": 4.0,
                "hr_stability_score_min": 24.0, "hr_stability_score_max": 32.0,
                "temp_mean": temp, "temp_var": 0.01,
                "temp_min": temp - 0.1, "temp_max": temp + 0.1,
                "spo2_mean": spo2, "spo2_var": 0.5,
                "spo2_min": spo2 - 0.5, "spo2_max": spo2 + 0.5,
                "acc_mag_mean": 0.98, "acc_mag_var": 0.001,
                "acc_mag_min": 0.97, "acc_mag_max": 0.99,
            },
        }

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------
    scenarios = [
        # 1. Warming up (baseline not ready)
        ("Warming up (baseline not ready)",
         _make_output(hr=72, spo2=98.5, temp=36.6, baseline_ready=False,
                      window_n=3)),

        # 2. Normal resting (baseline ready, no deviations)
        ("Normal resting (all clear)",
         _make_output(hr=72, spo2=98.5, temp=36.6,
                      z_hr=0.3, z_spo2=0.1, z_temp=-0.2)),

        # 3. Stress — HR elevated (z > 2)
        ("STRESS: HR elevated (z=+2.8)",
         _make_output(hr=105, spo2=97.5, temp=36.7,
                      z_hr=2.8, z_spo2=-0.3, z_temp=0.1)),

        # 4. Stress — HR elevated via absolute threshold (120 bpm)
        ("STRESS: HR=125 (absolute threshold)",
         _make_output(hr=125, spo2=97.8, temp=36.7,
                      z_hr=4.5, z_spo2=-0.2, z_temp=0.0)),

        # 5. Stress — SpO2 dipped below 95% threshold
        ("STRESS: SpO2=94% (below 95% normal threshold)",
         _make_output(hr=80, spo2=94.0, temp=36.6,
                      z_hr=0.5, z_spo2=-1.0, z_temp=0.0,
                      clinical_flags=["spo2_clinical_concern (94.0%)"])),

        # 6. Distress — HR elevated + SpO2 low (multi-signal)
        ("DISTRESS: HR=130 + SpO2=91 (multi-signal)",
         _make_output(hr=130, spo2=91.0, temp=36.8,
                      z_hr=5.0, z_spo2=-2.5, z_temp=0.5,
                      clinical_flags=["spo2_clinical_concern (91.0%)"])),

        # 7. Exercise — HR 145 but high motion (motion suppression)
        ("EXERCISE: HR=145 + moving (suppressed)",
         _make_output(hr=145, spo2=97.0, temp=37.0, dyn_acc=0.6,
                      z_hr=6.0, z_spo2=-0.5, z_temp=1.0)),

        # 8. Exercise + SpO2 drop — NOT suppressed
        ("EXERCISE + SpO2 drop: HR=145 + SpO2=91 (NOT suppressed)",
         _make_output(hr=145, spo2=91.0, temp=37.0, dyn_acc=0.6,
                      z_hr=6.0, z_spo2=-2.5, z_temp=1.0,
                      clinical_flags=["spo2_clinical_concern (91.0%)"])),

        # 9. Emergency — SpO2 88% (bypasses persistence)
        ("EMERGENCY: SpO2=88% (immediate escalation)",
         _make_output(hr=110, spo2=88.0, temp=36.7,
                      z_hr=3.0, z_spo2=-4.0, z_temp=0.0,
                      clinical_flags=["spo2_hypoxemia (88.0%)"])),

        # 10. Temp only elevated (weak signal — stays normal)
        ("TEMP ONLY: skin 37.9°C (weak signal, advisory only)",
         _make_output(hr=74, spo2=98.2, temp=37.9,
                      z_hr=0.2, z_spo2=0.0, z_temp=3.0,
                      clinical_flags=["elevated_skin_temp (37.90C)"])),

        # 11. Finger off (rejected frame)
        ("FINGER OFF (frame rejected)",
         _make_output(hr=72, spo2=98.5, temp=36.6,
                      accepted=False, finger_on=False)),

        # 12. Recovery — back to normal
        ("RECOVERY: back to normal",
         _make_output(hr=74, spo2=98.0, temp=36.6,
                      z_hr=0.1, z_spo2=-0.1, z_temp=0.0)),

        # 13. HR very high at rest (>150, bypasses persistence)
        ("EMERGENCY: HR=160 at rest (immediate escalation)",
         _make_output(hr=160, spo2=97.0, temp=36.8,
                      z_hr=8.0, z_spo2=-0.5, z_temp=0.3)),
    ]

    for i, (label, pipeline_out) in enumerate(scenarios):
        result = engine.evaluate(pipeline_out)

        s = result["state"]
        d = result["debug"]
        a = result["alert"]
        f = result["fall_detected"]

        print(f"\n{'─' * 80}")
        print(f"  {i+1}. {label}")
        print(f"{'─' * 80}")

        samp = pipeline_out["sample"]
        z    = pipeline_out["zscores"]
        print(f"  Input → hr={samp['hr']}  spo2={samp['spo2']}  "
              f"temp={samp['temp']}  dyn_acc={samp['dyn_acc_mag']}")
        print(f"  Accel → acc_mag={samp['acc_mag']}  "
              f"acc_x={samp['acc_x']}  acc_y={samp['acc_y']}  "
              f"acc_z={samp['acc_z']}")
        if pipeline_out["baseline_ready"]:
            print(f"  Z-scores → hr={z['hr']:+.1f}  spo2={z['spo2']:+.1f}  "
                  f"temp={z['temp']:+.1f}  hr_stab={z['hr_stability_score']:+.1f}")
        else:
            print(f"  Z-scores → (baseline warming up)")

        if samp.get("clinical_flags"):
            print(f"  Clinical → {samp['clinical_flags']}")

        print()
        print(f"  ┌─ RESULT ──────────────────────────────────────")
        print(f"  │ State:       {s.upper()}")
        print(f"  │ Confidence:  {result['confidence']}")
        print(f"  │ Signals:     {result['contributing_signals'] or '(none)'}")
        print(f"  │ Motion:      {d['motion_state']}")
        print(f"  │")
        print(f"  │ Fall:        stage={f['stage']}  detected={f['detected']}")
        print(f"  │              {f['description']}")
        print(f"  │")
        print(f"  │ Alert [{a['severity'].upper()}]: {a['title']}")
        print(f"  │ \"{a['message']}\"")
        print(f"  └───────────────────────────────────────────────")

        if d.get("skipped"):
            print(f"  (skipped: {d['skip_reason']})")

    # ------------------------------------------------------------------
    # Fall detection demo (multi-frame sequence)
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print(" FALL DETECTION DEMO (multi-frame sequence)")
    print("=" * 80)

    fall_engine = DistressEngine()
    _ts_fall = int(time.time()) + 1000

    fall_sequence = [
        # Frame 1: Normal standing
        ("Normal standing",
         {"timestamp": _ts_fall, "sequence": 1,
          "hr": 72, "hr_stability_score": 28.0, "temp": 36.6, "temp_raw": 36.6,
          "spo2": 98.5, "acc_mag": 0.98, "dyn_acc_mag": 0.02,
          "acc_x": 0.02, "acc_y": -0.01, "acc_z": 0.98,
          "gravity_x": 0.02, "gravity_y": -0.01, "gravity_z": 0.98,
          "finger_on": True, "charging": False, "battery_mv": 3850,
          "die_temp": 34.5, "adc_raw": 112000, "thermal_bias": 0.0,
          "sqi": {"hr": 0.92, "hrv": 0.65, "temp": 0.90,
                  "acc": 0.95, "spo2": 0.90, "ppg": 0.85, "overall": 0.88},
          "accepted": True, "reject_reasons": [], "clinical_flags": []},
         0.0),

        # Frame 2: FREE-FALL (acc_mag drops to 0.3g)
        ("FREE-FALL detected (acc_mag=0.3g)",
         {"timestamp": _ts_fall+1, "sequence": 2,
          "hr": 85, "hr_stability_score": 22.0, "temp": 36.6, "temp_raw": 36.6,
          "spo2": 98.0, "acc_mag": 0.3, "dyn_acc_mag": 0.7,
          "acc_x": 0.1, "acc_y": 0.1, "acc_z": 0.2,
          "gravity_x": 0.02, "gravity_y": -0.01, "gravity_z": 0.94,
          "finger_on": True, "charging": False, "battery_mv": 3850,
          "die_temp": 34.5, "adc_raw": 112000, "thermal_bias": 0.0,
          "sqi": {"hr": 0.80, "hrv": 0.50, "temp": 0.90,
                  "acc": 0.60, "spo2": 0.85, "ppg": 0.70, "overall": 0.72},
          "accepted": True, "reject_reasons": [], "clinical_flags": []},
         1.0),

        # Frame 3: IMPACT (acc_mag spikes to 4.5g — raw acc dominated by dynamic force)
        ("IMPACT spike (acc_mag=4.5g, gravity still transitioning)",
         {"timestamp": _ts_fall+2, "sequence": 3,
          "hr": 110, "hr_stability_score": 15.0, "temp": 36.6, "temp_raw": 36.6,
          "spo2": 97.5, "acc_mag": 4.5, "dyn_acc_mag": 3.5,
          "acc_x": 0.95, "acc_y": 0.05, "acc_z": 0.1,
          "gravity_x": 0.07, "gravity_y": -0.01, "gravity_z": 0.90,
          "finger_on": True, "charging": False, "battery_mv": 3850,
          "die_temp": 34.5, "adc_raw": 112000, "thermal_bias": 0.0,
          "sqi": {"hr": 0.75, "hrv": 0.40, "temp": 0.90,
                  "acc": 0.50, "spo2": 0.80, "ppg": 0.60, "overall": 0.65},
          "accepted": True, "reject_reasons": [], "clinical_flags": []},
         2.0),

        # Frame 4: Post-impact stillness (1s) — gravity settling to horizontal
        ("Post-impact: still (1s) — gravity settling",
         {"timestamp": _ts_fall+3, "sequence": 4,
          "hr": 100, "hr_stability_score": 18.0, "temp": 36.6, "temp_raw": 36.6,
          "spo2": 97.8, "acc_mag": 0.99, "dyn_acc_mag": 0.02,
          "acc_x": 0.95, "acc_y": 0.05, "acc_z": 0.1,
          "gravity_x": 0.50, "gravity_y": 0.02, "gravity_z": 0.52,
          "finger_on": True, "charging": False, "battery_mv": 3850,
          "die_temp": 34.5, "adc_raw": 112000, "thermal_bias": 0.0,
          "sqi": {"hr": 0.88, "hrv": 0.55, "temp": 0.90,
                  "acc": 0.90, "spo2": 0.88, "ppg": 0.80, "overall": 0.82},
          "accepted": True, "reject_reasons": [], "clinical_flags": []},
         3.0),

        # Frame 5: Post-impact stillness (2s) — gravity settled to horizontal
        ("Post-impact: still (2s) — gravity settled",
         {"timestamp": _ts_fall+4, "sequence": 5,
          "hr": 95, "hr_stability_score": 20.0, "temp": 36.6, "temp_raw": 36.6,
          "spo2": 98.0, "acc_mag": 0.99, "dyn_acc_mag": 0.01,
          "acc_x": 0.95, "acc_y": 0.05, "acc_z": 0.1,
          "gravity_x": 0.80, "gravity_y": 0.04, "gravity_z": 0.25,
          "finger_on": True, "charging": False, "battery_mv": 3850,
          "die_temp": 34.5, "adc_raw": 112000, "thermal_bias": 0.0,
          "sqi": {"hr": 0.90, "hrv": 0.60, "temp": 0.90,
                  "acc": 0.95, "spo2": 0.90, "ppg": 0.85, "overall": 0.85},
          "accepted": True, "reject_reasons": [], "clinical_flags": []},
         4.0),

        # Frame 6: Post-impact stillness (3s) — gravity fully settled
        ("Post-impact: still (3s) — gravity settled",
         {"timestamp": _ts_fall+5, "sequence": 6,
          "hr": 92, "hr_stability_score": 22.0, "temp": 36.6, "temp_raw": 36.6,
          "spo2": 98.0, "acc_mag": 0.99, "dyn_acc_mag": 0.01,
          "acc_x": 0.95, "acc_y": 0.05, "acc_z": 0.1,
          "gravity_x": 0.90, "gravity_y": 0.04, "gravity_z": 0.15,
          "finger_on": True, "charging": False, "battery_mv": 3850,
          "die_temp": 34.5, "adc_raw": 112000, "thermal_bias": 0.0,
          "sqi": {"hr": 0.92, "hrv": 0.65, "temp": 0.90,
                  "acc": 0.95, "spo2": 0.90, "ppg": 0.85, "overall": 0.88},
          "accepted": True, "reject_reasons": [], "clinical_flags": []},
         5.0),

        # Frame 7: Post-impact stillness (4s) → CONFIRMED
        ("Post-impact: still (4s) → FALL CONFIRMED",
         {"timestamp": _ts_fall+6, "sequence": 7,
          "hr": 88, "hr_stability_score": 24.0, "temp": 36.6, "temp_raw": 36.6,
          "spo2": 98.0, "acc_mag": 0.99, "dyn_acc_mag": 0.01,
          "acc_x": 0.95, "acc_y": 0.05, "acc_z": 0.1,
          "gravity_x": 0.93, "gravity_y": 0.04, "gravity_z": 0.12,
          "finger_on": True, "charging": False, "battery_mv": 3850,
          "die_temp": 34.5, "adc_raw": 112000, "thermal_bias": 0.0,
          "sqi": {"hr": 0.92, "hrv": 0.65, "temp": 0.90,
                  "acc": 0.95, "spo2": 0.90, "ppg": 0.85, "overall": 0.88},
          "accepted": True, "reject_reasons": [], "clinical_flags": []},
         6.0),
    ]

    # Use a fixed base time so fall detection timing works correctly
    _base_time = time.time()

    # Monkey-patch time.time for the fall demo so frames have correct spacing
    _original_time = time.time

    for i, (label, samp_data, t_offset) in enumerate(fall_sequence):
        pipeline_out = {
            "sample": samp_data,
            "zscores": {"hr": 0.0, "hr_stability_score": 0.0, "temp": 0.0,
                        "spo2": 0.0, "acc_mag": 0.0},
            "baseline_ready": True,
            "window": {
                "window_n": 25,
                "hr_mean": samp_data["hr"], "hr_var": 2.0,
                "hr_min": samp_data["hr"]-2, "hr_max": samp_data["hr"]+2,
                "hr_stability_score_mean": 28.0, "hr_stability_score_var": 4.0,
                "hr_stability_score_min": 24.0, "hr_stability_score_max": 32.0,
                "temp_mean": 36.6, "temp_var": 0.01,
                "temp_min": 36.5, "temp_max": 36.7,
                "spo2_mean": 98.0, "spo2_var": 0.5,
                "spo2_min": 97.5, "spo2_max": 98.5,
                "acc_mag_mean": 0.98, "acc_mag_var": 0.001,
                "acc_mag_min": 0.97, "acc_mag_max": 0.99,
            },
        }

        # Override time.time for controlled fall detection timing
        time.time = lambda t=t_offset: _base_time + t
        result = fall_engine.evaluate(pipeline_out)

        f = result["fall_detected"]
        a = result["alert"]

        print(f"\n{'─' * 60}")
        print(f"  Frame {i+1}: {label}")
        print(f"{'─' * 60}")
        print(f"  acc_mag={samp_data['acc_mag']}  dyn_acc={samp_data['dyn_acc_mag']}  "
              f"orientation=({samp_data['acc_x']}, {samp_data['acc_y']}, {samp_data['acc_z']})")
        print(f"  ┌─ FALL DETECTION ────────────────────────────")
        print(f"  │ Stage:       {f['stage']}")
        print(f"  │ Detected:    {f['detected']}")
        print(f"  │ Confidence:  {f['confidence']}")
        print(f"  │ Description: {f['description']}")
        print(f"  │")
        print(f"  │ Alert [{a['severity'].upper()}]: {a['title']}")
        print(f"  │ \"{a['message']}\"")
        print(f"  └─────────────────────────────────────────────")

    # Restore time.time
    time.time = _original_time

    # ------------------------------------------------------------------
    # Geospatial Safety Risk Demo
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 80}")
    print(" GEOSPATIAL SAFETY RISK DEMO")
    print(" (unified engine: physiology + context)")
    print("=" * 80)

    unified = UnifiedSafetyEngine()

    # Disable persistence for demo
    PERSISTENCE_STRESS_S = 0
    PERSISTENCE_DISTRESS_S = 0

    geo_scenarios = [
        # 1. Normal at home — daytime
        (
            "Normal at home, daytime",
            _make_output(hr=72, spo2=98.5, temp=36.6,
                         z_hr=0.3, z_spo2=0.1, z_temp=-0.2),
            {
                "latitude": 17.385,  "longitude": 78.486,
                "timestamp": 1700000000,
                "speed_kmph": 0.0, "heading": 0.0,
                "is_home_zone": True, "is_work_zone": False,
                "is_known_area": True, "is_unfamiliar_area": False,
                "distance_from_home_km": 0.1,
                "hour_of_day": 14, "is_night": False,
                "is_stationary": True, "is_walking": False,
                "is_vehicle_like_motion": False,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": True,
            },
        ),

        # 2. Stressed at gym (known area, walking, daytime) → suppressed
        (
            "Stressed at gym — known area, daytime (SUPPRESSED)",
            _make_output(hr=110, spo2=97.0, temp=37.0,
                         z_hr=3.0, z_spo2=-0.3, z_temp=0.5),
            {
                "latitude": 17.390,  "longitude": 78.490,
                "timestamp": 1700000000,
                "speed_kmph": 2.0, "heading": 90.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": True, "is_unfamiliar_area": False,
                "distance_from_home_km": 1.5,
                "hour_of_day": 10, "is_night": False,
                "is_stationary": False, "is_walking": True,
                "is_vehicle_like_motion": False,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": True,
            },
        ),

        # 3. Distress at night in unfamiliar area
        (
            "DISTRESS at night in unfamiliar area",
            _make_output(hr=130, spo2=91.0, temp=36.8,
                         z_hr=5.0, z_spo2=-2.5, z_temp=0.5,
                         clinical_flags=["spo2_clinical_concern (91.0%)"]),
            {
                "latitude": 17.500,  "longitude": 78.600,
                "timestamp": 1700000000,
                "speed_kmph": 0.0, "heading": 0.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": False, "is_unfamiliar_area": True,
                "distance_from_home_km": 15.0,
                "hour_of_day": 23, "is_night": True,
                "is_stationary": True, "is_walking": False,
                "is_vehicle_like_motion": False,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": True,
            },
        ),

        # 4. Distress in vehicle with sudden stop
        (
            "DISTRESS in vehicle — sudden stop, unfamiliar area",
            _make_output(hr=125, spo2=93.0, temp=36.7,
                         z_hr=4.5, z_spo2=-2.0, z_temp=0.2,
                         clinical_flags=["spo2_clinical_concern (93.0%)"]),
            {
                "latitude": 17.600,  "longitude": 78.700,
                "timestamp": 1700000000,
                "speed_kmph": 0.0, "heading": 180.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": False, "is_unfamiliar_area": True,
                "distance_from_home_km": 25.0,
                "hour_of_day": 1, "is_night": True,
                "is_stationary": True, "is_walking": False,
                "is_vehicle_like_motion": True,
                "sudden_route_change": True, "sudden_stop": True,
                "phone_connected": True,
            },
        ),

        # 5. Stress at work — daytime, known zone
        (
            "Stress at work — daytime, known zone (SUPPRESSED)",
            _make_output(hr=100, spo2=97.5, temp=36.7,
                         z_hr=2.5, z_spo2=-0.2, z_temp=0.1),
            {
                "latitude": 17.385,  "longitude": 78.486,
                "timestamp": 1700000000,
                "speed_kmph": 0.0, "heading": 0.0,
                "is_home_zone": False, "is_work_zone": True,
                "is_known_area": True, "is_unfamiliar_area": False,
                "distance_from_home_km": 3.0,
                "hour_of_day": 11, "is_night": False,
                "is_stationary": True, "is_walking": False,
                "is_vehicle_like_motion": False,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": True,
            },
        ),

        # 6. Distress + night + unfamiliar + phone disconnected (CRITICAL)
        (
            "CRITICAL: distress + night + unfamiliar + no phone",
            _make_output(hr=140, spo2=89.0, temp=36.8,
                         z_hr=6.0, z_spo2=-3.5, z_temp=0.3,
                         clinical_flags=["spo2_hypoxemia (89.0%)"]),
            {
                "latitude": 17.800,  "longitude": 78.900,
                "timestamp": 1700000000,
                "speed_kmph": 0.0, "heading": 0.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": False, "is_unfamiliar_area": True,
                "distance_from_home_km": 50.0,
                "hour_of_day": 2, "is_night": True,
                "is_stationary": True, "is_walking": False,
                "is_vehicle_like_motion": False,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": False,
            },
        ),

        # 7. Normal walking in known area at evening
        (
            "Normal — evening walk in known area",
            _make_output(hr=82, spo2=98.0, temp=36.6,
                         z_hr=0.8, z_spo2=-0.1, z_temp=0.0),
            {
                "latitude": 17.385,  "longitude": 78.486,
                "timestamp": 1700000000,
                "speed_kmph": 4.5, "heading": 45.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": True, "is_unfamiliar_area": False,
                "distance_from_home_km": 0.5,
                "hour_of_day": 20, "is_night": True,
                "is_stationary": False, "is_walking": True,
                "is_vehicle_like_motion": False,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": True,
            },
        ),

        # 8. Stress + route deviation at night
        (
            "Stress + route deviation at night",
            _make_output(hr=105, spo2=96.5, temp=36.7,
                         z_hr=2.8, z_spo2=-0.8, z_temp=0.1),
            {
                "latitude": 17.450,  "longitude": 78.550,
                "timestamp": 1700000000,
                "speed_kmph": 35.0, "heading": 270.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": False, "is_unfamiliar_area": True,
                "distance_from_home_km": 8.0,
                "hour_of_day": 22, "is_night": True,
                "is_stationary": False, "is_walking": False,
                "is_vehicle_like_motion": True,
                "sudden_route_change": True, "sudden_stop": False,
                "phone_connected": True,
            },
        ),

        # 9. Travelling in vehicle, normal physiology, daytime
        (
            "Normal — travelling in vehicle, daytime",
            _make_output(hr=75, spo2=98.0, temp=36.6,
                         z_hr=0.2, z_spo2=0.0, z_temp=-0.1),
            {
                "latitude": 17.400,  "longitude": 78.500,
                "timestamp": 1700000000,
                "speed_kmph": 60.0, "heading": 90.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": True, "is_unfamiliar_area": False,
                "distance_from_home_km": 12.0,
                "hour_of_day": 15, "is_night": False,
                "is_stationary": False, "is_walking": False,
                "is_vehicle_like_motion": True,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": True,
            },
        ),

        # 10. Distress while travelling — vehicle, night
        (
            "DISTRESS while travelling — vehicle, night, unfamiliar",
            _make_output(hr=135, spo2=92.0, temp=36.8,
                         z_hr=5.5, z_spo2=-2.2, z_temp=0.4,
                         clinical_flags=["spo2_clinical_concern (92.0%)"]),
            {
                "latitude": 17.650,  "longitude": 78.750,
                "timestamp": 1700000000,
                "speed_kmph": 80.0, "heading": 180.0,
                "is_home_zone": False, "is_work_zone": False,
                "is_known_area": False, "is_unfamiliar_area": True,
                "distance_from_home_km": 35.0,
                "hour_of_day": 0, "is_night": True,
                "is_stationary": False, "is_walking": False,
                "is_vehicle_like_motion": True,
                "sudden_route_change": False, "sudden_stop": False,
                "phone_connected": True,
            },
        ),
    ]

    for i, (label, pipeline_out, geo_ctx) in enumerate(geo_scenarios):
        result = unified.evaluate(pipeline_out, geo_context=geo_ctx)

        safety = result["safety"]
        a = result["alert"]
        s = result["state"]

        print(f"\n{'─' * 80}")
        print(f"  {i+1}. {label}")
        print(f"{'─' * 80}")

        samp = pipeline_out["sample"]
        print(f"  Physio → hr={samp['hr']}  spo2={samp['spo2']}  "
              f"temp={samp['temp']}  state={s.upper()}")
        print(f"  Context → hour={geo_ctx['hour_of_day']}  "
              f"night={geo_ctx['is_night']}  "
              f"home={geo_ctx['is_home_zone']}  "
              f"unfamiliar={geo_ctx['is_unfamiliar_area']}  "
              f"dist={geo_ctx['distance_from_home_km']}km")
        if geo_ctx.get("sudden_route_change") or geo_ctx.get("sudden_stop"):
            print(f"  Movement → route_change={geo_ctx['sudden_route_change']}  "
                  f"sudden_stop={geo_ctx['sudden_stop']}  "
                  f"vehicle={geo_ctx['is_vehicle_like_motion']}")
        if not geo_ctx.get("phone_connected"):
            print(f"  ⚠ Phone disconnected")

        print()
        print(f"  ┌─ SAFETY ASSESSMENT ────────────────────────────")
        print(f"  │ Risk Level:  {safety['risk_level'].upper()}")
        print(f"  │ Risk Score:  {safety['risk_score']}")
        print(f"  │ Reasoning:")
        for r in safety["reasoning"]:
            print(f"  │   • {r}")
        print(f"  │")
        print(f"  │ Action: {safety['recommended_action']}")
        print(f"  │")
        print(f"  │ Alert [{a['severity'].upper()}]: {a['title']}")
        print(f"  │ \"{a['message']}\"")
        print(f"  └─────────────────────────────────────────────────")

    print(f"\n{'=' * 80}")
    print(" Distress Engine Stats")
    print("=" * 80)
    print(json.dumps(engine.stats(), indent=2))

    print(f"\n{'=' * 80}")
    print(" Unified Safety Engine Stats")
    print("=" * 80)
    print(json.dumps(unified.stats(), indent=2))
