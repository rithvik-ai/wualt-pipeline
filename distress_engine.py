"""
WUALT — Rule-Based Physiological Distress Detection Engine
============================================================

A standalone, interpretable rule-based engine that consumes the output of
preprocessing_pipeline.AnomalyInputBuilder.step() and produces a
human-friendly stress/distress assessment with STAR-principle alerts.

Design goals:
    - Real-time:      < 100 ms per evaluation (trivially satisfied, O(1))
    - Interpretable:  every decision is traceable — no ML, no black boxes
    - Robust:         handles noisy wearable data, cold-start, motion artifacts
    - Human-first:    alerts are calm, clear, actionable — no medical jargon

---------------------------------------------------------------------------
INPUT  (exactly what AnomalyInputBuilder.step() returns)
---------------------------------------------------------------------------
    {
        "sample":         { ... },    # processed frame from preprocessing
        "zscores":        { ... },    # personal baseline deviations
        "baseline_ready": bool,
        "window":         { ... },    # 30-sec rolling stats
    }

---------------------------------------------------------------------------
OUTPUT
---------------------------------------------------------------------------
    {
        "state":       "normal" | "stress" | "distress",
        "confidence":  float 0.0..1.0,
        "contributing_signals": [str, ...],
        "alert": {
            "title":    str,
            "message":  str,
            "severity": "low" | "medium" | "high",
        },
        "debug": {
            "flags":         { signal: bool },
            "scores":        { signal: float },
            "persistence_s": int,
            "motion_state":  str,
        },
    }

---------------------------------------------------------------------------
DETECTION LOGIC
---------------------------------------------------------------------------
    1. GATE:    skip if frame rejected or baseline not ready (absolute only)
    2. MOTION:  detect exercise to avoid false stress alerts
    3. FLAG:    per-signal boolean flags from z-scores + absolute thresholds
    4. SCORE:   weighted signal scoring (temp is weak/supporting)
    5. PERSIST: require condition to hold for >= 60s before promoting state
    6. DECIDE:  normal / stress / distress based on flag count + persistence
    7. ALERT:   generate calm, actionable, STAR-principle user message

Run directly for a demo:
    python distress_engine.py
"""

from __future__ import annotations

import time
import random
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Deque, Dict, List, Optional, Tuple


# ===========================================================================
# 0. CONFIGURATION
# ===========================================================================

# --- Z-score thresholds for flagging ---
# Direction matters: HR ↑ = stress, HRV ↓ = stress, SpO2 ↓ = stress
ZSCORE_THRESHOLDS = {
    "hr":        2.0,    # z > +2.0 → flag (elevated)
    "hrv_rmssd": -1.5,   # z < -1.5 → flag (suppressed — note: inverted)
    "spo2":      -2.0,   # z < -2.0 → flag (desaturation)
    "temp":      2.5,    # z > +2.5 → flag (weak signal, high threshold)
}

# --- Absolute thresholds (always active, even during cold start) ---
# These fire regardless of baseline — universal physiological limits.
ABSOLUTE_THRESHOLDS = {
    "hr_high":      120.0,   # bpm — clearly elevated for a resting person
    "hr_very_high": 150.0,   # bpm — concerning even during exercise
    "spo2_low":      94.0,   # % — clinical concern threshold (BTS/WHO)
    "spo2_very_low": 90.0,   # % — hypoxemia / medical emergency
    "temp_high":     37.8,   # °C skin — possible fever (skin is 4-7°C below core)
}

# --- Signal weights for confidence scoring ---
# Temperature is deliberately weak — it's a supporting signal, not primary.
SIGNAL_WEIGHTS = {
    "hr":        0.35,
    "hrv_rmssd": 0.25,
    "spo2":      0.30,
    "temp":      0.10,
}

# --- Motion detection ---
# Dynamic acceleration magnitude thresholds for activity classification.
MOTION_THRESHOLDS = {
    "still":    0.05,   # dyn_acc_mag < 0.05 → user is still
    "walking":  0.25,   # 0.05 .. 0.25 → light motion
    "active":   0.50,   # 0.25 .. 0.50 → moderate activity
    # > 0.50 → vigorous exercise
}

# --- Persistence ---
PERSISTENCE_STRESS_S   = 60    # seconds before promoting to "stress"
PERSISTENCE_DISTRESS_S = 60    # seconds before promoting to "distress"

# --- SQI gate ---
MIN_SQI_FOR_DETECTION = 0.5    # overall SQI below this → skip detection


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
    hr:        bool = False
    hrv_rmssd: bool = False
    spo2:      bool = False
    temp:      bool = False

    hr_score:        float = 0.0
    hrv_rmssd_score: float = 0.0
    spo2_score:      float = 0.0
    temp_score:      float = 0.0

    source: str = "none"   # "zscore", "absolute", "clinical_flag"

    @property
    def count(self) -> int:
        return sum([self.hr, self.hrv_rmssd, self.spo2, self.temp])

    @property
    def triggered(self) -> List[str]:
        out = []
        if self.hr:        out.append("hr")
        if self.hrv_rmssd: out.append("hrv_rmssd")
        if self.spo2:      out.append("spo2")
        if self.temp:      out.append("temp")
        return out

    def weighted_score(self) -> float:
        """Weighted confidence score from triggered signals."""
        total = 0.0
        if self.hr:
            total += SIGNAL_WEIGHTS["hr"] * min(1.0, self.hr_score)
        if self.hrv_rmssd:
            total += SIGNAL_WEIGHTS["hrv_rmssd"] * min(1.0, self.hrv_rmssd_score)
        if self.spo2:
            total += SIGNAL_WEIGHTS["spo2"] * min(1.0, self.spo2_score)
        if self.temp:
            total += SIGNAL_WEIGHTS["temp"] * min(1.0, self.temp_score)
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

        # HRV: suppressed z-score → stress (INVERTED direction)
        z_hrv = zscores.get("hrv_rmssd", 0.0)
        if z_hrv < ZSCORE_THRESHOLDS["hrv_rmssd"]:
            flags.hrv_rmssd = True
            flags.hrv_rmssd_score = (
                abs(z_hrv - ZSCORE_THRESHOLDS["hrv_rmssd"]) / 2.0 + 0.5
            )

        # SpO2: suppressed z-score → distress (INVERTED direction)
        z_spo2 = zscores.get("spo2", 0.0)
        if z_spo2 < ZSCORE_THRESHOLDS["spo2"]:
            flags.spo2 = True
            flags.spo2_score = (
                abs(z_spo2 - ZSCORE_THRESHOLDS["spo2"]) / 2.0 + 0.5
            )

        # Temp: elevated z-score → weak supporting signal
        z_temp = zscores.get("temp", 0.0)
        if z_temp > ZSCORE_THRESHOLDS["temp"]:
            flags.temp = True
            flags.temp_score = (z_temp - ZSCORE_THRESHOLDS["temp"]) / 3.0 + 0.3

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
    clinical = sample.get("clinical_flags", [])
    for cf in clinical:
        if "spo2_hypoxemia" in cf and not flags.spo2:
            flags.spo2 = True
            flags.spo2_score = max(flags.spo2_score, 0.85)
            flags.source = "clinical_flag"
        elif "spo2_clinical_concern" in cf and not flags.spo2:
            flags.spo2 = True
            flags.spo2_score = max(flags.spo2_score, 0.55)
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
# 5. ALERT SELECTOR
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
        if flags.temp and flags.count == 1:
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
        self._cycle = 0   # for alert message rotation
        self._last_state = "normal"
        self._state_history: Deque[str] = deque(maxlen=120)  # ~2 min at 1 Hz

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
        # Step 4: Raw state determination
        # ------------------------------------------------------------------
        raw_state = "normal"
        if flags.count >= 2:
            raw_state = "distress"
        elif flags.count == 1:
            # Temp alone is a "supporting" signal — not enough for stress
            if flags.temp and flags.count == 1:
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
        # SpO2 < 90% or HR > 150 → immediate escalation
        spo2_val = sample.get("spo2")
        hr_val   = sample.get("hr")
        if spo2_val is not None and spo2_val <= ABSOLUTE_THRESHOLDS["spo2_very_low"]:
            final_state = "distress"
            persistence_s = max(persistence_s, 1.0)
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
            "debug": {
                "flags": {
                    "hr":        flags.hr,
                    "hrv_rmssd": flags.hrv_rmssd,
                    "spo2":      flags.spo2,
                    "temp":      flags.temp,
                },
                "scores": {
                    "hr":        round(flags.hr_score, 3),
                    "hrv_rmssd": round(flags.hrv_rmssd_score, 3),
                    "spo2":      round(flags.spo2_score, 3),
                    "temp":      round(flags.temp_score, 3),
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
# 7. DEMO — run:  python distress_engine.py
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
                "hr": hr, "hrv_rmssd": 28.0, "temp": temp, "temp_raw": temp,
                "spo2": spo2, "acc_mag": 0.98, "dyn_acc_mag": dyn_acc,
                "acc_x": 0.02, "acc_y": -0.01, "acc_z": 0.98,
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
                "hr": z_hr, "hrv_rmssd": z_hrv,
                "temp": z_temp, "spo2": z_spo2, "acc_mag": 0.0,
            },
            "baseline_ready": baseline_ready,
            "window": {
                "window_n": window_n,
                "hr_mean": hr, "hr_var": 2.0, "hr_min": hr - 2, "hr_max": hr + 2,
                "hrv_rmssd_mean": 28.0, "hrv_rmssd_var": 4.0,
                "hrv_rmssd_min": 24.0, "hrv_rmssd_max": 32.0,
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

        # 5. Stress — SpO2 dipped (clinical flag)
        ("STRESS: SpO2=93% (clinical concern flag)",
         _make_output(hr=80, spo2=93.0, temp=36.6,
                      z_hr=0.5, z_spo2=-1.0, z_temp=0.0,
                      clinical_flags=["spo2_clinical_concern (93.0%)"])),

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

        print(f"\n{'─' * 80}")
        print(f"  {i+1}. {label}")
        print(f"{'─' * 80}")

        samp = pipeline_out["sample"]
        z    = pipeline_out["zscores"]
        print(f"  Input → hr={samp['hr']}  spo2={samp['spo2']}  "
              f"temp={samp['temp']}  dyn_acc={samp['dyn_acc_mag']}")
        if pipeline_out["baseline_ready"]:
            print(f"  Z-scores → hr={z['hr']:+.1f}  spo2={z['spo2']:+.1f}  "
                  f"temp={z['temp']:+.1f}  hrv={z['hrv_rmssd']:+.1f}")
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
        print(f"  │ Alert [{a['severity'].upper()}]: {a['title']}")
        print(f"  │ \"{a['message']}\"")
        print(f"  └───────────────────────────────────────────────")

        if d.get("skipped"):
            print(f"  (skipped: {d['skip_reason']})")

    print(f"\n{'=' * 80}")
    print(" Engine Stats")
    print("=" * 80)
    print(json.dumps(engine.stats(), indent=2))
