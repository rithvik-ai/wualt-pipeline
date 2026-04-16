"""
WUALT — Synthetic Dataset Generator for Rule-Based Distress Engine
====================================================================

Generates a comprehensive, physiologically realistic synthetic dataset
that can be used to:
    1. Calibrate thresholds in the distress engine
    2. Validate detection accuracy across edge cases
    3. Stress-test persistence logic and state transitions
    4. Benchmark false-positive / false-negative rates
    5. Simulate diverse user demographics (age, fitness, health)

Output:
    - synthetic_dataset.csv           — 10,000+ labelled frames
    - synthetic_scenarios.csv         — multi-frame scenario sequences
    - synthetic_dataset_summary.json  — statistics and metadata

Each row represents one pipeline_output frame with ground-truth labels.

Run:
    python generate_synthetic_dataset.py

Dependencies: none beyond Python stdlib
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42

# ===========================================================================
# 1. DEMOGRAPHIC PROFILES — physiological baselines vary by person
# ===========================================================================

@dataclass
class UserProfile:
    """Simulated user with realistic resting baselines."""
    profile_id: str
    label: str
    age: int
    sex: str                 # "M" / "F"
    fitness: str             # "sedentary" / "moderate" / "athletic"
    resting_hr: float        # bpm
    resting_hrv: float       # ms RMSSD
    resting_spo2: float      # %
    resting_temp: float      # °C (skin)
    hr_std: float            # natural HR variability
    hrv_std: float
    spo2_std: float
    temp_std: float
    notes: str = ""

PROFILES: List[UserProfile] = [
    # ── Healthy adults ─────────────────────────────────────────────
    UserProfile("H01", "Healthy young male",         25, "M", "moderate",
                72,  35, 98.2, 36.5,  4.0, 6.0, 0.5, 0.15),
    UserProfile("H02", "Healthy young female",       28, "F", "moderate",
                76,  32, 98.0, 36.6,  4.5, 5.5, 0.5, 0.15),
    UserProfile("H03", "Athletic male",              22, "M", "athletic",
                58,  52, 98.8, 36.3,  3.0, 8.0, 0.3, 0.12),
    UserProfile("H04", "Athletic female",            24, "F", "athletic",
                62,  48, 98.6, 36.4,  3.5, 7.5, 0.4, 0.12),
    UserProfile("H05", "Sedentary male",             35, "M", "sedentary",
                82,  22, 97.5, 36.7,  5.0, 4.0, 0.6, 0.18),
    UserProfile("H06", "Sedentary female",           40, "F", "sedentary",
                80,  24, 97.8, 36.8,  5.5, 4.5, 0.6, 0.18),

    # ── Older adults ───────────────────────────────────────────────
    UserProfile("H07", "Older male (60)",            60, "M", "moderate",
                70,  20, 97.0, 36.4,  5.0, 3.5, 0.7, 0.15),
    UserProfile("H08", "Older female (65)",          65, "F", "sedentary",
                74,  18, 96.8, 36.5,  5.5, 3.0, 0.8, 0.18),

    # ── Adolescents ────────────────────────────────────────────────
    UserProfile("H09", "Teen male (16)",             16, "M", "moderate",
                78,  38, 98.5, 36.5,  5.0, 7.0, 0.4, 0.14),
    UserProfile("H10", "Teen female (17)",           17, "F", "moderate",
                80,  36, 98.3, 36.6,  5.5, 6.5, 0.5, 0.14),

    # ── Special populations (higher false-positive risk) ──────────
    UserProfile("S01", "Anxiety disorder",           30, "F", "moderate",
                85,  20, 97.8, 36.7,  8.0, 5.0, 0.5, 0.20,
                "Naturally elevated resting HR, lower HRV"),
    UserProfile("S02", "COPD patient",               55, "M", "sedentary",
                78,  19, 94.5, 36.6,  5.0, 3.5, 1.2, 0.18,
                "Chronically low SpO2, should not always alarm"),
    UserProfile("S03", "Obese, hypertensive",        45, "M", "sedentary",
                88,  16, 97.0, 37.0,  6.0, 3.0, 0.7, 0.20,
                "Higher baseline HR and temp"),
    UserProfile("S04", "Pregnant female (32wk)",     29, "F", "moderate",
                90,  22, 97.5, 36.9,  6.0, 4.0, 0.6, 0.20,
                "Elevated resting HR, slightly warmer"),
    UserProfile("S05", "Athlete w/ low resting HR",  26, "M", "athletic",
                48,  65, 99.0, 36.2,  2.5, 10.0, 0.3, 0.10,
                "Very low resting HR, very high HRV — exercise jumps are large"),
]


# ===========================================================================
# 2. SCENARIO TEMPLATES — labelled physiological situations
# ===========================================================================

@dataclass
class ScenarioTemplate:
    """Defines a physiological scenario with expected ground-truth."""
    name: str
    category: str               # "normal" / "stress" / "distress" / "exercise" / "edge"
    expected_state: str         # ground-truth label
    description: str
    # Deltas from profile baseline (additive)
    hr_delta: Tuple[float, float]        # (mean_delta, delta_std)
    hrv_delta: Tuple[float, float]       # (mean_delta, delta_std) — negative = suppression
    spo2_delta: Tuple[float, float]
    temp_delta: Tuple[float, float]
    dyn_acc: Tuple[float, float]         # (mean, std) — motion level
    # Flags
    finger_on: bool = True
    accepted: bool = True
    clinical_flags_gen: Optional[str] = None  # "spo2_concern" / "spo2_hypoxemia" / "temp_elevated"
    duration_frames: int = 1            # how many frames to generate per instance
    weight: float = 1.0                 # sampling weight


SCENARIOS: List[ScenarioTemplate] = [
    # ═══════════════════════════════════════════════════════════════
    # NORMAL — baseline / resting / everyday
    # ═══════════════════════════════════════════════════════════════
    ScenarioTemplate(
        "resting_calm", "normal", "normal",
        "User at rest, all vitals at baseline",
        hr_delta=(0, 3), hrv_delta=(0, 4), spo2_delta=(0, 0.3),
        temp_delta=(0, 0.1), dyn_acc=(0.01, 0.01),
        weight=4.0, duration_frames=5,
    ),
    ScenarioTemplate(
        "resting_slight_hr_up", "normal", "normal",
        "Mild HR elevation within normal range (stood up, drank coffee)",
        hr_delta=(8, 4), hrv_delta=(-3, 3), spo2_delta=(0, 0.3),
        temp_delta=(0, 0.1), dyn_acc=(0.02, 0.01),
        weight=2.0, duration_frames=3,
    ),
    ScenarioTemplate(
        "sleeping", "normal", "normal",
        "User sleeping — low HR, high HRV, stable SpO2",
        hr_delta=(-8, 2), hrv_delta=(8, 5), spo2_delta=(0.2, 0.2),
        temp_delta=(-0.3, 0.1), dyn_acc=(0.005, 0.003),
        weight=2.5, duration_frames=10,
    ),
    ScenarioTemplate(
        "light_walking", "normal", "normal",
        "Casual walking — slight HR bump, motion detected",
        hr_delta=(15, 5), hrv_delta=(-5, 3), spo2_delta=(0, 0.3),
        temp_delta=(0.1, 0.1), dyn_acc=(0.15, 0.05),
        weight=2.0, duration_frames=5,
    ),
    ScenarioTemplate(
        "deep_breathing", "normal", "normal",
        "Relaxation exercise — HRV transiently high",
        hr_delta=(-3, 3), hrv_delta=(12, 6), spo2_delta=(0.5, 0.3),
        temp_delta=(0, 0.1), dyn_acc=(0.01, 0.005),
        weight=1.0, duration_frames=3,
    ),
    ScenarioTemplate(
        "post_meal", "normal", "normal",
        "After eating — slightly elevated HR, temp",
        hr_delta=(10, 4), hrv_delta=(-4, 3), spo2_delta=(0, 0.3),
        temp_delta=(0.3, 0.1), dyn_acc=(0.02, 0.01),
        weight=1.5, duration_frames=3,
    ),
    ScenarioTemplate(
        "ambient_temp_shift", "normal", "normal",
        "Moved to warmer environment — skin temp rises naturally",
        hr_delta=(3, 2), hrv_delta=(0, 3), spo2_delta=(0, 0.3),
        temp_delta=(0.8, 0.2), dyn_acc=(0.02, 0.01),
        weight=1.0, duration_frames=3,
    ),

    # ═══════════════════════════════════════════════════════════════
    # EXERCISE — elevated HR is expected, NOT stress
    # ═══════════════════════════════════════════════════════════════
    ScenarioTemplate(
        "moderate_exercise", "exercise", "normal",
        "Jogging / cycling — HR elevated, high motion, SpO2 normal",
        hr_delta=(50, 10), hrv_delta=(-15, 5), spo2_delta=(-0.5, 0.5),
        temp_delta=(0.5, 0.2), dyn_acc=(0.55, 0.15),
        weight=2.0, duration_frames=8,
    ),
    ScenarioTemplate(
        "vigorous_exercise", "exercise", "normal",
        "Sprinting / HIIT — HR very high, strong motion",
        hr_delta=(80, 12), hrv_delta=(-20, 5), spo2_delta=(-1.0, 0.5),
        temp_delta=(0.8, 0.3), dyn_acc=(0.75, 0.20),
        weight=1.5, duration_frames=6,
    ),
    ScenarioTemplate(
        "exercise_cooldown", "exercise", "normal",
        "Post-exercise cooldown — HR gradually dropping, still moving slightly",
        hr_delta=(25, 8), hrv_delta=(-8, 4), spo2_delta=(0, 0.3),
        temp_delta=(0.4, 0.2), dyn_acc=(0.12, 0.05),
        weight=1.5, duration_frames=5,
    ),
    ScenarioTemplate(
        "exercise_warmup", "exercise", "normal",
        "Starting exercise — HR climbing, motion increasing",
        hr_delta=(20, 8), hrv_delta=(-8, 4), spo2_delta=(0, 0.3),
        temp_delta=(0.1, 0.1), dyn_acc=(0.35, 0.10),
        weight=1.0, duration_frames=4,
    ),

    # ═══════════════════════════════════════════════════════════════
    # STRESS — single signal deviation (expect "stress" label)
    # ═══════════════════════════════════════════════════════════════
    ScenarioTemplate(
        "stress_hr_mild", "stress", "stress",
        "Mild psychological stress — HR elevated, other signals normal",
        hr_delta=(25, 5), hrv_delta=(-8, 3), spo2_delta=(0, 0.3),
        temp_delta=(0.1, 0.1), dyn_acc=(0.02, 0.01),
        weight=2.0, duration_frames=5,
    ),
    ScenarioTemplate(
        "stress_hr_moderate", "stress", "stress",
        "Moderate stress — HR clearly above baseline at rest",
        hr_delta=(40, 6), hrv_delta=(-12, 4), spo2_delta=(0, 0.4),
        temp_delta=(0.2, 0.1), dyn_acc=(0.02, 0.01),
        weight=2.0, duration_frames=5,
    ),
    ScenarioTemplate(
        "stress_hr_absolute", "stress", "stress",
        "HR above absolute 120bpm threshold — clear stress at rest",
        hr_delta=(45, 5), hrv_delta=(-10, 4), spo2_delta=(0, 0.3),
        temp_delta=(0.1, 0.1), dyn_acc=(0.02, 0.01),
        weight=1.5, duration_frames=4,
    ),
    ScenarioTemplate(
        "stress_spo2_mild", "stress", "stress",
        "SpO2 dip below 94% — clinical concern, single signal",
        hr_delta=(5, 3), hrv_delta=(-2, 3), spo2_delta=(-4.0, 0.8),
        temp_delta=(0, 0.1), dyn_acc=(0.01, 0.005),
        clinical_flags_gen="spo2_concern",
        weight=1.5, duration_frames=4,
    ),
    ScenarioTemplate(
        "stress_hrv_suppressed", "stress", "stress",
        "HRV suppression from chronic stress — HR normal, HRV very low",
        hr_delta=(5, 3), hrv_delta=(-20, 4), spo2_delta=(0, 0.3),
        temp_delta=(0, 0.1), dyn_acc=(0.01, 0.005),
        weight=1.5, duration_frames=5,
    ),
    ScenarioTemplate(
        "stress_panic_onset", "stress", "stress",
        "Panic attack onset — sudden HR spike at rest, HRV drops",
        hr_delta=(35, 8), hrv_delta=(-15, 5), spo2_delta=(-0.5, 0.3),
        temp_delta=(0.3, 0.15), dyn_acc=(0.03, 0.02),
        weight=1.5, duration_frames=3,
    ),

    # ═══════════════════════════════════════════════════════════════
    # DISTRESS — multi-signal (expect "distress" label)
    # ═══════════════════════════════════════════════════════════════
    ScenarioTemplate(
        "distress_hr_spo2", "distress", "distress",
        "HR elevated + SpO2 dropping — two primary signals",
        hr_delta=(45, 6), hrv_delta=(-15, 4), spo2_delta=(-6.0, 1.0),
        temp_delta=(0.2, 0.15), dyn_acc=(0.02, 0.01),
        clinical_flags_gen="spo2_concern",
        weight=2.0, duration_frames=5,
    ),
    ScenarioTemplate(
        "distress_hr_hrv", "distress", "distress",
        "HR elevated + HRV suppressed — autonomic distress",
        hr_delta=(40, 6), hrv_delta=(-25, 5), spo2_delta=(-0.5, 0.3),
        temp_delta=(0.2, 0.1), dyn_acc=(0.02, 0.01),
        weight=1.5, duration_frames=4,
    ),
    ScenarioTemplate(
        "distress_spo2_hrv", "distress", "distress",
        "SpO2 dropping + HRV suppressed — respiratory distress pattern",
        hr_delta=(10, 4), hrv_delta=(-22, 5), spo2_delta=(-5.5, 1.0),
        temp_delta=(0, 0.1), dyn_acc=(0.01, 0.005),
        clinical_flags_gen="spo2_concern",
        weight=1.5, duration_frames=4,
    ),
    ScenarioTemplate(
        "distress_triple_signal", "distress", "distress",
        "HR + SpO2 + HRV all deviated — severe physiological distress",
        hr_delta=(55, 8), hrv_delta=(-28, 5), spo2_delta=(-7.0, 1.2),
        temp_delta=(0.5, 0.2), dyn_acc=(0.02, 0.01),
        clinical_flags_gen="spo2_concern",
        weight=1.5, duration_frames=4,
    ),
    ScenarioTemplate(
        "distress_with_fever", "distress", "distress",
        "HR elevated + temperature elevated + HRV suppressed — possible infection",
        hr_delta=(35, 6), hrv_delta=(-18, 5), spo2_delta=(-1.5, 0.5),
        temp_delta=(1.5, 0.3), dyn_acc=(0.01, 0.005),
        clinical_flags_gen="temp_elevated",
        weight=1.0, duration_frames=4,
    ),

    # ═══════════════════════════════════════════════════════════════
    # EMERGENCY — bypasses persistence
    # ═══════════════════════════════════════════════════════════════
    ScenarioTemplate(
        "emergency_spo2_critical", "distress", "distress",
        "SpO2 < 90% — medical emergency, immediate escalation",
        hr_delta=(25, 8), hrv_delta=(-15, 5), spo2_delta=(-10.0, 1.5),
        temp_delta=(0, 0.15), dyn_acc=(0.01, 0.005),
        clinical_flags_gen="spo2_hypoxemia",
        weight=1.0, duration_frames=3,
    ),
    ScenarioTemplate(
        "emergency_hr_extreme", "distress", "distress",
        "HR > 150 at rest — tachycardia emergency",
        hr_delta=(80, 10), hrv_delta=(-25, 5), spo2_delta=(-2.0, 0.8),
        temp_delta=(0.3, 0.15), dyn_acc=(0.02, 0.01),
        weight=1.0, duration_frames=3,
    ),
    ScenarioTemplate(
        "emergency_combined", "distress", "distress",
        "HR > 150 + SpO2 < 90% — multiple emergency signals",
        hr_delta=(85, 10), hrv_delta=(-30, 5), spo2_delta=(-11.0, 1.5),
        temp_delta=(0.5, 0.2), dyn_acc=(0.02, 0.01),
        clinical_flags_gen="spo2_hypoxemia",
        weight=0.8, duration_frames=2,
    ),

    # ═══════════════════════════════════════════════════════════════
    # EDGE CASES — tricky for rule-based systems
    # ═══════════════════════════════════════════════════════════════
    ScenarioTemplate(
        "edge_finger_off", "edge", "rejected",
        "Ring removed / finger not detected — frame rejected",
        hr_delta=(0, 0), hrv_delta=(0, 0), spo2_delta=(0, 0),
        temp_delta=(0, 0), dyn_acc=(0.01, 0.005),
        finger_on=False, accepted=False,
        weight=1.5, duration_frames=3,
    ),
    ScenarioTemplate(
        "edge_charging", "edge", "rejected",
        "Device on charger — frame rejected",
        hr_delta=(0, 0), hrv_delta=(0, 0), spo2_delta=(0, 0),
        temp_delta=(0, 0), dyn_acc=(0.0, 0.0),
        finger_on=False, accepted=False,
        weight=0.5, duration_frames=2,
    ),
    ScenarioTemplate(
        "edge_low_sqi", "edge", "normal",
        "Sensor noise — accepted but SQI very low",
        hr_delta=(5, 10), hrv_delta=(0, 8), spo2_delta=(0, 1.0),
        temp_delta=(0, 0.3), dyn_acc=(0.03, 0.02),
        weight=1.0, duration_frames=3,
    ),
    ScenarioTemplate(
        "edge_transient_hr_spike", "edge", "normal",
        "One-frame HR spike (motion artifact) — should NOT trigger stress",
        hr_delta=(50, 5), hrv_delta=(-5, 3), spo2_delta=(0, 0.3),
        temp_delta=(0, 0.1), dyn_acc=(0.30, 0.10),
        weight=1.5, duration_frames=1,  # only 1 frame
    ),
    ScenarioTemplate(
        "edge_temp_only_elevated", "edge", "normal",
        "Temperature elevated alone — weak signal, should stay normal",
        hr_delta=(3, 3), hrv_delta=(0, 3), spo2_delta=(0, 0.3),
        temp_delta=(1.5, 0.3), dyn_acc=(0.01, 0.005),
        clinical_flags_gen="temp_elevated",
        weight=1.5, duration_frames=3,
    ),
    ScenarioTemplate(
        "edge_exercise_plus_spo2", "edge", "distress",
        "Exercising + SpO2 drops — should NOT be suppressed",
        hr_delta=(60, 8), hrv_delta=(-18, 5), spo2_delta=(-6.0, 1.0),
        temp_delta=(0.5, 0.2), dyn_acc=(0.60, 0.15),
        clinical_flags_gen="spo2_concern",
        weight=1.5, duration_frames=4,
    ),
    ScenarioTemplate(
        "edge_copd_baseline", "edge", "normal",
        "COPD patient with chronically low SpO2 — should not alarm at their baseline",
        hr_delta=(0, 3), hrv_delta=(0, 3), spo2_delta=(0, 0.5),
        temp_delta=(0, 0.1), dyn_acc=(0.01, 0.005),
        weight=1.0, duration_frames=5,
    ),
    ScenarioTemplate(
        "edge_anxiety_resting", "edge", "normal",
        "Anxiety patient at rest — higher HR is their normal",
        hr_delta=(0, 5), hrv_delta=(0, 4), spo2_delta=(0, 0.3),
        temp_delta=(0, 0.1), dyn_acc=(0.01, 0.005),
        weight=1.0, duration_frames=5,
    ),
    ScenarioTemplate(
        "edge_recovery_after_distress", "edge", "normal",
        "Returning to normal after a distress episode — recovery phase",
        hr_delta=(10, 5), hrv_delta=(-5, 4), spo2_delta=(-0.5, 0.5),
        temp_delta=(0.1, 0.1), dyn_acc=(0.02, 0.01),
        weight=1.0, duration_frames=4,
    ),
    ScenarioTemplate(
        "edge_cold_start", "edge", "normal",
        "Baseline not yet established — only absolute thresholds active",
        hr_delta=(0, 5), hrv_delta=(0, 5), spo2_delta=(0, 0.5),
        temp_delta=(0, 0.15), dyn_acc=(0.01, 0.005),
        weight=1.0, duration_frames=5,
    ),

    # ═══════════════════════════════════════════════════════════════
    # GRADUAL TRANSITIONS — multi-frame sequences
    # ═══════════════════════════════════════════════════════════════
    ScenarioTemplate(
        "transition_normal_to_stress", "stress", "stress",
        "Gradual stress onset — HR rising over multiple frames",
        hr_delta=(30, 8), hrv_delta=(-10, 5), spo2_delta=(-0.5, 0.3),
        temp_delta=(0.1, 0.1), dyn_acc=(0.02, 0.01),
        weight=1.5, duration_frames=8,
    ),
    ScenarioTemplate(
        "transition_stress_to_distress", "distress", "distress",
        "Escalating from stress to distress — SpO2 starts dropping too",
        hr_delta=(45, 8), hrv_delta=(-18, 5), spo2_delta=(-5.0, 1.5),
        temp_delta=(0.2, 0.1), dyn_acc=(0.02, 0.01),
        clinical_flags_gen="spo2_concern",
        weight=1.5, duration_frames=8,
    ),
    ScenarioTemplate(
        "transition_distress_to_normal", "normal", "normal",
        "Distress resolving — vitals returning to baseline over time",
        hr_delta=(8, 6), hrv_delta=(-3, 4), spo2_delta=(-0.3, 0.4),
        temp_delta=(0.1, 0.1), dyn_acc=(0.02, 0.01),
        weight=1.0, duration_frames=6,
    ),
]


# ===========================================================================
# 3. FRAME GENERATOR
# ===========================================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _compute_zscore(value: float, baseline_mean: float, baseline_std: float) -> float:
    """Compute z-score given baseline stats."""
    if baseline_std < 0.01:
        return 0.0
    return (value - baseline_mean) / baseline_std


def _generate_sqi(accepted: bool, noisy: bool = False) -> Dict:
    """Generate realistic SQI sub-scores."""
    if not accepted:
        return {
            "hr": 0.0, "hrv": 0.0, "temp": 0.5, "acc": 0.3,
            "spo2": 0.0, "ppg": 0.0, "overall": 0.0,
        }
    if noisy:
        base = random.uniform(0.35, 0.55)
    else:
        base = random.uniform(0.70, 0.95)
    return {
        "hr":      round(_clamp(base + random.gauss(0, 0.05), 0, 1), 3),
        "hrv":     round(_clamp(base - 0.15 + random.gauss(0, 0.05), 0, 1), 3),
        "temp":    round(_clamp(base + 0.05 + random.gauss(0, 0.03), 0, 1), 3),
        "acc":     round(_clamp(base + 0.05 + random.gauss(0, 0.03), 0, 1), 3),
        "spo2":    round(_clamp(base + random.gauss(0, 0.05), 0, 1), 3),
        "ppg":     round(_clamp(base - 0.05 + random.gauss(0, 0.05), 0, 1), 3),
        "overall": round(_clamp(base + random.gauss(0, 0.03), 0, 1), 3),
    }


def generate_frame(
    profile: UserProfile,
    scenario: ScenarioTemplate,
    frame_idx: int,
    timestamp: int,
    sequence: int,
    baseline_ready: bool = True,
) -> Dict:
    """
    Generate one synthetic pipeline_output frame.

    For multi-frame scenarios, applies gradual ramp:
        - transition scenarios ramp linearly from 0% to 100% of delta
        - steady scenarios apply full delta with per-frame noise
    """
    # ── Ramping for transition scenarios ──────────────────────────
    n = max(1, scenario.duration_frames)
    if scenario.name.startswith("transition_"):
        # Ramp: 0% at frame 0 → 100% at last frame
        ramp = frame_idx / max(1, n - 1)
    else:
        ramp = 1.0

    # ── Apply delta with noise ────────────────────────────────────
    hr_raw = profile.resting_hr + ramp * scenario.hr_delta[0] + random.gauss(0, scenario.hr_delta[1])
    hr = round(_clamp(hr_raw, 30, 220), 1)

    hrv_raw = profile.resting_hrv + ramp * scenario.hrv_delta[0] + random.gauss(0, scenario.hrv_delta[1])
    hrv_rmssd = round(_clamp(hrv_raw, 2, 200), 1)

    spo2_raw = profile.resting_spo2 + ramp * scenario.spo2_delta[0] + random.gauss(0, scenario.spo2_delta[1])
    spo2 = round(_clamp(spo2_raw, 70, 100), 1)

    temp_raw = profile.resting_temp + ramp * scenario.temp_delta[0] + random.gauss(0, scenario.temp_delta[1])
    temp = round(_clamp(temp_raw, 33, 42), 2)

    dyn_acc = round(_clamp(
        scenario.dyn_acc[0] + random.gauss(0, scenario.dyn_acc[1]),
        0, 3.0,
    ), 3)

    # ── Z-scores (against this user's baseline) ──────────────────
    z_hr      = round(_compute_zscore(hr, profile.resting_hr, profile.hr_std), 2)
    z_hrv     = round(_compute_zscore(hrv_rmssd, profile.resting_hrv, profile.hrv_std), 2)
    z_spo2    = round(_compute_zscore(spo2, profile.resting_spo2, profile.spo2_std), 2)
    z_temp    = round(_compute_zscore(temp, profile.resting_temp, profile.temp_std), 2)

    # ── Clinical flags ────────────────────────────────────────────
    clinical_flags = []
    if scenario.clinical_flags_gen == "spo2_concern" and spo2 <= 94.0:
        clinical_flags.append(f"spo2_clinical_concern ({spo2}%)")
    if scenario.clinical_flags_gen == "spo2_hypoxemia" and spo2 <= 90.0:
        clinical_flags.append(f"spo2_hypoxemia ({spo2}%)")
        if spo2 <= 94.0:
            clinical_flags.append(f"spo2_clinical_concern ({spo2}%)")
    if scenario.clinical_flags_gen == "temp_elevated" and temp >= 37.5:
        clinical_flags.append(f"elevated_skin_temp ({temp:.1f}C)")
    # Auto-detect from generated values even if not explicitly flagged
    if spo2 <= 90.0 and "spo2_hypoxemia" not in str(clinical_flags):
        clinical_flags.append(f"spo2_hypoxemia ({spo2}%)")
    if 90.0 < spo2 <= 94.0 and "spo2_clinical_concern" not in str(clinical_flags):
        clinical_flags.append(f"spo2_clinical_concern ({spo2}%)")
    if temp >= 37.5 and "elevated_skin_temp" not in str(clinical_flags):
        clinical_flags.append(f"elevated_skin_temp ({temp:.1f}C)")

    # ── SQI ───────────────────────────────────────────────────────
    is_noisy = scenario.name == "edge_low_sqi"
    sqi = _generate_sqi(scenario.accepted, noisy=is_noisy)

    # ── Reject reasons ────────────────────────────────────────────
    reject_reasons = []
    if not scenario.accepted:
        if not scenario.finger_on:
            reject_reasons.append("finger_off")
        else:
            reject_reasons.append("low_signal_quality")

    # ── Acceleration components ───────────────────────────────────
    acc_z = round(0.98 + random.gauss(0, 0.02), 3)
    acc_x = round(dyn_acc * random.uniform(0.3, 0.7), 3)
    acc_y = round(dyn_acc * random.uniform(0.2, 0.5), 3)
    acc_mag = round(math.sqrt(acc_x**2 + acc_y**2 + acc_z**2), 3)

    # ── Build pipeline_output dict ────────────────────────────────
    sample = {
        "timestamp": timestamp,
        "sequence": sequence,
        "hr": hr,
        "hrv_rmssd": hrv_rmssd,
        "temp": temp,
        "temp_raw": round(temp + random.gauss(0, 0.05), 2),
        "spo2": spo2,
        "acc_mag": acc_mag,
        "dyn_acc_mag": dyn_acc,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "finger_on": scenario.finger_on,
        "charging": not scenario.finger_on and scenario.name == "edge_charging",
        "battery_mv": random.randint(3500, 4100),
        "die_temp": round(34 + random.gauss(0, 0.5), 1),
        "adc_raw": random.randint(80000, 150000),
        "thermal_bias": round(random.gauss(0, 0.05), 3),
        "sqi": sqi,
        "accepted": scenario.accepted,
        "reject_reasons": reject_reasons,
        "clinical_flags": clinical_flags,
    }

    # ── Window statistics (simulated 30-sec rolling) ──────────────
    window_n = 25 if baseline_ready else random.randint(2, 8)
    window = {
        "window_n": window_n,
        "hr_mean": round(profile.resting_hr + ramp * scenario.hr_delta[0] * 0.8, 1),
        "hr_var": round(profile.hr_std ** 2, 2),
        "hr_min": round(hr - random.uniform(2, 6), 1),
        "hr_max": round(hr + random.uniform(2, 6), 1),
        "hrv_rmssd_mean": round(profile.resting_hrv + ramp * scenario.hrv_delta[0] * 0.8, 1),
        "hrv_rmssd_var": round(profile.hrv_std ** 2, 2),
        "hrv_rmssd_min": round(hrv_rmssd - random.uniform(2, 5), 1),
        "hrv_rmssd_max": round(hrv_rmssd + random.uniform(2, 5), 1),
        "temp_mean": round(profile.resting_temp + ramp * scenario.temp_delta[0] * 0.8, 2),
        "temp_var": round(profile.temp_std ** 2, 4),
        "temp_min": round(temp - random.uniform(0.05, 0.15), 2),
        "temp_max": round(temp + random.uniform(0.05, 0.15), 2),
        "spo2_mean": round(profile.resting_spo2 + ramp * scenario.spo2_delta[0] * 0.8, 1),
        "spo2_var": round(profile.spo2_std ** 2, 2),
        "spo2_min": round(spo2 - random.uniform(0.3, 0.8), 1),
        "spo2_max": round(spo2 + random.uniform(0.3, 0.8), 1),
        "acc_mag_mean": round(acc_mag, 3),
        "acc_mag_var": round(random.uniform(0.001, 0.01), 4),
        "acc_mag_min": round(acc_mag - random.uniform(0.01, 0.03), 3),
        "acc_mag_max": round(acc_mag + random.uniform(0.01, 0.03), 3),
    }

    pipeline_output = {
        "sample": sample,
        "zscores": {
            "hr": z_hr if baseline_ready else 0.0,
            "hrv_rmssd": z_hrv if baseline_ready else 0.0,
            "spo2": z_spo2 if baseline_ready else 0.0,
            "temp": z_temp if baseline_ready else 0.0,
            "acc_mag": 0.0,
        },
        "baseline_ready": baseline_ready,
        "window": window,
    }

    return pipeline_output


# ===========================================================================
# 4. DATASET GENERATION
# ===========================================================================

def generate_dataset(
    n_target: int = 10000,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate the full synthetic dataset.

    Returns:
        rows:      flat list of labelled frames (for CSV)
        scenarios: list of multi-frame scenario sequences (for scenario CSV)
    """
    random.seed(seed)

    rows: List[Dict] = []
    scenario_records: List[Dict] = []
    ts_base = int(time.time()) - 100000
    sequence = 0
    scenario_id = 0

    # ── Calculate how many instances of each scenario ─────────────
    total_weight = sum(s.weight * s.duration_frames for s in SCENARIOS)
    instances_per_scenario: Dict[str, int] = {}
    for s in SCENARIOS:
        target_frames = int(n_target * (s.weight * s.duration_frames) / total_weight)
        n_instances = max(1, target_frames // max(1, s.duration_frames))
        instances_per_scenario[s.name] = n_instances

    # ── Generate ──────────────────────────────────────────────────
    for scenario in SCENARIOS:
        n_instances = instances_per_scenario[scenario.name]

        for inst in range(n_instances):
            # Pick a random profile for this instance
            profile = random.choice(PROFILES)

            # Cold start scenarios only use non-baseline-ready
            if scenario.name == "edge_cold_start":
                baseline_ready = False
            else:
                baseline_ready = True

            scenario_id += 1
            scenario_frames = []

            for frame_i in range(scenario.duration_frames):
                sequence += 1
                ts = ts_base + sequence

                frame = generate_frame(
                    profile=profile,
                    scenario=scenario,
                    frame_idx=frame_i,
                    timestamp=ts,
                    sequence=sequence,
                    baseline_ready=baseline_ready,
                )

                # ── Flatten to CSV row ────────────────────────────
                s = frame["sample"]
                z = frame["zscores"]
                w = frame["window"]

                row = {
                    # Metadata
                    "frame_id": sequence,
                    "scenario_id": scenario_id,
                    "scenario_name": scenario.name,
                    "scenario_category": scenario.category,
                    "expected_state": scenario.expected_state,
                    "profile_id": profile.profile_id,
                    "profile_label": profile.label,
                    "age": profile.age,
                    "sex": profile.sex,
                    "fitness": profile.fitness,
                    "frame_in_scenario": frame_i,
                    "total_frames": scenario.duration_frames,

                    # Vitals
                    "hr": s["hr"],
                    "hrv_rmssd": s["hrv_rmssd"],
                    "spo2": s["spo2"],
                    "temp": s["temp"],
                    "temp_raw": s["temp_raw"],
                    "dyn_acc_mag": s["dyn_acc_mag"],
                    "acc_mag": s["acc_mag"],
                    "acc_x": s["acc_x"],
                    "acc_y": s["acc_y"],
                    "acc_z": s["acc_z"],

                    # Sensor state
                    "finger_on": s["finger_on"],
                    "charging": s["charging"],
                    "battery_mv": s["battery_mv"],
                    "accepted": s["accepted"],

                    # SQI
                    "sqi_hr": s["sqi"]["hr"],
                    "sqi_hrv": s["sqi"]["hrv"],
                    "sqi_temp": s["sqi"]["temp"],
                    "sqi_acc": s["sqi"]["acc"],
                    "sqi_spo2": s["sqi"]["spo2"],
                    "sqi_ppg": s["sqi"]["ppg"],
                    "sqi_overall": s["sqi"]["overall"],

                    # Z-scores
                    "z_hr": z["hr"],
                    "z_hrv_rmssd": z["hrv_rmssd"],
                    "z_spo2": z["spo2"],
                    "z_temp": z["temp"],

                    # Clinical flags
                    "clinical_flags": "|".join(s["clinical_flags"]) if s["clinical_flags"] else "",
                    "reject_reasons": "|".join(s["reject_reasons"]) if s["reject_reasons"] else "",

                    # Baseline
                    "baseline_ready": baseline_ready,

                    # Window stats
                    "window_n": w["window_n"],
                    "window_hr_mean": w["hr_mean"],
                    "window_hr_var": w["hr_var"],
                    "window_spo2_mean": w["spo2_mean"],
                    "window_spo2_var": w["spo2_var"],
                    "window_hrv_mean": w["hrv_rmssd_mean"],
                    "window_hrv_var": w["hrv_rmssd_var"],
                    "window_temp_mean": w["temp_mean"],
                    "window_temp_var": w["temp_var"],

                    # Profile baselines (for validation)
                    "baseline_hr": profile.resting_hr,
                    "baseline_hrv": profile.resting_hrv,
                    "baseline_spo2": profile.resting_spo2,
                    "baseline_temp": profile.resting_temp,
                }

                rows.append(row)
                scenario_frames.append(row)

            # ── Scenario-level record ─────────────────────────────
            scenario_records.append({
                "scenario_id": scenario_id,
                "scenario_name": scenario.name,
                "scenario_category": scenario.category,
                "expected_state": scenario.expected_state,
                "description": scenario.description,
                "profile_id": profile.profile_id,
                "profile_label": profile.label,
                "n_frames": scenario.duration_frames,
                "mean_hr": round(sum(f["hr"] for f in scenario_frames) / len(scenario_frames), 1),
                "mean_spo2": round(sum(f["spo2"] for f in scenario_frames) / len(scenario_frames), 1),
                "mean_temp": round(sum(f["temp"] for f in scenario_frames) / len(scenario_frames), 2),
                "mean_dyn_acc": round(sum(f["dyn_acc_mag"] for f in scenario_frames) / len(scenario_frames), 3),
                "mean_z_hr": round(sum(f["z_hr"] for f in scenario_frames) / len(scenario_frames), 2),
                "mean_z_spo2": round(sum(f["z_spo2"] for f in scenario_frames) / len(scenario_frames), 2),
                "has_clinical_flags": any(f["clinical_flags"] for f in scenario_frames),
            })

    return rows, scenario_records


# ===========================================================================
# 5. VALIDATION — run engine against dataset and measure accuracy
# ===========================================================================

def validate_with_engine(rows: List[Dict]) -> Dict:
    """
    Run the distress engine against every frame and compare with ground truth.

    Returns accuracy metrics, confusion matrix, and per-scenario breakdown.
    """
    # Import our engine
    from distress_engine import DistressEngine, PERSISTENCE_STRESS_S, PERSISTENCE_DISTRESS_S
    import distress_engine as de

    # Disable persistence for per-frame validation
    original_stress = de.PERSISTENCE_STRESS_S
    original_distress = de.PERSISTENCE_DISTRESS_S
    de.PERSISTENCE_STRESS_S = 0
    de.PERSISTENCE_DISTRESS_S = 0

    engine = DistressEngine()

    # Track results
    total = 0
    correct = 0
    confusion = {
        "normal":   {"normal": 0, "stress": 0, "distress": 0, "rejected": 0},
        "stress":   {"normal": 0, "stress": 0, "distress": 0, "rejected": 0},
        "distress": {"normal": 0, "stress": 0, "distress": 0, "rejected": 0},
        "rejected": {"normal": 0, "stress": 0, "distress": 0, "rejected": 0},
    }
    per_scenario: Dict[str, Dict] = {}
    mismatches: List[Dict] = []

    for row in rows:
        expected = row["expected_state"]

        # Reconstruct pipeline_output from the CSV row
        pipeline_output = _row_to_pipeline_output(row)

        # Handle rejected frames
        if expected == "rejected":
            # Engine should skip these gracefully
            result = engine.evaluate(pipeline_output)
            predicted = "rejected" if result["debug"].get("skipped") else result["state"]
        else:
            # Reset engine for each scenario change (so persistence doesn't carry over)
            result = engine.evaluate(pipeline_output)
            predicted = result["state"]

        # Score
        total += 1
        match = (predicted == expected)
        if match:
            correct += 1
        else:
            if len(mismatches) < 50:  # cap mismatch log
                mismatches.append({
                    "frame_id": row["frame_id"],
                    "scenario": row["scenario_name"],
                    "expected": expected,
                    "predicted": predicted,
                    "hr": row["hr"],
                    "spo2": row["spo2"],
                    "z_hr": row["z_hr"],
                    "z_spo2": row["z_spo2"],
                    "dyn_acc": row["dyn_acc_mag"],
                })

        # Confusion matrix
        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1

        # Per-scenario tracking
        sname = row["scenario_name"]
        if sname not in per_scenario:
            per_scenario[sname] = {"total": 0, "correct": 0, "expected": expected}
        per_scenario[sname]["total"] += 1
        if match:
            per_scenario[sname]["correct"] += 1

    # Restore persistence
    de.PERSISTENCE_STRESS_S = original_stress
    de.PERSISTENCE_DISTRESS_S = original_distress

    # Compute per-scenario accuracy
    for k, v in per_scenario.items():
        v["accuracy_pct"] = round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0.0

    # Overall accuracy
    accuracy = round(correct / total * 100, 2) if total > 0 else 0.0

    # Per-class precision / recall
    class_metrics = {}
    for cls in ["normal", "stress", "distress", "rejected"]:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in confusion if other != cls)
        fn = sum(confusion[cls][other] for other in confusion[cls] if other != cls)
        precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0.0
        recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0.0
        f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0.0
        class_metrics[cls] = {
            "precision_pct": precision,
            "recall_pct": recall,
            "f1_pct": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    return {
        "total_frames": total,
        "correct": correct,
        "accuracy_pct": accuracy,
        "confusion_matrix": confusion,
        "class_metrics": class_metrics,
        "per_scenario_accuracy": per_scenario,
        "sample_mismatches": mismatches,
    }


def _row_to_pipeline_output(row: Dict) -> Dict:
    """Convert a flat CSV row back into a pipeline_output dict."""
    return {
        "sample": {
            "timestamp": row["frame_id"],
            "sequence": row["frame_id"],
            "hr": row["hr"],
            "hrv_rmssd": row["hrv_rmssd"],
            "temp": row["temp"],
            "temp_raw": row["temp_raw"],
            "spo2": row["spo2"],
            "acc_mag": row["acc_mag"],
            "dyn_acc_mag": row["dyn_acc_mag"],
            "acc_x": row["acc_x"],
            "acc_y": row["acc_y"],
            "acc_z": row["acc_z"],
            "finger_on": row["finger_on"],
            "charging": row["charging"],
            "battery_mv": row["battery_mv"],
            "die_temp": 34.5,
            "adc_raw": 112000,
            "thermal_bias": 0.0,
            "sqi": {
                "hr": row["sqi_hr"],
                "hrv": row["sqi_hrv"],
                "temp": row["sqi_temp"],
                "acc": row["sqi_acc"],
                "spo2": row["sqi_spo2"],
                "ppg": row["sqi_ppg"],
                "overall": row["sqi_overall"],
            },
            "accepted": row["accepted"],
            "reject_reasons": row["reject_reasons"].split("|") if row["reject_reasons"] else [],
            "clinical_flags": row["clinical_flags"].split("|") if row["clinical_flags"] else [],
        },
        "zscores": {
            "hr": row["z_hr"],
            "hrv_rmssd": row["z_hrv_rmssd"],
            "spo2": row["z_spo2"],
            "temp": row["z_temp"],
            "acc_mag": 0.0,
        },
        "baseline_ready": row["baseline_ready"],
        "window": {
            "window_n": row["window_n"],
            "hr_mean": row["window_hr_mean"],
            "hr_var": row["window_hr_var"],
            "hrv_rmssd_mean": row["window_hrv_mean"],
            "hrv_rmssd_var": row["window_hrv_var"],
            "temp_mean": row["window_temp_mean"],
            "temp_var": row["window_temp_var"],
            "spo2_mean": row["window_spo2_mean"],
            "spo2_var": row["window_spo2_var"],
            "acc_mag_mean": row["acc_mag"],
            "acc_mag_var": 0.001,
            "acc_mag_min": row["acc_mag"] - 0.02,
            "acc_mag_max": row["acc_mag"] + 0.02,
            "hr_min": row["hr"] - 3,
            "hr_max": row["hr"] + 3,
            "hrv_rmssd_min": row["hrv_rmssd"] - 3,
            "hrv_rmssd_max": row["hrv_rmssd"] + 3,
            "temp_min": row["temp"] - 0.1,
            "temp_max": row["temp"] + 0.1,
            "spo2_min": row["spo2"] - 0.5,
            "spo2_max": row["spo2"] + 0.5,
        },
    }


# ===========================================================================
# 6. MAIN — generate, save, validate
# ===========================================================================

if __name__ == "__main__":
    print("=" * 80)
    print(" WUALT Synthetic Dataset Generator")
    print("=" * 80)
    print()

    # ── Generate dataset ──────────────────────────────────────────
    print("  [1/4] Generating synthetic frames...")
    rows, scenario_records = generate_dataset(n_target=10000)
    print(f"         Generated {len(rows):,} frames across {len(scenario_records):,} scenario instances")

    # ── Distribution breakdown ────────────────────────────────────
    from collections import Counter
    state_dist = Counter(r["expected_state"] for r in rows)
    cat_dist   = Counter(r["scenario_category"] for r in rows)
    prof_dist  = Counter(r["profile_id"] for r in rows)

    print(f"\n         State distribution:")
    for state in ["normal", "stress", "distress", "rejected"]:
        n = state_dist.get(state, 0)
        print(f"           {state:>10s}: {n:>5,} ({n/len(rows)*100:.1f}%)")

    print(f"\n         Category distribution:")
    for cat in sorted(cat_dist.keys()):
        n = cat_dist[cat]
        print(f"           {cat:>10s}: {n:>5,} ({n/len(rows)*100:.1f}%)")

    print(f"\n         Profile distribution:")
    for pid in sorted(prof_dist.keys()):
        n = prof_dist[pid]
        print(f"           {pid:>6s}: {n:>5,} frames")

    # ── Save CSV ──────────────────────────────────────────────────
    print(f"\n  [2/4] Saving dataset CSV...")
    csv_path = os.path.join(OUT_DIR, "synthetic_dataset.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"         → {csv_path}")
    print(f"           {len(rows):,} rows × {len(fieldnames)} columns")

    # ── Save scenario CSV ─────────────────────────────────────────
    print(f"\n  [3/4] Saving scenario summary CSV...")
    scen_path = os.path.join(OUT_DIR, "synthetic_scenarios.csv")
    scen_fields = list(scenario_records[0].keys())
    with open(scen_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scen_fields)
        writer.writeheader()
        writer.writerows(scenario_records)
    print(f"         → {scen_path}")
    print(f"           {len(scenario_records):,} scenario instances")

    # ── Validate against engine ───────────────────────────────────
    print(f"\n  [4/4] Running distress engine validation...")
    print(f"         (evaluating every frame with persistence disabled)")
    results = validate_with_engine(rows)

    print(f"\n  {'═' * 60}")
    print(f"  VALIDATION RESULTS")
    print(f"  {'═' * 60}")
    print(f"  Total frames:  {results['total_frames']:,}")
    print(f"  Correct:       {results['correct']:,}")
    print(f"  ACCURACY:      {results['accuracy_pct']}%")
    print()

    print(f"  ┌────────────┬───────────┬──────────┬──────────┐")
    print(f"  │ Class      │ Precision │ Recall   │ F1       │")
    print(f"  ├────────────┼───────────┼──────────┼──────────┤")
    for cls in ["normal", "stress", "distress", "rejected"]:
        m = results["class_metrics"][cls]
        print(f"  │ {cls:<10s} │ {m['precision_pct']:>7.1f}%  │ {m['recall_pct']:>6.1f}%  │ {m['f1_pct']:>6.1f}%  │")
    print(f"  └────────────┴───────────┴──────────┴──────────┘")

    print(f"\n  Confusion Matrix (rows=expected, cols=predicted):")
    cm = results["confusion_matrix"]
    labels = ["normal", "stress", "distress", "rejected"]
    print(f"  {'':>12s}", end="")
    for l in labels:
        print(f" {l:>10s}", end="")
    print()
    for row_label in labels:
        print(f"  {row_label:>12s}", end="")
        for col_label in labels:
            print(f" {cm[row_label][col_label]:>10d}", end="")
        print()

    print(f"\n  Per-scenario accuracy:")
    print(f"  {'─' * 65}")
    for sname in sorted(results["per_scenario_accuracy"].keys()):
        info = results["per_scenario_accuracy"][sname]
        bar = "█" * int(info["accuracy_pct"] / 5)
        print(f"    {sname:<35s} {info['accuracy_pct']:>5.1f}%  {bar}")

    if results["sample_mismatches"]:
        print(f"\n  Sample mismatches (first {len(results['sample_mismatches'])}):")
        print(f"  {'─' * 80}")
        for mm in results["sample_mismatches"][:15]:
            print(f"    {mm['scenario']:<30s}  exp={mm['expected']:<8s}  got={mm['predicted']:<8s}  "
                  f"hr={mm['hr']}  spo2={mm['spo2']}  z_hr={mm['z_hr']}  acc={mm['dyn_acc']}")

    # ── Save summary JSON ─────────────────────────────────────────
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "total_frames": len(rows),
        "total_scenarios": len(scenario_records),
        "n_profiles": len(PROFILES),
        "state_distribution": dict(state_dist),
        "category_distribution": dict(cat_dist),
        "profile_distribution": dict(prof_dist),
        "validation": {
            "accuracy_pct": results["accuracy_pct"],
            "class_metrics": results["class_metrics"],
            "confusion_matrix": results["confusion_matrix"],
        },
        "columns": fieldnames,
        "profiles": [
            {
                "id": p.profile_id,
                "label": p.label,
                "age": p.age,
                "sex": p.sex,
                "fitness": p.fitness,
                "resting_hr": p.resting_hr,
                "resting_hrv": p.resting_hrv,
                "resting_spo2": p.resting_spo2,
                "resting_temp": p.resting_temp,
                "notes": p.notes,
            }
            for p in PROFILES
        ],
        "scenarios": [
            {
                "name": s.name,
                "category": s.category,
                "expected_state": s.expected_state,
                "description": s.description,
            }
            for s in SCENARIOS
        ],
    }

    json_path = os.path.join(OUT_DIR, "synthetic_dataset_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary → {json_path}")

    print(f"\n{'=' * 80}")
    print(f" Done. Files ready for training and validation.")
    print(f"{'=' * 80}")
