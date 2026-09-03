import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

try:
    from scipy.signal import savgol_filter, butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️  scipy not found - using basic smoothing (install scipy for better results)")

def smooth_data(data, window=11):
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    padded = np.pad(data, (window//2, window//2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(data)]

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """Zero-lag 4th order Butterworth low-pass filter (Biomechanics standard)"""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def savgol_smooth_and_diff(data, window=11, polyorder=3, deriv=0, dt=1.0):
    if not HAS_SCIPY:
        if deriv == 0:
            return smooth_data(data, window)
        elif deriv == 1:
            smoothed = smooth_data(data, window)
            return np.gradient(smoothed, dt)
        elif deriv == 2:
            smoothed = smooth_data(data, window)
            vel = np.gradient(smoothed, dt)
            return np.gradient(vel, dt)
    if len(data) < window:
        window = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window < polyorder + 2:
            polyorder = max(1, window - 2)
    result = savgol_filter(data, window, polyorder, deriv=deriv, delta=dt)
    return result

SEGMENT_PARAMS = {
    'forearm': {'mass_pct': 0.016, 'length_pct': 0.146, 'com_pct': 0.430, 'rg_pct': 0.303},
    'upper_arm': {'mass_pct': 0.028, 'length_pct': 0.186, 'com_pct': 0.436, 'rg_pct': 0.322},
    'trunk': {'mass_pct': 0.497, 'length_pct': 0.288, 'com_pct': 0.500, 'rg_pct': 0.496},
    'thigh': {'mass_pct': 0.100, 'length_pct': 0.245, 'com_pct': 0.433, 'rg_pct': 0.323},
    'shank': {'mass_pct': 0.0465, 'length_pct': 0.246, 'com_pct': 0.433, 'rg_pct': 0.302},
}

# ============================================================
# 12-DIMENSION THRESHOLD TABLE
# Maps each of the 4 phases / 12 dimensions onto our
# computed physics metrics, with per-skill-level corridors
# that yield a 1-5 star rating (5 = elite / Excellent,
# 3-4 = Satisfactory, 1-2 = Off-Target).
# ============================================================
DIMENSION_THRESHOLDS = {
    # PHASE 1 — BALANCE & LOAD
    'negative_move': {
        # Pelvis backward shift before stride (m).
        # SOURCE: Coaching consensus (Lau, Epstein); no level-stratified biomechanics
        # literature found. Treat as relative benchmarks only.
        'youth':        [(-0.02,1),(-0.01,2),(0.01,3),(0.03,4),(0.05,5)],
        'high_school':  [(-0.02,1),(-0.01,2),(0.02,3),(0.04,4),(0.06,5)],
        'college':      [(-0.02,1),(0.00,2),(0.03,3),(0.05,4),(0.08,5)],
        'professional': [(-0.02,1),(0.00,2),(0.03,3),(0.06,4),(0.10,5)],
    },
    'pelvis_load': {
        # Pelvis KE during load (J).
        # SOURCE: No direct hitting literature with KE values by level found.
        # Relative benchmarks only — use for within-athlete comparison.
        'youth':        [(0,1),(5,2),(15,3),(30,4),(50,5)],
        'high_school':  [(0,1),(10,2),(25,3),(50,4),(80,5)],
        'college':      [(0,1),(15,2),(40,3),(80,4),(120,5)],
        'professional': [(0,1),(25,2),(60,3),(110,4),(160,5)],
    },
    'upper_torso_load': {
        # Torso KE during load (J).
        # SOURCE: No direct hitting literature. Relative benchmarks only.
        'youth':        [(0,1),(5,2),(12,3),(25,4),(40,5)],
        'high_school':  [(0,1),(8,2),(18,3),(40,4),(65,5)],
        'college':      [(0,1),(12,2),(28,3),(60,4),(95,5)],
        'professional': [(0,1),(18,2),(40,3),(85,4),(130,5)],
    },
    # PHASE 2 — STRIDE
    'stride_length': {
        # stride_ratio (stride / height).
        # SOURCE: Escamilla et al. 2009 (J Appl Biomech, n=20 NCAA Div I):
        #   Mean stride length = 0.72 ± 0.11 × height for collegiate hitters.
        # Welch et al. 1995 (JOSPT): stride described qualitatively, no ratio given.
        # Fortenbaugh et al. 2011 (Sports Biomech): stride 0.60-0.85 × height in MLB.
        # Professional anchor: Fortenbaugh 2011 mean ~0.75, elite ~0.85.
        # Youth/HS scaled down proportionally (no published data).
        'youth':        [(0.0,1),(0.25,2),(0.45,3),(0.60,4),(0.75,5)],
        'high_school':  [(0.0,1),(0.30,2),(0.50,3),(0.65,4),(0.80,5)],
        'college':      [(0.0,1),(0.35,2),(0.55,3),(0.72,4),(0.85,5)],
        'professional': [(0.0,1),(0.40,2),(0.60,3),(0.75,4),(0.88,5)],
    },
    'forward_move': {
        # stride_efficiency_pct. No direct hitting literature by level.
        # SOURCE: Relative benchmarks only.
        'youth':        [(0,1),(40,2),(65,3),(90,4),(115,5)],
        'high_school':  [(0,1),(45,2),(70,3),(95,4),(115,5)],
        'college':      [(0,1),(50,2),(75,3),(98,4),(115,5)],
        'professional': [(0,1),(50,2),(75,3),(100,4),(115,5)],
    },
    # PHASE 3 — POWER MOVE
    'max_hip_shoulder_separation': {
        # max_separation_deg (shoulder − hip axial rotation angle).
        # SOURCE: Fleisig et al. 2013 (Sports Biomech, n=40 professional MLB batters):
        #   Max trunk axial rotation = 46 ± 9° (full swing, includes follow-through).
        # Taguchi et al. 2023 (Fukushima J Med Sci, n=18 university Div I):
        #   Hip-shoulder separation at foot contact = 22-24° (hip leads shoulder).
        # NOTE: X-Factor at load (pre-peak pelvis omega) typically 20-35° in hitting.
        # Professional 5-star anchor = Fleisig mean + 1SD = ~55°.
        # Youth/HS/college scaled proportionally; no level-stratified data published.
        'youth':        [(0,1),(8,2),(18,3),(28,4),(38,5)],
        'high_school':  [(0,1),(12,2),(22,3),(34,4),(44,5)],
        'college':      [(0,1),(15,2),(26,3),(38,4),(50,5)],
        'professional': [(0,1),(18,2),(30,3),(42,4),(55,5)],
    },
    'pelvis_rotation_range': {
        # Total pelvis rotation from load to contact (degrees).
        # SOURCE: Welch et al. 1995 (JOSPT): peak hip omega = 714 deg/s (adult hitters).
        # Taguchi et al. 2023: hip angle at follow-through = ~89-91° from closed position.
        # Fortenbaugh et al. 2011: pelvis rotation load-to-contact ~55-80° in MLB.
        # Professional anchor: Fortenbaugh 2011 mean ~68°, elite ~80°.
        # Youth/HS scaled down; no level-stratified data published.
        'youth':        [(0,1),(18,2),(32,3),(46,4),(60,5)],
        'high_school':  [(0,1),(22,2),(38,3),(52,4),(66,5)],
        'college':      [(0,1),(28,2),(44,3),(58,4),(72,5)],
        'professional': [(0,1),(32,2),(48,3),(62,4),(80,5)],
    },
    'upper_torso_rotation_range': {
        # Total shoulder/torso rotation from load to contact (degrees).
        # SOURCE: Welch et al. 1995 (JOSPT): peak shoulder omega = 937 deg/s.
        # Taguchi et al. 2023: shoulder at follow-through ~152° from ~-52° at foot
        #   contact → total range ~200°, load-to-contact ~100-120° for elite.
        # Fleisig et al. 2013: max trunk axial rotation 46 ± 9° (relative to hip).
        # Absolute shoulder rotation load-to-contact: ~90-110° for professional.
        # No level-stratified data; youth/HS scaled proportionally.
        'youth':        [(0,1),(28,2),(46,3),(64,4),(82,5)],
        'high_school':  [(0,1),(32,2),(52,3),(72,4),(90,5)],
        'college':      [(0,1),(38,2),(58,3),(78,4),(98,5)],
        'professional': [(0,1),(44,2),(64,3),(84,4),(108,5)],
    },
    # PHASE 4 — CONTACT & FOLLOW-THROUGH
    'pelvis_direction_at_contact': {
        # Deviation from square (90°) at ball contact (degrees). Lower = more open = better.
        # SOURCE: Taguchi et al. 2023 (university Div I, n=18):
        #   Hip angle at foot contact = -28 to -33° (closed); opens to ~89-91° by
        #   follow-through. At ball contact (after foot plant), hips are partially open.
        # Fortenbaugh et al. 2011: pelvis ~60-70° open at contact in MLB hitters.
        # Professional 5-star = ≤10° from square (fully open). No level-stratified data.
        'youth':        [(90,1),(65,2),(48,3),(32,4),(18,5)],
        'high_school':  [(90,1),(60,2),(44,3),(28,4),(14,5)],
        'college':      [(90,1),(55,2),(38,3),(22,4),(12,5)],
        'professional': [(90,1),(48,2),(32,3),(18,4),(8,5)],
    },
    'upper_torso_direction_at_contact': {
        # Shoulder deviation from square at contact (degrees). Lower = more open = better.
        # SOURCE: Taguchi et al. 2023: shoulder at foot contact = -52 to -55° (closed),
        #   lags hip by ~22-24°. Shoulders more closed than pelvis at contact.
        # Fortenbaugh et al. 2011: torso ~40-55° open at contact in MLB.
        # Professional 5-star = ≤14° from square. No level-stratified data.
        'youth':        [(90,1),(68,2),(52,3),(36,4),(22,5)],
        'high_school':  [(90,1),(62,2),(46,3),(30,4),(18,5)],
        'college':      [(90,1),(56,2),(40,3),(26,4),(16,5)],
        'professional': [(90,1),(50,2),(36,3),(22,4),(12,5)],
    },
    'kinetic_chain_efficiency': {
        # Distal energy share = (arm_ke + forearm_ke + bat_ke) / total_ke × 100.
        # SOURCE: No direct hitting literature with KE transfer ratios by level.
        # Driveline Baseball (internal research, publicly cited): distal KE fraction
        #   is a top predictor of bat speed. Elite MLB ~45-65% distal transfer.
        # Thresholds calibrated from observed data; treat as relative benchmarks.
        'youth':        [(0,1),(10,2),(22,3),(35,4),(50,5)],
        'high_school':  [(0,1),(12,2),(25,3),(40,4),(55,5)],
        'college':      [(0,1),(15,2),(28,3),(44,4),(60,5)],
        'professional': [(0,1),(18,2),(32,3),(48,4),(65,5)],
    },
    'sequence_quality': {
        # Computed from proper_sequence bool + sequence_timing_ms.
        # SOURCE: Taguchi et al. 2023 (university Div I, healthy group):
        #   Shoulder − pelvis peak omega lag = 52 ± 30ms.
        # Welch et al. 1995: hip peaks before shoulder confirmed (714 → 937 deg/s).
        # No level-stratified sequence timing data published.
        'youth':        [],
        'high_school':  [],
        'college':      [],
        'professional': [],
    },
    'hand_speed': {
        # Peak Hand Speed (mph) — handle of bat, 6" from knob. Occurs before contact.
        # SOURCE: Blast Motion official benchmarks (blastmotion.com):
        #   Youth:              15-21 mph
        #   High School (JV):   17-21 mph
        #   High School (Var):  19-23 mph  ← used for 'high_school'
        #   College:            21-25 mph
        #   Pro:                23-29 mph
        'youth':        [(0,1),(10,2),(15,3),(18,4),(21,5)],
        'high_school':  [(0,1),(12,2),(17,3),(20,4),(23,5)],
        'college':      [(0,1),(14,2),(19,3),(22,4),(25,5)],
        'professional': [(0,1),(16,2),(21,3),(25,4),(29,5)],
    },
    'follow_through_quality': {
        # Pelvis continued rotation after peak omega (degrees).
        # SOURCE: No direct hitting literature with level-stratified follow-through values.
        # Biomechanical principle: abrupt pelvis deceleration post-contact indicates
        #   bracing/energy leakage (Driveline, coaching consensus).
        # Elite hitters show 30-60° continued pelvis rotation post-contact.
        'youth':        [(0,1),(10,2),(20,3),(35,4),(50,5)],
        'high_school':  [(0,1),(15,2),(25,3),(40,4),(55,5)],
        'college':      [(0,1),(18,2),(30,3),(45,4),(60,5)],
        'professional': [(0,1),(20,2),(35,3),(50,4),(65,5)],
    },
    'lead_leg_block': {
        # Lead-knee extension from front-foot plant to contact (degrees straightened).
        # A firm, extending ("blocking") front leg posts up and redirects linear
        # momentum into rotation — one of the strongest bat-speed / exit-velocity
        # correlates in the Driveline OpenBiomechanics Project (OBP) hitting dataset.
        # SOURCE: OpenBiomechanics Project (baseball hitting); Driveline Baseball,
        #   "Hitting Biomechanics" (2022) — lead-knee extension velocity & ROM
        #   correlate to bat speed. No fully level-stratified public table exists;
        #   youth/HS/college scaled proportionally from the pro anchor. Treat as an
        #   approximate benchmark (relative-to-cohort), not a hard percentile.
        'youth':        [(0,1),(5,2),(12,3),(20,4),(28,5)],
        'high_school':  [(0,1),(8,2),(16,3),(24,4),(32,5)],
        'college':      [(0,1),(10,2),(18,3),(28,4),(36,5)],
        'professional': [(0,1),(12,2),(22,3),(32,4),(42,5)],
    },
}

# Percentile anchors for each of the 5 star-boundary thresholds (per dimension).
# Used to translate a raw value into an approximate cohort percentile (0-100).
PERCENTILE_ANCHORS = [5.0, 25.0, 50.0, 75.0, 92.0]

# Dimension weights for Swing Score (must sum to 1.0)
DIMENSION_WEIGHTS = {
    'negative_move': 0.04,
    'pelvis_load': 0.05,
    'upper_torso_load': 0.04,
    'stride_length': 0.05,
    'forward_move': 0.05,
    'max_hip_shoulder_separation': 0.12,
    'pelvis_rotation_range': 0.08,
    'upper_torso_rotation_range': 0.06,
    'pelvis_direction_at_contact': 0.07,
    'upper_torso_direction_at_contact': 0.07,
    'kinetic_chain_efficiency': 0.10,
    'sequence_quality': 0.10,
    'hand_speed': 0.13,  # Most reliable discriminator — directly measured from TRC
    'follow_through_quality': 0.04,
    'lead_leg_block': 0.07,  # OBP: lead-knee extension is a top bat-speed correlate
}
# NOTE: weights need not sum to 1.0 — build_phase_report normalises by total weight.

DIMENSION_LABELS = {
    'negative_move': 'Negative Move',
    'pelvis_load': 'Pelvis Load',
    'upper_torso_load': 'Upper Torso Load',
    'stride_length': 'Stride Length',
    'forward_move': 'Forward Move',
    'max_hip_shoulder_separation': 'Max Hip-Shoulder Separation',
    'pelvis_rotation_range': 'Pelvis Total Rotation Range',
    'upper_torso_rotation_range': 'Upper Torso Total Rotation Range',
    'pelvis_direction_at_contact': 'Pelvis Direction at Contact',
    'upper_torso_direction_at_contact': 'Upper Torso Direction at Contact',
    'kinetic_chain_efficiency': 'Energy Transfer',
    'sequence_quality': 'Sequence Quality',
    'hand_speed': 'Hand / Bat Speed',
    'follow_through_quality': 'Follow-Through Quality',
    'lead_leg_block': 'Lead-Leg Block',
}

SWING_PHASES = {
    'balance_load': {
        'label': 'Balance & Load',
        'icon': '',
        'dimensions': ['pelvis_load', 'upper_torso_load'],
    },
    'stride': {
        'label': 'Stride',
        'icon': '',
        'dimensions': ['negative_move', 'stride_length', 'forward_move'],
    },
    'power_move': {
        'label': 'Power Move',
        'icon': '',
        'dimensions': ['max_hip_shoulder_separation', 'pelvis_rotation_range', 'upper_torso_rotation_range'],
    },
    'contact': {
        'label': 'Contact & Follow-Through',
        'icon': '',
        'dimensions': ['pelvis_direction_at_contact', 'upper_torso_direction_at_contact', 'lead_leg_block', 'kinetic_chain_efficiency', 'sequence_quality', 'hand_speed', 'follow_through_quality'],
    },
}

# Dimensions where a LOWER value is better (deviation-from-square metrics).
INVERT_DIMS = {'pelvis_direction_at_contact', 'upper_torso_direction_at_contact'}

# Prescription library — each weak dimension maps to a concrete cue + drill + rationale.
# Modeled on Driveline's "constraint → intent → drill" coaching structure.
DRILL_LIBRARY = {
    'negative_move': {
        'cue': 'Gather back before you go forward',
        'drill': 'Slow "load-and-hold" reps: coil into the rear hip, pause 1s, then stride. Builds a repeatable rearward load.',
        'why': 'Without a rearward gather there is no stretch to unload — you start the swing already leaking forward.',
    },
    'pelvis_load': {
        'cue': 'Coil into the back hip',
        'drill': 'Banded hip-hinge coils and rear-hip loaded med-ball scoop tosses to store elastic energy in the pelvis.',
        'why': 'A quiet pelvis at load means less rotational energy available to release through contact.',
    },
    'upper_torso_load': {
        'cue': 'Keep the barrel back as the hips start',
        'drill': 'Counter-rotation "show-the-number" drills against a light band to deepen trunk coil.',
        'why': 'Insufficient trunk coil shrinks the stretch across the core and caps how much the torso can whip.',
    },
    'stride_length': {
        'cue': 'Stride to a firm, athletic base',
        'drill': 'Line/tee stride-length drill: mark a target ~70-85% of height and repeat the stride to plant.',
        'why': 'Stride length sets how much forward momentum you can convert into ground force and rotation.',
    },
    'forward_move': {
        'cue': 'Move forward, then stop hard',
        'drill': 'Walk-through swings that decelerate into a braced front side at plant.',
        'why': 'Momentum that never stops never converts to rotation — the front side must catch and redirect it.',
    },
    'max_hip_shoulder_separation': {
        'cue': 'Let the hips open while the shoulders wait',
        'drill': 'Hip-lead separation drill with a dowel across the shoulders; fire the pelvis first, feel the X-Factor stretch.',
        'why': 'Hip-shoulder separation stores elastic energy in the trunk — the biggest single lever on rotational speed.',
    },
    'pelvis_rotation_range': {
        'cue': 'Clear the hips fully to contact',
        'drill': '90/90 hip mobility work + rotational cable chops to expand usable pelvis rotation range.',
        'why': 'A short pelvis arc limits the runway the torso and arms have to accelerate over.',
    },
    'upper_torso_rotation_range': {
        'cue': 'Finish the shoulder turn through the ball',
        'drill': 'Thoracic rotation drills (open-book, side-lying windmills) plus full-turn dry swings.',
        'why': 'Restricted trunk rotation truncates the whip and pulls the barrel off the ball early.',
    },
    'pelvis_direction_at_contact': {
        'cue': 'Get the hips open (square to the pitcher) at contact',
        'drill': 'Front-hip "clearance" reps against a wall; feel the belt buckle face the pitcher at contact.',
        'why': 'Blocked, closed hips at contact trap energy in the lower half instead of releasing it into the barrel.',
    },
    'upper_torso_direction_at_contact': {
        'cue': 'Let the chest arrive slightly behind the hips',
        'drill': 'Connection-ball tee work to keep the torso stacked and sequenced behind the pelvis into contact.',
        'why': 'Shoulders that fly open early spend separation too soon; too closed and the barrel drags.',
    },
    'lead_leg_block': {
        'cue': 'Post up on a firm, straightening front leg',
        'drill': 'Rear-foot-elevated split squats for front-leg strength + "step-back / walkaway" swings to feel the lead knee brace and extend into contact.',
        'why': 'Lead-knee extension into contact is one of the strongest bat-speed correlates in the OpenBiomechanics dataset — a soft front leg leaks energy that should whip into the barrel.',
    },
    'kinetic_chain_efficiency': {
        'cue': 'Slow the hips so the hands can fly',
        'drill': 'Overload/underload rotational throws that train proximal deceleration and distal acceleration (the "whip").',
        'why': 'Efficiency is about the hand-off: energy must pass hips → torso → hands, not stall in the big segments.',
    },
    'sequence_quality': {
        'cue': 'Fire from the ground up: hips, then torso, then hands',
        'drill': 'Step-behind and pause-load drills that exaggerate pelvis-first sequencing with a clean 30-60ms lag.',
        'why': 'Out-of-order or simultaneous firing collapses the kinetic chain — the #1 predictor of velocity in Driveline OBP work.',
    },
    'hand_speed': {
        'cue': 'Whip the barrel, don’t push it',
        'drill': 'Overload/underload bat speed training (heavy + light bats) plus max-intent dry swings.',
        'why': 'Hand speed is the primary output of the kinetic chain and the most reliable proxy for power.',
    },
    'follow_through_quality': {
        'cue': 'Let the finish be long and loose',
        'drill': 'Full-extension finish swings; exhale and decelerate smoothly rather than stopping the barrel abruptly.',
        'why': 'An abrupt stop after contact means energy was braked by the body instead of delivered to the ball.',
    },
}

# ---------------------------------------------------------------------------
# Empirical cohort-percentile model (optional).
# Built by build_cohort.py from the user's OWN library of swings, grouped by
# level. When present, percentiles are computed against real swings at the same
# level instead of being estimated from research benchmarks.
# ---------------------------------------------------------------------------
# Stride detection. These trials cover a whole at-bat (several seconds), so the
# stride is searched for only in the window just before contact; a "stride"
# below the minimum ratio is stance jitter or a capture that began after front
# foot plant, and is reported as not-measured rather than as a bad stride.
STRIDE_LOOKBACK_S = 1.2
STRIDE_QUIET_MPS = 0.10
STRIDE_MIN_RATIO = 0.15

COHORT_MIN_N = 5  # minimum swings in a level before its empirical data is blended in at all
# Empirical-Bayes shrinkage constant. The user's cohort gets weight n/(n+K) and the
# research benchmark gets the remainder, so research guidance anchors small cohorts and
# is never fully abandoned. K=25 => a 25-swing cohort is a 50/50 blend.
COHORT_SHRINKAGE = 25.0
_COHORT_MODEL = None
_COHORT_LOADED = False


def load_cohort_model(path: str = None):
    """Load the level-grouped cohort-percentile model, if present. Idempotent."""
    global _COHORT_MODEL, _COHORT_LOADED
    import json
    p = path or os.environ.get('COHORT_MODEL_PATH') or os.path.join(os.path.dirname(__file__), 'cohort_percentiles.json')
    try:
        with open(p) as f:
            _COHORT_MODEL = json.load(f)
    except Exception:
        _COHORT_MODEL = None
    _COHORT_LOADED = True
    return _COHORT_MODEL


def get_cohort_model():
    if not _COHORT_LOADED:
        load_cohort_model()
    return _COHORT_MODEL


SKILL_LEVEL_BENCHMARKS = {
    'youth': {
        # Power: Blast Motion database (blastconnect.com): Youth 900-2,500 W
        'power_range_W': (900, 2500),
        'hip_power_per_kg_elite': 8.0,
        'ke_per_kg_elite': 2.0,
        'chain_efficiency_elite': 25.0,
        'torso_pelvis_ratio_optimal': (0.8, 1.1),
        'x_factor_optimal': (12, 32),
        'sequence_timing_ms': (15, 45),
        # Hand speed: Blast Motion database: Youth 17-23 mph (Peak Hand Speed)
        'max_hand_speed_mph': (17, 23),
        # Time to Contact: Blast Motion database: Youth 0.17-0.23 s
        'time_to_contact_range_s': (0.17, 0.23),
        # Body Rotation Ratio: Blast Motion: optimal 40-50% for all levels
        'pelvis_torso_contribution_pct': (40, 50),
    },
    'high_school': {
        # Power: Blast Motion database: HS Varsity 2,300-4,300 W
        'power_range_W': (2300, 4300),
        'hip_power_per_kg_elite': 12.0,
        'ke_per_kg_elite': 3.5,
        'chain_efficiency_elite': 35.0,
        'torso_pelvis_ratio_optimal': (1.0, 1.2),
        'x_factor_optimal': (20, 42),
        'sequence_timing_ms': (20, 55),
        # Hand speed: Blast Motion database: HS Varsity 20-26 mph
        'max_hand_speed_mph': (20, 26),
        # Time to Contact: Blast Motion database: HS Varsity 0.14-0.18 s
        'time_to_contact_range_s': (0.14, 0.18),
        'pelvis_torso_contribution_pct': (40, 50),
    },
    'college': {
        # Power: Blast Motion database: College 2,750-4,750 W
        'power_range_W': (2750, 4750),
        'hip_power_per_kg_elite': 16.0,
        'ke_per_kg_elite': 5.0,
        'chain_efficiency_elite': 40.0,
        'torso_pelvis_ratio_optimal': (1.1, 1.3),
        'x_factor_optimal': (25, 50),
        'sequence_timing_ms': (22, 82),
        # Hand speed: Blast Motion database: College 21-27 mph
        'max_hand_speed_mph': (21, 27),
        # Time to Contact: Blast Motion database: College 0.14-0.18 s
        'time_to_contact_range_s': (0.14, 0.18),
        'pelvis_torso_contribution_pct': (40, 50),
    },
    'professional': {
        # Power: Blast Motion database: Professional MLB 3,650-5,650 W
        'power_range_W': (3650, 5650),
        'hip_power_per_kg_elite': 20.0,
        'ke_per_kg_elite': 7.0,
        'chain_efficiency_elite': 45.0,
        'torso_pelvis_ratio_optimal': (1.2, 1.5),
        'x_factor_optimal': (37, 55),
        'sequence_timing_ms': (30, 60),
        # Hand speed: Blast Motion database: Professional MLB 23-29 mph
        'max_hand_speed_mph': (23, 29),
        # Time to Contact: Blast Motion database: Professional 0.13-0.17 s
        'time_to_contact_range_s': (0.13, 0.17),
        'pelvis_torso_contribution_pct': (40, 50),
    },
}

try:
    from metric_evidence import build_metric_evidence, TIER_LABELS
    _HAS_EVIDENCE = True
except ImportError:
    _HAS_EVIDENCE = False

@dataclass
class RefinedSwingMetrics:
    peak_hip_torque_Nm: float
    peak_shoulder_torque_Nm: float
    peak_hip_power_W: float
    peak_shoulder_power_W: float
    hip_inertia_kg_m2: float
    shoulder_inertia_kg_m2: float
    inertia_ratio: float
    hip_power_per_kg: float
    shoulder_power_per_kg: float
    max_separation_deg: float
    sequence_timing_ms: float
    proper_sequence: bool
    stride_length_m: float
    stride_ratio: float
    stride_efficiency_pct: float
    plant_frame: int
    plant_method: str
    estimated_hand_speed_mph: float
    # Internal composite grade, not a literature metric. Formula documented in
    # METRICS.md; versioned so the number can be compared across releases.
    swing_composite_score_v1: int
    # Driveline-inspired Energy Transfer Metrics
    pelvis_ke_J: float = 0.0
    torso_ke_J: float = 0.0
    arm_ke_J: float = 0.0
    forearm_ke_J: float = 0.0
    total_energy_transfer_J: float = 0.0
    torso_to_arm_transfer_ratio: float = 0.0
    pelvis_to_torso_transfer_ratio: float = 0.0
    torso_to_pelvis_rot_ratio: float = 0.0
    # Share of chain energy reaching the distal segments. "Efficiency" has no
    # standard definition in the literature — this is our own ratio, hence _proxy_.
    energy_transfer_proxy_pct: float = 0.0
    # Full 6-DOF Pelvis
    pelvis_tilt_range_deg: float = 0.0
    pelvis_list_range_deg: float = 0.0
    pelvis_tz_range_m: float = 0.0
    # Lower-body kinematics (bilateral peak values)
    peak_hip_flexion_r_deg: float = 0.0
    peak_hip_flexion_l_deg: float = 0.0
    peak_knee_flexion_r_deg: float = 0.0
    peak_knee_flexion_l_deg: float = 0.0
    peak_ankle_dorsiflexion_r_deg: float = 0.0
    peak_ankle_dorsiflexion_l_deg: float = 0.0
    hip_flexion_asymmetry_deg: float = 0.0
    knee_flexion_asymmetry_deg: float = 0.0
    # Lower-body kinetics (Newton-Euler)
    peak_knee_torque_r_Nm: float = 0.0
    peak_knee_torque_l_Nm: float = 0.0
    peak_ankle_torque_r_Nm: float = 0.0
    peak_ankle_torque_l_Nm: float = 0.0
    peak_knee_power_r_W: float = 0.0
    peak_knee_power_l_W: float = 0.0
    # Linear inverse dynamics (pelvis segment F=ma).
    # PROXY, not a true joint reaction force: derived from segment kinematics with
    # no measured ground reaction force, so the ground-contact term is absent.
    # Use for relative/between-swing comparison, not absolute load. See METRICS.md.
    peak_pelvis_force_ap_N: float = 0.0   # anterior-posterior (tx)
    peak_pelvis_force_vert_N: float = 0.0  # vertical (ty)
    peak_pelvis_force_lat_N: float = 0.0   # lateral (tz)
    peak_pelvis_force_resultant_N: float = 0.0
    # Weight shift / lateral balance
    lateral_sway_range_m: float = 0.0
    lateral_sway_at_plant_m: float = 0.0
    weight_shift_timing_pct: float = 0.0   # % of swing when peak lateral shift occurs
    # Bilateral arm kinematics
    peak_arm_flex_l_deg: float = 0.0
    peak_elbow_flex_l_deg: float = 0.0
    arm_flex_asymmetry_deg: float = 0.0
    peak_prosup_r_deg: float = 0.0
    peak_prosup_l_deg: float = 0.0
    # Blast Motion-aligned metrics
    time_to_contact_s: float = 0.0
    rotational_acceleration_deg_s2: float = 0.0
    pelvis_torso_contribution_pct: float = 0.0
    max_hand_speed_mph: float = 0.0
    peak_pelvis_omega_3d_deg_s: float = 0.0
    # ── Pelvis / lower-half metrics (see METRICS.md for citations) ──────────
    # X-Factor Stretch: separation gained AFTER transition begins, over and above
    # the separation already held at the top. Cheetham et al. (2001, 2008) found
    # stretch — not static X-Factor — is the stronger clubhead/bat-speed correlate.
    x_factor_stretch_deg: float = 0.0
    # Second link of the kinetic chain. sequence_timing_ms covers pelvis→torso;
    # this covers torso→lead arm. Kwon et al. (2013).
    torso_arm_sequence_gap_ms: float = 0.0
    # Pelvis rotation opened from swing start to contact. Single-frame read, so it
    # carries no derivative noise — the most robust metric in this group on
    # markerless data. Welch et al. (1995); Fortenbaugh et al. (2011): 40–75° in pros.
    pelvis_rotation_at_contact_deg: float = 0.0
    # Rotation swept from the top of the backswing to contact. Invariant to how the
    # athlete was oriented to the cameras, so it compares across setups where the
    # absolute angle above does not.
    pelvis_rotation_excursion_deg: float = 0.0
    # How the contact frame was found: peak_hand_speed, hand_deceleration_impact,
    # or peak_pelvis_omega_fallback. Surfaced because every "at contact" number
    # inherits the accuracy of this event.
    contact_detection_method: str = "none"
    # Peak lead-hip internal-rotation torque — the drive behind pelvis rotation.
    # MacWilliams et al. (1998).
    peak_lead_hip_ir_torque_Nm: float = 0.0
    # Rate the pelvis sheds angular velocity after its peak. Putnam (1993): rapid
    # proximal deceleration is what transfers momentum distally. A pelvis that stays
    # fast is a pelvis that never handed its energy up the chain.
    pelvis_decel_rate_deg_s2: float = 0.0
    # Time from swing start to peak pelvis angular velocity.
    time_to_peak_pelvis_ms: float = 0.0
    # Same, measured from front-foot plant — the form reported in the literature.
    # Welch et al. (1995): elite hitters peak ~50–100 ms after plant.
    time_to_peak_pelvis_from_plant_ms: float = 0.0

def _metric_evidence_block(metrics: dict, rotation_ctx: dict) -> dict:
    """Per-metric evidence tier + per-capture reliability. Degrades to {} if the
    registry is unavailable, so a missing module cannot break an analysis."""
    if not _HAS_EVIDENCE:
        return {}
    try:
        return build_metric_evidence({'metrics': metrics, '_rotation_context': rotation_ctx})
    except Exception:
        return {}


def _build_data_quality(trc_metrics: dict, has_grf: bool = False) -> dict:
    """
    Returns a data quality and validation context block for the API response.
    """
    has_trc = trc_metrics.get('max_hand_speed_mps', 0.0) > 0

    grf_status = {
        'force_plates_used': False,
        'grf_source': (
            'whole_body_CoM_Newton (F=ma_com from TRC markers) — ~10-15% BW MAE'
            if has_grf else
            'none — no TRC markers available for CoM estimation'
        ),
        'impact': (
            'CoM-derived GRFs reduce pelvis residuals and improve lower-extremity torque accuracy. '
            'Vertical GRF validated within ~5% of Welch 1995 (123% BW reference).'
            if has_grf else
            'Lower-extremity joint torques (knee, ankle) are less accurate without GRFs. '
            'Upper-body torques (shoulder, elbow, lumbar) are less GRF-dependent and more reliable.'
        ),
    }

    return {
        "opencap_validation": {
            "source": "Uhlrich et al. 2023, Nature Communications",
            "finding": "OpenCap kinematics validated against force plates: GRF MAE = 6.2% BW",
            "implication": (
                "The .mot kinematics driving this analysis are from the same validated pipeline. "
                "Joint angle accuracy is the primary input quality metric."
            ),
        },
        "grf_status": grf_status,
        "metric_reliability": {
            "high": [
                "pelvis_rotation_range_deg",
                "max_separation_deg (X-Factor)",
                "sequence_timing_ms",
                "peak_pelvis_omega_rad_s",
                "peak_shoulder_torque_Nm (OpenSim ID)",
                "peak_elbow_torque_Nm (OpenSim ID)",
                "peak_lumbar_torque_Nm (OpenSim ID)",
            ],
            "moderate": [
                "peak_hip_torque_Nm (no GRF)",
                "peak_knee_torque_Nm (no GRF)",
                "energy_transfer_proxy_pct",
                "estimated_hand_speed_mph (angular velocity method)" if not has_trc
                else "max_hand_speed_mph (TRC wrist markers — valid, but monocular reconstruction has high geometric sensitivity: accuracy degrades when wrist trajectory has large depth component relative to camera axis. Cross-session comparisons unreliable; use within-session trends only.)",
            ],
            "low_or_data_dependent": [
                "stride_length_m (unreliable for long trials >2s)",
                "pelvis_residual_force_N (expected to be large without GRFs)",
                "negative_move_m (no published hitting normative data)",
                "pelvis_ke_J / torso_ke_J (no published hitting normative data)",
            ],
        },
        "trc_markers_used": has_trc,
        "bat_ke_method": (
            "TRC wrist marker velocity (½mv²) — directly measured"
            if has_trc else
            "Angular velocity × lever arm estimate (½mv² with v = ω × forearm_length)"
        ),
        "threshold_evidence": {
            "literature_supported": [
                "max_hip_shoulder_separation (Fleisig et al. 2013, Sports Biomech, n=40 pro batters)",
                "sequence_timing_ms (Taguchi et al. 2023, Fukushima J Med Sci, n=18 Div I batters)",
                "pelvis_rotation_range (Welch et al. 1995, JOSPT)",
                "upper_torso_rotation_range (Welch 1995; Taguchi 2023)",
            ],
            "coaching_consensus_only": [
                "negative_move", "stride_length", "forward_move",
                "pelvis_load", "upper_torso_load", "kinetic_chain_efficiency",
            ],
        },
    }


# Below this score a trial is treated as degraded: it still appears in the report
# and stays individually selectable, but it is kept out of multi-swing averages
# so one bad capture can't move an athlete's numbers.
CAPTURE_QUALITY_MIN = 55


def assess_capture_quality(rotation: dict, dims: dict, has_markers: bool,
                           dt: float = None) -> dict:
    """Score how much this capture can be trusted (0-100), with reasons.

    Some trials contain a check swing, a practice move, or a mistracked capture
    rather than a real swing — a 5-star pelvis rotation alongside 3 mph hands is
    contradictory, not talent. Scoring the *capture* separately from the swing
    lets degraded trials be held out of averages instead of silently dragging
    them, and gives the athlete a reason rather than a mysterious number.
    """
    score = 100
    reasons = []

    peak_dgs = float((rotation or {}).get('peak_pelvis_omega_3d_deg_s') or 0.0)
    if peak_dgs <= 0:
        score -= 50
        reasons.append("pelvis rotation could not be measured")
    elif peak_dgs < 150:
        # Disqualifying on its own: no competitive swing turns the pelvis this
        # slowly, so this must not depend on a second flag to be excluded.
        score -= 60
        reasons.append(f"very little pelvis rotation ({peak_dgs:.0f}°/s) — likely a check swing or practice move")
    elif peak_dgs < 250:
        score -= 20
        reasons.append(f"weak pelvis rotation ({peak_dgs:.0f}°/s) for a full swing")

    if (rotation or {}).get('sequence_indeterminate'):
        score -= 15
        reasons.append("sequence order is indeterminate at this capture rate")

    if not has_markers:
        score -= 15
        reasons.append("no marker data — several metrics can't be measured")

    unavailable = [d.get('label', k) for k, d in (dims or {}).items()
                   if d.get('available') is False]
    if unavailable:
        score -= min(20, 5 * len(unavailable))
        reasons.append(f"{len(unavailable)} metric(s) not measurable: {', '.join(unavailable)}")

    # A swing takes roughly 100-400 ms; a window far outside that means the
    # segmentation latched onto the wrong movement.
    swing_ms = None
    if rotation is not None and dt:
        pom = rotation.get('pelvis_omega')
        start = rotation.get('swing_start_frame')
        if pom is not None and start is not None and len(pom):
            swing_ms = float((int(np.argmax(np.abs(pom))) - int(start)) * dt * 1000.0)
            if swing_ms < 60 or swing_ms > 600:
                score -= 20
                reasons.append(f"implausible swing window ({swing_ms:.0f} ms) — segmentation may have misfired")

    score = int(max(0, min(100, score)))
    return {
        'score': score,
        'usable': score >= CAPTURE_QUALITY_MIN,
        'reasons': reasons,
        'peak_pelvis_omega_deg_s': round(peak_dgs, 1),
        'swing_window_ms': round(swing_ms, 1) if swing_ms is not None else None,
        'has_markers': bool(has_markers),
    }


class RefinedHittingOptimizer:
    def __init__(self, body_mass_kg: float, body_height_m: float, skill_level: str = 'high_school',
                 bat_mass_kg: float = 0.88, bat_length_m: float = 0.864,
                 instrument: str = None, instrument_note: str = None,
                 handedness: str = None):
        self.body_mass_kg = float(body_mass_kg)
        self.body_height_m = float(body_height_m)
        self.skill_level = skill_level if skill_level in SKILL_LEVEL_BENCHMARKS else 'high_school'
        self.bat_mass_kg = float(bat_mass_kg)
        self.bat_length_m = float(bat_length_m)
        self.instrument = instrument          # e.g. 'wood_stick', 'aluminum_bat', None
        self.instrument_note = instrument_note  # free-text caveat for this session
        # Which side the athlete bats from. The .mot carries no handedness, so
        # without this the lead leg has to be guessed from the pose — which fails
        # on check swings and short captures. 'right' | 'left' | None.
        hd = str(handedness).lower().strip() if handedness else None
        self.handedness = hd if hd in ('right', 'left') else None
        self.g = 9.81
        self.calculate_segment_properties()
        
    def calculate_segment_properties(self):
        self.segments = {}
        for segment_name, params in SEGMENT_PARAMS.items():
            self.segments[segment_name] = {
                'mass': self.body_mass_kg * params['mass_pct'],
                'length': self.body_height_m * params['length_pct'],
                'com_dist': self.body_height_m * params['length_pct'] * params['com_pct'],
                'I': self.body_mass_kg * params['mass_pct'] * 
                     (self.body_height_m * params['length_pct'] * params['rg_pct'])**2
            }
            
    def load_mot_file(self, filepath: str) -> pd.DataFrame:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header_end = 0
        event_times = {}
        for i, line in enumerate(lines):
            if 'endheader' in line.lower():
                header_end = i + 1
                break
            if '=' in line and any(k in line for k in ('fp_10_time', 'contact_time')):
                key, val = line.strip().split('=', 1)
                event_times[key.strip()] = float(val.strip())
        data = pd.read_csv(filepath, sep='\t', skiprows=header_end, skipinitialspace=True)
        data.columns = data.columns.str.strip()
        # Attach event times as metadata via attrs (pandas 1.0+)
        data.attrs.update(event_times)
        return data
        
    def load_trc_file(self, filepath: str) -> pd.DataFrame:
        """Parse native .trc file with OpenSim header stripping and flatten columns"""
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except Exception:
            return None
            
        start_row = 0
        header_line = ""
        for i in range(min(20, len(lines))):
            if 'Frame#' in lines[i] or 'Time' in lines[i] or 'time' in lines[i].lower() or 'frame#' in lines[i].lower():
                start_row = i
                header_line = lines[i]
                break
                
        if start_row == 0:
            return None
            
        main_cols = [c.strip() for c in header_line.split('\t') if c.strip() != '']
        data_start = start_row + 2
        
        try:
            df = pd.read_csv(filepath, sep='\t', skiprows=data_start, header=None)
            flat_cols = ["Frame", "Time"]
            marker_idx = 2
            for col_idx in range(2, len(df.columns)):
                if marker_idx < len(main_cols):
                    marker_name = main_cols[marker_idx]
                else:
                    marker_name = f"M{marker_idx}"
                    
                axis_idx = (col_idx - 2) % 3
                if axis_idx == 0: axis = 'X'
                elif axis_idx == 1: axis = 'Y'
                else: 
                    axis = 'Z'
                    marker_idx += 1 
                flat_cols.append(f"{marker_name}_{axis}")
                
            df.columns = flat_cols[:len(df.columns)]
            return df
        except Exception:
            return None
            
    def calculate_trc_metrics(self, trc_data: pd.DataFrame) -> Dict:
        """Extract spatial metrics (e.g. hand speed) from 3D marker coordinates"""
        if trc_data is None or len(trc_data) == 0:
            return {'max_hand_speed_mph': 0.0}
            
        dt = trc_data['Time'].diff().mean()
        if dt <= 0 or np.isnan(dt): dt = 1/60.0
        fs = 1.0 / dt
        
        max_hand_speed = 0.0
        peak_time = 0.0
        contact_time = 0.0
        # Find whichever wrist is moving faster (proxy for bat speed)
        wrist_markers = ['r_mwrist_study', 'L_mwrist_study', 'r_lwrist_study', 'L_lwrist_study',
                         'r_wrist_radius', 'l_wrist_radius', 'r_wrist_ulna', 'l_wrist_ulna',
                         'RWrist', 'LWrist', 'RWRA', 'RWRB', 'LWRA', 'LWRB', 'RFIN', 'LFIN']

        for wrist in wrist_markers:
            if f'{wrist}_X' in trc_data.columns:
                wx = trc_data[f'{wrist}_X'].values
                wy = trc_data[f'{wrist}_Y'].values
                wz = trc_data[f'{wrist}_Z'].values
                
                if HAS_SCIPY:
                    wx = butter_lowpass_filter(wx, 15.0, fs)
                    wy = butter_lowpass_filter(wy, 15.0, fs)
                    wz = butter_lowpass_filter(wz, 15.0, fs)
                    
                vx = np.gradient(wx, dt)
                vy = np.gradient(wy, dt)
                vz = np.gradient(wz, dt)
                speed = np.sqrt(vx**2 + vy**2 + vz**2)
                
                cur_max = np.max(speed)
                if cur_max > max_hand_speed:
                    max_hand_speed = cur_max
                    times = trc_data['Time'].values
                    pk = int(np.argmax(speed))
                    peak_time = float(times[pk])

                    # Contact is the sharpest DECELERATION after the speed peak, not
                    # the peak itself. The wrists top out while the barrel is still
                    # whipping past them, so wrist peak speed leads bat-ball contact
                    # by a few frames; the hands then get braked as the barrel meets
                    # the ball. This is the hand force-feedback signal, read from
                    # kinematics.
                    #
                    # Validated on Emilio_1_10 against video: peak speed 0.667s,
                    # deceleration 0.717s, visible contact ~0.730s. The speed peak is
                    # 4 frames early and costs 17 deg of pelvis rotation at the ~265
                    # deg/s the pelvis is turning there; the deceleration is inside
                    # one frame.
                    contact_time = peak_time
                    hi = min(len(speed), pk + int(0.15 * fs) + 1)
                    if hi - pk >= 3:
                        acc = np.gradient(speed, dt)
                        contact_time = float(times[int(np.argmin(acc[pk:hi])) + pk])

        metrics = {
            'max_hand_speed_mps': float(max_hand_speed),
            'max_hand_speed_mph': float(max_hand_speed) * 2.23694,
            'hand_speed_peak_time_s': peak_time,
            'hand_contact_time_s': contact_time,
        }
        return metrics
        
    def calculate_marker_separation(self, trc_data: pd.DataFrame) -> Optional[Dict]:
        """Thorax-vs-pelvis axial separation (X-Factor) computed from markers.

        OpenSim's lumbar_rotation coordinate is not usable for this on OpenCap
        output: it clips at exactly 90.0 deg (the joint's stop) on 3 of 4 Emilio
        trials, and where it does not clip it still roughly doubles the true value
        — 74.0 deg against a marker-derived 38.3 deg on Emilio_1_10, versus an
        anatomical ceiling near 50-70 deg. Torso markers were checked and are clean,
        so the inflation is in the model/IK, not the capture.

        Each shoulder/pelvis line is projected onto the plane normal to the TRUNK's
        own long axis before the angle is taken. Projecting onto the global
        horizontal instead lets forward trunk tilt leak into the axial measure,
        which is what produced a 132 deg reading on the same trial.

        Returns None when the required markers are absent.
        """
        if trc_data is None or len(trc_data) == 0:
            return None

        # OpenCap emits two different marker vocabularies: the monocular pipeline
        # uses sternum/r_shoulder/r_ASIS, while the standard multi-camera
        # PostAugmentation set uses Neck/RShoulder/r.ASIS_study. Resolve either.
        def pick(*candidates):
            for c in candidates:
                if f'{c}_X' in trc_data.columns:
                    return c
            return None

        n_lsh = pick('L_shoulder_study', 'l_shoulder', 'LShoulder', 'L_shoulder')
        n_rsh = pick('r_shoulder_study', 'r_shoulder', 'RShoulder', 'R_shoulder')
        n_las = pick('L.ASIS_study', 'l_ASIS', 'LASI', 'LHip')
        n_ras = pick('r.ASIS_study', 'r_ASIS', 'RASI', 'RHip')
        n_top = pick('C7_study', 'C7', 'Neck', 'sternum')
        # PSIS pair is optional but strongly preferred: with all four pelvis markers
        # the transverse plane is defined anatomically instead of being inferred from
        # the ASIS line plus a trunk vector.
        n_lps = pick('L.PSIS_study', 'l_PSIS', 'LPSI')
        n_rps = pick('r.PSIS_study', 'r_PSIS', 'RPSI')
        if not all([n_lsh, n_rsh, n_las, n_ras, n_top]):
            return None

        dt = trc_data['Time'].diff().mean()
        if dt <= 0 or np.isnan(dt):
            dt = 1 / 60.0
        fs = 1.0 / dt

        def M(n):
            cols = [trc_data[f'{n}_{a}'].values for a in 'XYZ']
            if HAS_SCIPY:
                cols = [butter_lowpass_filter(c, 15.0, fs) for c in cols]
            return np.column_stack(cols)

        def unit(v):
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return v / n

        asis_mid = (M(n_ras) + M(n_las)) / 2.0
        pel_ml = unit(M(n_las) - M(n_ras))          # pelvis medio-lateral axis

        # Pelvis superior axis. With the PSIS pair the transverse plane is defined
        # anatomically (ISB convention): the ASIS-PSIS plane sets it, independent of
        # how the trunk above happens to be leaning. Without PSIS we fall back to the
        # trunk vector, which is the weaker definition the earlier version used
        # throughout — it lets trunk lean contaminate the axial measure.
        if n_lps and n_rps:
            psis_mid = (M(n_rps) + M(n_lps)) / 2.0
            fwd = unit(asis_mid - psis_mid)          # pelvis anterior axis
            pel_sup = unit(np.cross(pel_ml, fwd))
            frame_basis = 'asis_psis'
        else:
            pel_sup = unit(M(n_top) - asis_mid)
            frame_basis = 'trunk_vector'

        # Re-orthogonalise the ML axis against the superior axis so the pair forms a
        # genuine frame rather than two nearly-parallel vectors.
        pel_ml = unit(pel_ml - np.sum(pel_ml * pel_sup, axis=1, keepdims=True) * pel_sup)

        # Thorax orientation is taken about the C7-to-shoulder-midpoint axis rather
        # than from the raw shoulder line alone. The shoulder line carries scapular
        # protraction/retraction on top of thoracic rotation — the lead shoulder
        # protracts while the trail retracts through a swing — which inflated the
        # earlier reading to 91.8 deg on the multi-camera trial, past the ~70 deg
        # anatomical ceiling. Referencing C7 removes the part of that motion that
        # translates the shoulders without rotating the thorax.
        sh_mid = (M(n_lsh) + M(n_rsh)) / 2.0
        c7 = M(n_top)
        thx_sup = unit(sh_mid - asis_mid)
        thx_ml = unit(M(n_lsh) - M(n_rsh))
        thx_ml = unit(thx_ml - np.sum(thx_ml * thx_sup, axis=1, keepdims=True) * thx_sup)

        # Axial separation: thorax ML axis expressed in the PELVIS transverse plane.
        # Projecting into the pelvis's own plane — not a global horizontal and not a
        # generic trunk-normal plane — is what makes this an axial measure rather
        # than a mixture of twist, lean and tilt.
        proj = unit(thx_ml - np.sum(thx_ml * pel_sup, axis=1, keepdims=True) * pel_sup)
        pel_fwd = unit(np.cross(pel_sup, pel_ml))
        sep = np.degrees(np.arctan2(np.sum(proj * pel_fwd, axis=1),
                                    np.sum(proj * pel_ml, axis=1)))
        # Express as deviation from the athlete's own neutral stance rather than as a
        # raw geometric angle, which carries an arbitrary offset from marker placement.
        sep = np.unwrap(np.radians(sep)) * 180.0 / np.pi
        sep = sep - np.median(sep[:max(3, int(0.25 * fs))])

        return {
            'separation_deg': sep,
            'time': trc_data['Time'].values,
            'max_separation_deg': float(np.max(np.abs(sep))),
            'frame_basis': frame_basis,
            'markers_used': {'shoulders': [n_lsh, n_rsh], 'asis': [n_las, n_ras],
                             'psis': [n_lps, n_rps], 'top': n_top},
        }

    def calculate_rotational_torques_refined(self, data: pd.DataFrame, wrist_speed_mps: float = 0.0,
                                             hand_speed_peak_time_s: float = 0.0,
                                             marker_separation: Optional[Dict] = None) -> Dict:
        dt = data['time'].diff().mean()
        fs = 1.0 / dt if dt > 0 else 60.0
        
        # OpenCap generates jumping angles across -180/180 boundaries. We must UNWRAP them!
        if 'pelvis_rotation' not in data.columns or 'lumbar_rotation' not in data.columns:
            return None
            
        pelvis_angle_raw = np.deg2rad(data['pelvis_rotation'].values)
        pelvis_angle_unwrapped = np.unwrap(pelvis_angle_raw)

        # BUG 4 FIX: lumbar_rotation hits OpenSim's ±90° joint limit and can jump 174° in
        # one frame. np.unwrap only handles ±180° wraps so it misses this.
        # Detect frames where the signal is clamped at the joint limit and interpolate over
        # them before filtering — this prevents the Butterworth filter from smoothing through
        # the discontinuity and creating spurious high-velocity artifacts.
        lumbar_raw_deg = data['lumbar_rotation'].values.copy()
        # Repair IK joint-limit artifacts in lumbar_rotation:
        # OpenSim's Rajagopal model clamps lumbar_rotation at ±90°. When the IK hits the
        # limit, the angle saturates for several frames then snaps back. The saturation
        # frames AND the release frame immediately after are both artifacts.
        # Strategy:
        #  1. Flag frames stuck at the joint limit (|angle| >= 89.5°).
        #  2. Also flag the frame *after* any clamped run (the release/snap frame).
        #  3. Linearly interpolate over all flagged frames.
        # Secondary check: any single-frame jump > 30° that isn't already flagged is also
        # a discontinuity artifact — extend the mask to cover it.
        limit = 89.5  # deg — just inside the ±90° OpenSim joint limit
        bad = np.abs(lumbar_raw_deg) >= limit
        # Dilate by 1 frame to the right (mask release frame after each saturated run)
        bad_dilated = bad.copy()
        bad_dilated[1:] |= bad[:-1]
        # Also mask frames with single-frame jumps > 30° that aren't yet masked
        jumps = np.abs(np.diff(lumbar_raw_deg))
        jump_frames = np.where(jumps > 30.0)[0] + 1  # frame where the snap appears
        for jf in jump_frames:
            bad_dilated[max(0, jf-1):min(len(bad_dilated), jf+2)] = True
        # Interpolate over bad frames using surrounding valid values
        if np.any(bad_dilated):
            good = ~bad_dilated
            if good.sum() >= 2:  # need at least 2 valid points to interpolate
                idx = np.arange(len(lumbar_raw_deg))
                lumbar_raw_deg = np.interp(idx, idx[good], lumbar_raw_deg[good])
        lumbar_angle_unwrapped = np.unwrap(np.deg2rad(lumbar_raw_deg))

        # shoulder_angle = absolute thorax orientation (pelvis + lumbar relative twist)
        shoulder_angle_unwrapped = pelvis_angle_unwrapped + lumbar_angle_unwrapped
        
        if HAS_SCIPY:
            # 1. Zero-lag Butterworth filter (Cutoff: 15Hz) to remove OpenCap high-frequency noise
            cutoff_hz = 15.0
            pelvis_angle = butter_lowpass_filter(pelvis_angle_unwrapped, cutoff_hz, fs)
            lumbar_angle = butter_lowpass_filter(lumbar_angle_unwrapped, cutoff_hz, fs)
            # Absolute thorax orientation = pelvis + lumbar (for shoulder kinetics)
            shoulder_angle = pelvis_angle + lumbar_angle

            # 2. Dynamic Savitzky-Golay window based on actual framerate (~100ms window)
            window_size = int(0.10 * fs)
            if window_size % 2 == 0:
                window_size += 1
            window_size = max(11, window_size)

            pelvis_omega = savgol_smooth_and_diff(pelvis_angle, window=window_size, polyorder=3, deriv=1, dt=dt)
            pelvis_alpha = savgol_smooth_and_diff(pelvis_angle, window=window_size, polyorder=3, deriv=2, dt=dt)

            lumbar_omega = savgol_smooth_and_diff(lumbar_angle, window=window_size, polyorder=3, deriv=1, dt=dt)

            # BUG 4 FIX: Use absolute thorax omega/alpha for shoulder kinetics.
            # shoulder_omega = d/dt(pelvis + lumbar) = pelvis_omega + lumbar_omega.
            # This is correct physics. The inflated values in the original code were caused
            # by the joint-limit artifact in lumbar_rotation (now fixed by interpolation above).
            shoulder_omega = savgol_smooth_and_diff(shoulder_angle, window=window_size, polyorder=3, deriv=1, dt=dt)
            shoulder_alpha = savgol_smooth_and_diff(shoulder_angle, window=window_size, polyorder=3, deriv=2, dt=dt)
        else:
            pelvis_angle  = pelvis_angle_unwrapped
            lumbar_angle  = lumbar_angle_unwrapped
            shoulder_angle = pelvis_angle + lumbar_angle

            pelvis_smooth  = smooth_data(pelvis_angle, window=11)
            pelvis_omega   = np.gradient(pelvis_smooth, dt)
            pelvis_alpha   = np.gradient(pelvis_omega, dt)

            lumbar_smooth  = smooth_data(lumbar_angle, window=11)
            lumbar_omega   = np.gradient(lumbar_smooth, dt)

            shoulder_smooth = smooth_data(shoulder_angle, window=11)
            shoulder_omega  = np.gradient(shoulder_smooth, dt)
            shoulder_alpha  = np.gradient(shoulder_omega, dt)
        
        trunk_I = self.segments['trunk']['I']
        hip_inertia = trunk_I
        hip_torque = hip_inertia * pelvis_alpha
        
        upper_arm_I = self.segments['upper_arm']['I']
        forearm_I = self.segments['forearm']['I']
        bat_mass = self.bat_mass_kg
        # Bat modelled as uniform rod: I = (1/12)*m*L^2 about CoM, then parallel axis to handle
        # Simplified: use (1/3)*m*L^2 (rod rotating about one end — grip end)
        bat_I = (1.0 / 3.0) * bat_mass * (self.bat_length_m ** 2)
        
        shoulder_inertia = trunk_I + 2 * (upper_arm_I + forearm_I) + bat_I
        shoulder_torque = shoulder_inertia * shoulder_alpha

        inertia_ratio = shoulder_inertia / hip_inertia
        hip_power     = hip_torque * pelvis_omega
        shoulder_power = shoulder_torque * shoulder_omega

        # Detect swing start: find the last quiet frame before the sustained ramp to peak.
        # 
        # Design: a baseball swing has a clear signature — pelvis omega is near-zero for
        # most of the trial, then ramps monotonically to a single dominant peak over 100-400ms.
        # The swing start is the last frame below an onset threshold before that ramp begins.
        #
        # Algorithm: walk backward from peak_pelvis_frame.
        #   - onset_threshold = max(20 deg/s, 12% of peak omega)
        #   - When we drop below threshold after having been above it, that crossing IS the
        #     swing start (the bottom of the ramp). Stop there.
        #   - Hard bound: never search more than 600ms before peak.
        #     600ms chosen to handle slow/deliberate practice swings (all real swings < 500ms).
        #
        # Edge cases handled:
        #   - Long trials (4-8s of setup): the backward scan terminates at the ramp bottom,
        #     not at a sign reversal 3 seconds earlier.
        #   - Multiple movements before the swing: the ramp-finding scan stops at the quiet
        #     period between the last preparatory movement and the swing ramp.
        #   - Drill files (upper_first, lower_first): if the preceding drill movement has
        #     already decayed to quiet before the actual swing ramp, the detector finds the
        #     correct quiet gap between them.
        peak_pelvis_frame_global = int(np.argmax(np.abs(pelvis_omega)))
        swing_start = 0
        _max_swing_frames = int(0.60 * fs)   # 600ms hard bound

        # Use Driveline event timestamp (fp_10_time) when available — eliminates detection variability
        if 'fp_10_time' in data.attrs:
            fp_time = data.attrs['fp_10_time']
            times = data['time'].values
            swing_start = int(np.argmin(np.abs(times - fp_time)))
        else:
            pk = peak_pelvis_frame_global
            peak_omega_dgs = abs(pelvis_omega[pk]) * 180.0 / np.pi
            onset_thr_dgs = max(20.0, peak_omega_dgs * 0.12)   # deg/s
            onset_thr_rad = onset_thr_dgs * np.pi / 180.0      # rad/s
            search_lo = max(0, pk - _max_swing_frames)

            # Walk backward: track whether we've crossed above the onset threshold.
            # The first time we drop back below it (going backward) is the swing start.
            above = False
            ss_found = False
            for i in range(pk - 1, search_lo - 1, -1):
                if abs(pelvis_omega[i]) >= onset_thr_rad:
                    above = True
                else:
                    if above:
                        # Just crossed below threshold going backward — this is the
                        # bottom of the ramp: the last quiet frame before the swing.
                        swing_start = i
                        ss_found = True
                        break
                    # Still in quiet zone before the ramp — keep tracking
                    swing_start = i

            # Fallback: if the whole 600ms window is above threshold (very short trial
            # that starts mid-swing), use the first frame of the search window.
            if not ss_found or swing_start == 0:
                swing_start = search_lo

        sw = slice(swing_start, None)
        peak_hip_torque      = float(np.max(np.abs(hip_torque[sw])))
        peak_shoulder_torque = float(np.max(np.abs(shoulder_torque[sw])))
        peak_hip_power       = float(np.max(np.abs(hip_power[sw])))
        peak_shoulder_power  = float(np.max(np.abs(shoulder_power[sw])))

        hip_power_per_kg      = peak_hip_power / self.body_mass_kg
        shoulder_power_per_kg = peak_shoulder_power / self.body_mass_kg

        # X-Factor = max hip-shoulder separation BEFORE peak pelvis omega.
        # Use lumbar_angle (relative trunk twist) as the separation signal, measured
        # only in the window from swing_start to peak_pelvis_frame.
        separation_full = lumbar_angle * 180.0 / np.pi
        separation_source = 'model_lumbar_rotation'
        _sep_raw_peak = 0.0
        # Markers win when available: the model coordinate clips at its joint stop
        # and inflates axial separation roughly 2x (see calculate_marker_separation).
        if marker_separation is not None:
            _ms = np.interp(data['time'].values,
                            marker_separation['time'],
                            marker_separation['separation_deg'])
            separation_full = _ms
            separation_source = 'trc_markers'
            # Peak over the WHOLE signal, kept alongside the windowed value so the
            # plausibility check sees the raw excursion. A windowed number can sit
            # under the anatomical ceiling while the signal it came from is wildly
            # past it — which is exactly how the monocular trial passed.
            _sep_raw_peak = float(marker_separation.get('max_separation_deg', 0.0))
        pre_peak_sep = separation_full[swing_start:peak_pelvis_frame_global + 1]
        max_separation = float(np.max(np.abs(pre_peak_sep))) if len(pre_peak_sep) > 0 else float(np.max(np.abs(separation_full[swing_start:])))
        
        # Incorporating the Arms and Elbow details for Kinematics.
        # The LEAD arm is the front arm: left for a right-handed hitter, right for
        # a lefty. This was previously hardcoded to the right arm, so every
        # right-handed hitter had their TRAIL arm reported as "Lead Arm" — which
        # peaks at a different time and made the kinematic sequence look reversed.
        lead_arm_side = 'l' if (self.handedness or 'right') == 'right' else 'r'

        def _joint_omega(col):
            out = np.zeros_like(pelvis_omega)
            if col not in data.columns:
                return out
            unwrapped = np.unwrap(np.deg2rad(data[col].values))
            if HAS_SCIPY:
                filtered = butter_lowpass_filter(unwrapped, cutoff_hz, fs)
                return savgol_smooth_and_diff(filtered, window=window_size, polyorder=3, deriv=1, dt=dt)
            return np.gradient(smooth_data(unwrapped, window=11), dt)

        arm_col = f'arm_flex_{lead_arm_side}'
        elb_col = f'elbow_flex_{lead_arm_side}'
        # Fall back to the other side if the lead-side column is missing.
        if arm_col not in data.columns:
            arm_col = 'arm_flex_r' if 'arm_flex_r' in data.columns else arm_col
        if elb_col not in data.columns:
            elb_col = 'elbow_flex_r' if 'elbow_flex_r' in data.columns else elb_col
        arm_omega = _joint_omega(arm_col)
        elb_omega = _joint_omega(elb_col)

        # Slice arm/elbow to swing window
        arm_omega_sw  = arm_omega[swing_start:]
        elb_omega_sw  = elb_omega[swing_start:]
        p_omega_sw    = pelvis_omega[swing_start:]
        s_omega_sw    = shoulder_omega[swing_start:]
        lumbar_omega_sw = lumbar_omega[swing_start:]

        # Proximal-to-Distal Sequencing — within swing window only.
        # Use 3D pelvis omega magnitude for pelvis peak (matches Driveline pipeline).
        # Validated ground truth: ~50ms pelvis-before-torso (Driveline landmarks).
        def _om1d(col):
            if col not in data.columns: return np.zeros(len(pelvis_omega))
            arr = np.unwrap(np.deg2rad(data[col].values))
            if HAS_SCIPY:
                return savgol_smooth_and_diff(butter_lowpass_filter(arr, cutoff_hz, fs), window=window_size, polyorder=3, deriv=1, dt=dt)
            return np.gradient(smooth_data(arr, 11), dt)

        pelvis_om3d = np.sqrt(pelvis_omega**2 + _om1d('pelvis_tilt')**2 + _om1d('pelvis_list')**2)
        # Absolute thorax angular velocity = pelvis_rotation + lumbar_rotation (additive in planar approx).
        # Using relative lumbar alone is WRONG — it peaks early relative to pelvis because the
        # lumbar joint reaches its fastest twist while the pelvis is still accelerating.
        # Absolute thorax correctly peaks AFTER the pelvis, which is what sequence timing measures.
        thorax_abs_omega = pelvis_omega + _om1d('lumbar_rotation')
        torso_om3d = np.sqrt(thorax_abs_omega**2 + _om1d('lumbar_bending')**2)

        # Sub-frame peak interpolation via quadratic fit around argmax.
        # At 60Hz each frame is 17ms, so raw argmax has ±8ms quantization error that
        # produces apparent sequence reversals even when the hitter is sequencing correctly.
        # Fitting a parabola to the 3 samples around the peak gives ~1ms resolution.
        def _subframe_peak(arr, offset):
            """Return fractional frame index of the peak of arr[offset:], offset from start.
            Falls back to integer argmax if the parabola is ill-conditioned."""
            idx = int(np.argmax(arr[offset:]))
            i = idx + offset
            if i == 0 or i >= len(arr) - 1:
                return float(i)
            y0, y1, y2 = arr[i-1], arr[i], arr[i+1]
            denom = y0 - 2*y1 + y2
            if abs(denom) < 1e-12:
                return float(i)
            frac = 0.5 * (y0 - y2) / denom
            frac = max(-1.0, min(1.0, frac))   # clamp to ±1 frame
            return float(i) + frac

        # Pelvis peak first — used to constrain the thorax search.
        peak_hip_frac = _subframe_peak(pelvis_om3d, swing_start)
        peak_hip_frame = int(round(peak_hip_frac))

        # Thorax peak: search only within ±200ms of the pelvis peak.
        # This prevents large pre-swing setup movements (arms loading, etc.) from
        # stealing the global argmax and producing -200 to -700ms phantom lags.
        _search_half = int(0.20 * fs)   # 200ms each side
        thr_lo = max(swing_start, peak_hip_frame - _search_half)
        thr_hi = min(len(torso_om3d), peak_hip_frame + _search_half + 1)
        peak_shoulder_frac = _subframe_peak(torso_om3d[thr_lo:thr_hi], 0) + thr_lo

        peak_shoulder_frame = int(round(peak_shoulder_frac))
        # Bound the arm peak the same way as the thorax. Left unbounded it caught
        # arbitrary local maxima anywhere in the trial (observed 129-640ms on
        # consecutive swings from one athlete), which is noise, not sequencing.
        if np.sum(np.abs(arm_omega_sw)) > 0:
            _arm_abs = np.abs(arm_omega)
            arm_lo = max(swing_start, peak_hip_frame - _search_half)
            arm_hi = min(len(_arm_abs), peak_hip_frame + _search_half + 1)
            peak_arm_frac = (_subframe_peak(_arm_abs[arm_lo:arm_hi], 0) + arm_lo
                             if arm_hi > arm_lo else peak_shoulder_frac + 1.0)
        else:
            peak_arm_frac = peak_shoulder_frac + 1.0

        sequence_timing_ms = float((peak_shoulder_frac - peak_hip_frac) * dt * 1000.0)

        # Resolution uncertainty flags:
        # 1. Timing within ±1 frame → cannot distinguish order at 60Hz.
        # 2. Peak pelvis omega < 120 deg/s → too weak a movement to sequence-analyze
        #    (check swing, practice move, or noisy capture).
        _frame_ms = dt * 1000.0
        _peak_pelvis_dgs = float(np.max(pelvis_om3d[swing_start:])) * 180.0 / np.pi
        sequence_indeterminate = bool(
            abs(sequence_timing_ms) < _frame_ms or
            _peak_pelvis_dgs < 120.0
        )

        frame_tol = 1
        proper_sequence = bool(
            (peak_hip_frac - frame_tol) <= peak_shoulder_frac and
            peak_shoulder_frac <= (peak_arm_frac + frame_tol)
        )

        # =========================================================================
        # DRIVELINE-INSPIRED: Segmental Kinetic Energy Transfer Analysis
        # =========================================================================
        eps = 1e-6

        peak_pelvis_w = float(np.max(np.abs(p_omega_sw)))

        # Constrain lumbar (torso) omega search to ±200ms around the pelvis peak.
        # Searching the full post-swing window (swing_start to end) picks up post-impact
        # oscillations and marker noise that can be 2–3× larger than the actual swing peak.
        _ke_half = int(0.20 * fs)
        _ke_lo = max(swing_start, peak_hip_frame - _ke_half)
        _ke_hi = min(len(lumbar_omega), peak_hip_frame + _ke_half + 1)
        peak_shoulder_w = float(np.max(np.abs(lumbar_omega[_ke_lo:_ke_hi])))

        peak_arm_w_val  = float(np.max(np.abs(arm_omega_sw)))
        peak_elb_w_val  = float(np.max(np.abs(elb_omega_sw)))

        # Segmental Kinetic Energy: KE = 0.5 * I * omega^2
        # All segments use their own angular velocity in their own reference frame.
        # pelvis: absolute rotation; torso: relative to pelvis (lumbar); arms: absolute.
        bat_I = (1.0 / 3.0) * self.bat_mass_kg * (self.bat_length_m ** 2)

        pelvis_ke = 0.5 * hip_inertia * (peak_pelvis_w ** 2)
        # Cap peak_shoulder_w to 2.0× pelvis omega.
        # Empirical calibration on 46 clean Kike swings: mean ratio = 1.22×, p90 = 1.47×,
        # max = 1.90×. A 2.0× cap passes all legitimate values with margin while catching
        # any IK artifact that survived the lumbar_rotation interpolation step above.
        # (Previously 3× — too permissive; Jett's spike at 1692/740=2.3× was passing through.)
        peak_shoulder_w_capped = min(peak_shoulder_w, 2.0 * peak_pelvis_w)
        torso_ke  = 0.5 * trunk_I * (peak_shoulder_w_capped ** 2)
        arm_ke    = 0.5 * (upper_arm_I + forearm_I) * 2 * (peak_arm_w_val ** 2)
        # Forearm segment KE about the elbow. Named for the segment that carries the
        # energy, not the joint it rotates about — the elbow itself has no inertia.
        forearm_ke = 0.5 * forearm_I * 2 * (peak_elb_w_val ** 2)

        # Bat KE = translational (½mv²) + rotational (½Iω²).
        # Translational term uses measured wrist speed from TRC markers when available
        # (most reliable), otherwise falls back to arm angular velocity × lever arm.
        if wrist_speed_mps > 0 and self.bat_mass_kg > 0:
            bat_ke_trans = 0.5 * self.bat_mass_kg * (wrist_speed_mps ** 2)
        else:
            # Fallback: estimate wrist linear speed from arm omega × forearm+hand length
            lever = self.body_height_m * 0.204  # forearm + hand ≈ 20.4% of height
            bat_ke_trans = 0.5 * self.bat_mass_kg * (peak_arm_w_val * lever) ** 2
        bat_ke_rot = 0.5 * bat_I * (peak_arm_w_val ** 2)
        bat_ke = bat_ke_trans + bat_ke_rot

        total_energy_transfer = pelvis_ke + torso_ke + arm_ke + forearm_ke + bat_ke

        torso_to_arm_ratio       = torso_ke / (arm_ke + eps)
        pelvis_to_torso_ratio    = pelvis_ke / (torso_ke + eps)
        torso_to_pelvis_rot_ratio = peak_shoulder_w_capped / (peak_pelvis_w + eps)

        # Fraction of total chain energy that reaches the distal segments (arms + bat).
        # Deliberately NOT called "efficiency": the literature has no agreed formula for
        # kinetic-chain efficiency, and a true one would follow joint-power flow
        # (Robertson & Winter 1980), not a ratio of peak segment energies. This is our
        # own defined proxy — see METRICS.md.
        distal_ke = arm_ke + forearm_ke + bat_ke
        energy_transfer_proxy = (distal_ke / (total_energy_transfer + eps)) * 100.0
        
        # =========================================================================
        # BLAST MOTION-ALIGNED METRICS
        # =========================================================================
        # Time to Contact: swing_start to peak_pelvis_frame (contact proxy) in seconds
        time_to_contact_s = float((peak_pelvis_frame_global - swing_start) * dt)

        # Rotational Acceleration: peak angular acceleration of the shoulder/arm system
        # during the swing window — measures how quickly bat accelerates into swing plane
        # (Blast: higher = better power efficiency, less hand-dominated)
        arm_alpha = np.zeros_like(pelvis_omega)
        if 'arm_flex_r' in data.columns:
            if HAS_SCIPY:
                arm_alpha = savgol_smooth_and_diff(
                    butter_lowpass_filter(np.unwrap(np.deg2rad(data['arm_flex_r'].values)), cutoff_hz, fs),
                    window=window_size, polyorder=3, deriv=2, dt=dt)
            else:
                arm_alpha = np.gradient(arm_omega, dt)
        peak_rotational_accel = float(np.max(np.abs(arm_alpha[swing_start:])) * 180.0 / np.pi)  # deg/s²

        # Pelvis share of total rotational contribution (pelvis vs pelvis + arm) at peak.
        # Conceptually aligned with Blast Motion's "Body Rotation", but NOT that metric:
        # their formula is proprietary and unpublished, so this is our own approximation
        # and the two should not be expected to agree numerically.
        total_omega_at_peak = peak_pelvis_w + peak_arm_w_val + eps
        pelvis_torso_contribution = float((peak_pelvis_w / total_omega_at_peak) * 100.0)

        # ── 3D Angular Velocity Magnitude (matches Driveline's angular_velocity columns) ──
        def _omega_1d(col):
            if col not in data.columns:
                return np.zeros(len(pelvis_omega))
            arr = np.unwrap(np.deg2rad(data[col].values))
            if HAS_SCIPY:
                arr = butter_lowpass_filter(arr, cutoff_hz, fs)
                return savgol_smooth_and_diff(arr, window=window_size, polyorder=3, deriv=1, dt=dt)
            return np.gradient(smooth_data(arr, 11), dt)

        pelvis_omega_x = _omega_1d('pelvis_tilt')
        pelvis_omega_z = _omega_1d('pelvis_list')
        pelvis_omega_3d = np.sqrt(pelvis_omega**2 + pelvis_omega_x**2 + pelvis_omega_z**2)
        torso_omega_x = _omega_1d('lumbar_bending')
        torso_omega_3d = np.sqrt(lumbar_omega**2 + torso_omega_x**2)
        peak_pelvis_omega_3d = float(np.max(pelvis_omega_3d[swing_start:]))
        peak_torso_omega_3d  = float(np.max(torso_omega_3d[swing_start:]))

        # =========================================================================
        # PELVIS / LOWER-HALF METRICS  (citations in METRICS.md)
        # =========================================================================
        # ── Contact frame ────────────────────────────────────────────────────────
        # Anchored to the HANDS, not the pelvis. Peak pelvis omega is systematically
        # early: the pelvis is the first link in the chain and keeps rotating long
        # after it stops accelerating. On Emilio_1_16 the pelvis peaked at t=1.23s
        # while the pelvis angle kept opening until t=1.63s — a 400ms error that made
        # "pelvis rotation at contact" read 26° against a true 54°.
        #
        # The hands are the distal end of the chain, so their speed peaks essentially
        # at ball contact. Falls back to the pelvis only when no arm data exists.
        hands_abs_omega = np.abs(thorax_abs_omega + arm_omega + elb_omega)
        contact_method = 'peak_pelvis_omega_fallback'
        contact_frame = int(min(max(peak_pelvis_frame_global, swing_start),
                                len(pelvis_angle) - 1))

        # Measured contact, when marker data is available. Wrist markers give real
        # hand linear speed, whose peak is bat-ball contact to within a frame or two.
        # Everything below this is inference from joint angles; this is measurement,
        # so it wins outright when present.
        if hand_speed_peak_time_s and hand_speed_peak_time_s > 0:
            _times = data['time'].values
            contact_frame = int(np.argmin(np.abs(_times - hand_speed_peak_time_s)))
            contact_method = 'trc_hand_deceleration'

        elif np.any(hands_abs_omega[swing_start:] > 0):
            # Bound to ±300ms of the pelvis peak so a post-follow-through wobble or a
            # second movement later in the trial cannot masquerade as contact.
            _c_half = int(0.30 * fs)
            c_lo = max(swing_start, peak_pelvis_frame_global - _c_half)
            c_hi = min(len(hands_abs_omega), peak_pelvis_frame_global + _c_half + 1)
            if c_hi > c_lo:
                hands_pk = int(np.argmax(hands_abs_omega[c_lo:c_hi])) + c_lo
                contact_frame = hands_pk
                contact_method = 'peak_hand_speed'

                # Impact refinement. A struck ball puts an impulsive brake on the hands,
                # so the sharpest deceleration just after peak speed marks contact more
                # precisely than the speed peak alone.
                #
                # Only accepted when the spike is distinct (>1.5x the surrounding
                # deceleration), because at 60Hz sampling with a 15Hz filter a real
                # ~1ms impact transient is largely smoothed away. It therefore fires on
                # solid contact and correctly declines to on dry swings and tee work,
                # where there is no impulse to find.
                d_hi = min(len(hands_abs_omega), hands_pk + int(0.08 * fs) + 1)
                if d_hi - hands_pk >= 3:
                    decel = -np.gradient(hands_abs_omega[hands_pk:d_hi], dt)
                    if np.max(decel) > 0:
                        sharp = int(np.argmax(decel)) + hands_pk
                        med = np.median(np.abs(decel))
                        # 3x median, not 1.5x. At 1.5x this fired on a slow rehearsal
                        # swing (peak pelvis 227 deg/s) and placed contact 250ms before
                        # the pelvis finished rotating, which is not a plausible impact.
                        # A struck ball produces a far sharper brake than that.
                        if med > 0 and np.max(decel) > 3.0 * med and sharp > hands_pk:
                            contact_frame = sharp
                            contact_method = 'hand_deceleration_impact'

        contact_frame = int(min(max(contact_frame, swing_start), len(pelvis_angle) - 1))

        # ── Transition (top of the backswing) ────────────────────────────────────
        # The instant the pelvis reverses from loading to firing. Found by walking back
        # from contact to the last sign change in pelvis angular velocity.
        #
        # This is a different event from swing_start, which is a speed threshold and is
        # therefore already past the top by the time it triggers. X-Factor stretch has
        # to be measured from the true top: measured from swing_start it came out
        # exactly 0.0 on every real trial, because separation is at its maximum the
        # moment the threshold trips and only decays afterward.
        transition_frame = swing_start
        for i in range(contact_frame - 1, swing_start, -1):
            if pelvis_omega[i] * pelvis_omega[contact_frame] <= 0:
                transition_frame = i
                break

        # ── X-Factor Stretch (Cheetham et al. 2001, 2008) ────────────────────────
        # Separation gained after the transition, above what was already held at the
        # top. max_separation_deg alone cannot distinguish a hitter who coils to 45°
        # and holds it from one who coils to 45° and stretches to 60° as the pelvis
        # fires — the second is the power pattern, and only stretch sees it.
        # Measured from the transition, per Cheetham: separation at the top vs. its
        # maximum during the early downswing. The gain exists because the pelvis
        # accelerates open faster than the torso follows.
        _sep_win = np.abs(separation_full[transition_frame:contact_frame + 1])
        sep_at_top = float(abs(separation_full[transition_frame]))
        x_factor_stretch = float(max(0.0, (np.max(_sep_win) if len(_sep_win) else 0.0) - sep_at_top))

        # Saturation guard. OpenSim's lumbar_rotation coordinate clips at its joint
        # limit, and thorax-vs-pelvis rotation beyond ~70 deg is not anatomically
        # reachable — both Emilio trials pin at exactly 90.0 deg, which is the solver
        # against the stop rather than the athlete's real coil. Any separation number
        # taken from a clipped signal is meaningless, so report 0 and flag it rather
        # than publish a figure that looks like a measurement.
        # Detect actual clipping rather than large magnitude. A clipped coordinate
        # sits pinned at one value for several consecutive frames; a real rotation
        # passes through its peak. Testing magnitude alone wrongly condemned honest
        # marker data, whose follow-through separation is legitimately large.
        # Only the model coordinate can clip — markers have no joint stop.
        # Tested on the RAW coordinate, not the filtered one: the low-pass rounds the
        # clipped plateau into a smooth peak, so the filtered signal no longer shows
        # the repeated values that identify a clip. Emilio_1_10 sits at exactly 90.0
        # for 29 raw frames and 1_16 for 18, while 1_7 and 1_8 reach their peaks for
        # 2 and 1 frames — real rotations passing through a maximum.
        _sep_saturated = False
        if separation_source == 'model_lumbar_rotation' and 'lumbar_rotation' in data.columns:
            _raw = np.abs(data['lumbar_rotation'].values)
            if len(_raw):
                _pk = float(np.max(_raw))
                _at_pk = int(np.sum(np.abs(_raw - _pk) < 0.05))
                _sep_saturated = bool(_pk >= 70.0 and _at_pk >= 3)
        if _sep_saturated:
            x_factor_stretch = 0.0

        # ── Torso → lead arm sequence gap (Kwon et al. 2013) ─────────────────────
        # sequence_timing_ms already reports pelvis→torso; this is the next link.
        # Both use the same sub-frame peaks, so the two gaps sum to pelvis→arm.
        torso_arm_gap_ms = float((peak_arm_frac - peak_shoulder_frac) * dt * 1000.0)

        # ── Pelvis rotation at contact (Welch 1995; Fortenbaugh 2011) ────────────
        # The ABSOLUTE pelvis orientation at contact, which is what the literature's
        # 40-75° range refers to and what the OpenCap pelvis_rotation curve plots.
        # Previously this returned the change from swing start, a different quantity
        # that read ~26° where the true angle was ~54°.
        #
        # Caveat: absolute angle is expressed in the capture's ground frame, so it
        # shifts if the athlete is set up at a different orientation to the cameras.
        # Comparable within a session; compare across sessions with care.
        pelvis_rot_at_contact = float(abs(pelvis_angle[contact_frame]) * 180.0 / np.pi)

        # Total rotation swept from the top of the backswing through to contact.
        # Setup-orientation invariant, so this is the safer number for comparing
        # one athlete's swings against another's.
        pelvis_rot_excursion = float(
            abs(pelvis_angle[contact_frame] - pelvis_angle[transition_frame]) * 180.0 / np.pi
        )

        # ── Pelvis deceleration rate after peak (Putnam 1993) ────────────────────
        # Steepest negative slope of pelvis omega in the 200 ms after its peak. The
        # proximal segment braking is the mechanism that drives the distal segment;
        # a pelvis that decelerates slowly is leaking energy rather than passing it.
        _decel_hi = min(len(pelvis_om3d), peak_hip_frame + int(0.20 * fs) + 1)
        if _decel_hi - peak_hip_frame >= 3:
            _decel_seg = pelvis_om3d[peak_hip_frame:_decel_hi]
            _decel_slope = np.gradient(_decel_seg, dt)          # rad/s²
            pelvis_decel_rate = float(abs(np.min(_decel_slope)) * 180.0 / np.pi)
        else:
            pelvis_decel_rate = 0.0

        # ── Time to peak pelvis angular velocity ─────────────────────────────────
        # From swing start. The plant-relative form the literature reports is added
        # later in comprehensive_diagnosis(), where the plant frame is known.
        time_to_peak_pelvis_ms = float((peak_hip_frac - swing_start) * dt * 1000.0)

        return {
            'peak_hip_torque_Nm': float(peak_hip_torque),
            'peak_shoulder_torque_Nm': float(peak_shoulder_torque),
            'peak_hip_power_W': float(peak_hip_power),
            'peak_shoulder_power_W': float(peak_shoulder_power),
            'hip_inertia_kg_m2': float(hip_inertia),
            'shoulder_inertia_kg_m2': float(shoulder_inertia),
            'inertia_ratio': float(inertia_ratio),
            'hip_power_per_kg': float(hip_power_per_kg),
            'shoulder_power_per_kg': float(shoulder_power_per_kg),
            'max_separation_deg': float(max_separation),
            'sequence_timing_ms': float(sequence_timing_ms),
            'proper_sequence': proper_sequence,
            'sequence_indeterminate': sequence_indeterminate,
            'peak_arm_omega_rad_s': float(peak_arm_w_val),
            'peak_elb_omega_rad_s': float(peak_elb_w_val),
            'peak_shoulder_omega_rad_s': float(peak_shoulder_w),
            'peak_pelvis_omega_rad_s': float(peak_pelvis_w),
            'peak_pelvis_omega_3d_deg_s': peak_pelvis_omega_3d * 180.0 / np.pi,
            'peak_torso_omega_3d_deg_s': peak_torso_omega_3d * 180.0 / np.pi,
            'pelvis_omega': pelvis_omega,
            'pelvis_angle': pelvis_angle,
            'swing_start_frame': swing_start,
            # Absolute segment angular-velocity series, exported so the kinematic
            # sequence chart is drawn from the same signals the sequence metric is
            # computed from. A separate proxy for the chart drifts out of agreement
            # with the metric — it previously rendered the sequence reversed.
            'segment_omega': {
                'Pelvis': pelvis_om3d,
                'Torso': torso_om3d,
                # Absolute = parent + relative, matching the thorax convention above.
                'Lead Arm': np.abs(thorax_abs_omega + arm_omega),
                'Hands/Bat': np.abs(thorax_abs_omega + arm_omega + elb_omega),
            },
            # Peak frames as the sequence METRIC found them — sub-frame interpolated
            # and bounded to ±200ms of the pelvis peak. The chart must mark these
            # exact frames, or it will contradict the Sequence Quality it sits next to.
            'segment_peak_frames': {
                'Pelvis': peak_hip_frac,
                'Torso': peak_shoulder_frac,
                'Lead Arm': peak_arm_frac,
            },
            'pelvis_ke_J': float(pelvis_ke),
            'torso_ke_J': float(torso_ke),
            'arm_ke_J': float(arm_ke),
            'forearm_ke_J': float(forearm_ke),
            'bat_ke_J': float(bat_ke),
            'total_energy_transfer_J': float(total_energy_transfer),
            'torso_to_arm_transfer_ratio': float(torso_to_arm_ratio),
            'pelvis_to_torso_transfer_ratio': float(pelvis_to_torso_ratio),
            'torso_to_pelvis_rot_ratio': float(torso_to_pelvis_rot_ratio),
            'energy_transfer_proxy_pct': float(energy_transfer_proxy),
            # Blast Motion-aligned metrics
            'time_to_contact_s': time_to_contact_s,
            'rotational_acceleration_deg_s2': peak_rotational_accel,
            'pelvis_torso_contribution_pct': pelvis_torso_contribution,
            # Pelvis / lower-half metrics
            'x_factor_stretch_deg': x_factor_stretch,
            'separation_signal_saturated': _sep_saturated,
            'separation_source': separation_source,
            'separation_raw_peak_deg': _sep_raw_peak,
            'torso_arm_sequence_gap_ms': torso_arm_gap_ms,
            'pelvis_rotation_at_contact_deg': pelvis_rot_at_contact,
            'pelvis_rotation_excursion_deg': pelvis_rot_excursion,
            'contact_detection_method': contact_method,
            'pelvis_decel_rate_deg_s2': pelvis_decel_rate,
            'time_to_peak_pelvis_ms': time_to_peak_pelvis_ms,
            # Exported so comprehensive_diagnosis() can express the peak relative to
            # the plant frame, which is not known until stride analysis has run.
            'peak_pelvis_frame': float(peak_hip_frac),
            'contact_frame': int(contact_frame),
        }
        
    def _stride_from_markers(self, trc_data: pd.DataFrame, data: pd.DataFrame,
                             rotation: Dict) -> Optional[Dict]:
        """Stride length from foot markers: the horizontal distance between the two
        ankles at contact — the standard foot-to-foot definition.

        Measured as a snapshot at contact rather than as lead-foot displacement,
        because these captures frequently begin with the athlete already in his
        strided stance (tee and drill work), which makes any displacement-based
        measure read ~0 for a stance that is in fact wide open. A snapshot is
        indifferent to when recording started.

        Returns None if markers are missing.
        """
        if trc_data is None or rotation is None or 'pelvis_omega' not in rotation:
            return None
        try:
            cols = trc_data.columns
            if 'Neck_X' not in cols or 'LAnkle_X' not in cols or 'RAnkle_X' not in cols:
                return None
            # Identify the vertical axis empirically (lab frames vary): it's the one
            # where the neck sits far above the ankle.
            vert = max('XYZ', key=lambda a: float(np.mean(trc_data['Neck_' + a].values)
                                                  - np.mean(trc_data['LAnkle_' + a].values)))
            horiz = [a for a in 'XYZ' if a != vert]

            # Contact ≈ peak pelvis angular velocity.
            contact = int(min(int(np.argmax(np.abs(rotation['pelvis_omega']))), len(trc_data) - 1))

            def ankle(side):
                return np.array([float(trc_data[f'{side}Ankle_{a}'].values[contact]) for a in horiz])

            sep = float(np.linalg.norm(ankle('L') - ankle('R')))
            ratio = sep / self.body_height_m
            # Feet closer together than this at contact means the markers or the
            # contact frame are wrong — a hitter is never fully closed at contact.
            if ratio < STRIDE_MIN_RATIO:
                return {'stride_detected': False, 'stride_source': 'markers',
                        'stride_reason': 'foot separation at contact is implausibly small',
                        'stride_length_m': sep, 'stride_ratio': ratio}
            # The lead foot is the one further from the pelvis along the stride axis;
            # reported for context (lead-leg block uses its own detection).
            return {'stride_detected': True, 'stride_source': 'markers',
                    'stride_length_m': sep, 'stride_ratio': ratio,
                    'contact_frame': contact}
        except Exception:
            return None

    def calculate_stride_refined(self, data: pd.DataFrame, rotation: Dict = None,
                                 trc_data: pd.DataFrame = None) -> Dict:
        if 'pelvis_tx' not in data.columns or 'pelvis_ty' not in data.columns:
            return None
            
        # Optional: apply low-pass filter to positions if scipy is available
        dt = data['time'].diff().mean()
        fs = 1.0 / dt if dt > 0 else 60.0
        
        pelvis_x = data['pelvis_tx'].values
        pelvis_y = data['pelvis_ty'].values
        
        if HAS_SCIPY:
            pelvis_x = butter_lowpass_filter(pelvis_x, 15.0, fs)
            pelvis_y = butter_lowpass_filter(pelvis_y, 15.0, fs)
        
        # Event Detection: finding plant frame robustly
        # BUG 2 FIX: The .mot file covers the full at-bat (5+ seconds). The actual swing
        # is only the last ~0.5s. We find the swing onset by working BACKWARD from the
        # peak pelvis omega — the plant frame is the last frame before peak where omega
        # drops below a low threshold (50 deg/s), i.e. the last quiet moment before the swing.
        if rotation and 'pelvis_omega' in rotation:
            pelvis_omega = rotation['pelvis_omega']
            pelvis_omega_abs_deg = np.abs(pelvis_omega) * 180.0 / np.pi

            peak_frame = int(np.argmax(pelvis_omega_abs_deg))
            peak_sign = np.sign(rotation['pelvis_omega'][peak_frame])
            plant_frame = 0
            for i in range(peak_frame - 1, -1, -1):
                if np.sign(rotation['pelvis_omega'][i]) != peak_sign and pelvis_omega_abs_deg[i] > 20.0:
                    plant_frame = i
                    break
            if plant_frame == 0:
                quiet_thresh = pelvis_omega_abs_deg[peak_frame] * 0.15
                for i in range(peak_frame, -1, -1):
                    if pelvis_omega_abs_deg[i] < quiet_thresh:
                        plant_frame = i
                        break
            plant_method = "sign_reversal_onset"
        else:
            plant_frame = len(data) // 2
            plant_method = "fallback_midframe"
        
        # ── Stride length ──────────────────────────────────────────────────
        # Stride length is the LEAD FOOT's travel to plant, so foot markers are
        # the only correct source. Without them we can still report how far the
        # pelvis travelled, but that is a different, much smaller quantity — it
        # is NOT benchmarked against stride-length research, because scoring it
        # as if it were a stride told every athlete their stride was "off target"
        # when the system simply could not see it.
        marker = self._stride_from_markers(trc_data, data, rotation)

        if marker and marker.get('stride_detected'):
            stride_length = marker['stride_length_m']
            stride_ratio = marker['stride_ratio']
            stride_detected, stride_source = True, 'markers'
            stride_reason = None
        else:
            # Pelvis travel in the HORIZONTAL plane (tx/tz) — never the vertical
            # axis — measured from a quiet stance reference near swing onset
            # rather than frame 0, since these trials span a whole at-bat.
            pelvis_z = data['pelvis_tz'].values if 'pelvis_tz' in data.columns else np.zeros(len(pelvis_x))
            if HAS_SCIPY:
                pelvis_z = butter_lowpass_filter(pelvis_z, 15.0, fs)
            ref = max(0, plant_frame - int(0.25 / dt)) if dt > 0 else 0
            start_pos = np.array([np.median(pelvis_x[ref:plant_frame + 1]),
                                  np.median(pelvis_z[ref:plant_frame + 1])])
            plant_pos = np.array([pelvis_x[plant_frame], pelvis_z[plant_frame]])
            stride_length = float(np.linalg.norm(plant_pos - start_pos))
            stride_ratio = stride_length / self.body_height_m
            stride_detected, stride_source = False, ('markers' if marker else 'pelvis_fallback')
            stride_reason = (marker or {}).get(
                'stride_reason',
                'no marker data — stride length needs foot markers (.trc)')

        # Only meaningful when a real stride was measured.
        optimal_stride_ratio = 0.75
        stride_efficiency_pct = (stride_ratio / optimal_stride_ratio) * 100.0 if stride_detected else 0.0

        return {
            'stride_length_m': float(stride_length),
            'stride_length_ft': float(stride_length * 3.28084),
            'stride_ratio': float(stride_ratio),
            'stride_efficiency_pct': float(stride_efficiency_pct),
            'stride_detected': bool(stride_detected),
            'stride_source': stride_source,
            'stride_reason': stride_reason,
            'lead_side': (marker or {}).get('lead_side'),
            'plant_frame': int(plant_frame),
            'plant_time': float(data['time'].iloc[plant_frame]),
            'plant_method': plant_method
        }
        
    def estimate_hand_speed(self, rotation: Dict, trc_metrics: Dict = None) -> Dict:
        """Estimate peak hand/wrist speed in mph.

        Priority:
          1. Direct wrist marker velocity from TRC data (most accurate).
          2. Reconstructed from distal segment angular velocities × lever arm
             (forearm + hand ≈ 14.6% + 5.8% of height from de Leva 1996).
        """
        # ── Method 1: TRC wrist markers ─────────────────────────────────────
        if trc_metrics and trc_metrics.get('max_hand_speed_mps', 0) > 0:
            hand_speed_mps = float(trc_metrics['max_hand_speed_mps'])
            source = 'trc_marker'
        else:
            # ── Method 2: Angular velocity × lever arm ───────────────────────
            # Forearm length ≈ 14.6% of height; hand ≈ 5.8% → total ≈ 20.4%
            lever_arm_m = self.body_height_m * 0.204

            peak_arm_w   = rotation.get('peak_arm_omega_rad_s', 0.0) if rotation else 0.0
            peak_elb_w   = rotation.get('peak_elb_omega_rad_s', 0.0) if rotation else 0.0

            if peak_arm_w > 0 or peak_elb_w > 0:
                hand_speed_mps = (peak_arm_w + peak_elb_w) * lever_arm_m
            else:
                # Final fallback: derive from shoulder omega × full arm span
                peak_shoulder_w = rotation.get('peak_shoulder_omega_rad_s', 0.0) if rotation else 0.0
                full_arm = self.body_height_m * 0.366  # upper arm + forearm + hand
                hand_speed_mps = peak_shoulder_w * full_arm

            source = 'angular_velocity'

        hand_speed_mph = hand_speed_mps * 2.23694

        return {
            'estimated_hand_speed_mph': float(hand_speed_mph),
            'estimated_hand_speed_mps': float(hand_speed_mps),
            'source': source,
        }
        
    def calculate_lower_body_kinematics(self, data: pd.DataFrame) -> Dict:
        """Extract bilateral hip/knee/ankle kinematics and lower-body Newton-Euler kinetics."""
        dt = data['time'].diff().mean()
        fs = 1.0 / dt if dt > 0 else 60.0

        def _filt(arr):
            if HAS_SCIPY:
                return butter_lowpass_filter(arr, 15.0, fs)
            return smooth_data(arr, 11)

        def _diff2(arr):
            if HAS_SCIPY:
                w = max(11, int(0.10 * fs) | 1)
                return savgol_smooth_and_diff(arr, window=w, polyorder=3, deriv=2, dt=dt)
            v = np.gradient(smooth_data(arr, 11), dt)
            return np.gradient(v, dt)

        result = {}
        # ── Bilateral joint angles ──────────────────────────────────────────
        for side in ('r', 'l'):
            for joint, col in [('hip_flex', f'hip_flexion_{side}'),
                                ('hip_add',  f'hip_adduction_{side}'),
                                ('hip_rot',  f'hip_rotation_{side}'),
                                ('knee',     f'knee_angle_{side}'),
                                ('ankle',    f'ankle_angle_{side}')]:
                if col in data.columns:
                    arr = _filt(data[col].values)
                    result[f'{joint}_{side}'] = arr
                    result[f'peak_{joint}_{side}_deg'] = float(np.max(np.abs(arr)))

        # ── Asymmetry ───────────────────────────────────────────────────────
        for joint in ('hip_flex', 'knee'):
            r = result.get(f'peak_{joint}_r_deg', 0.0)
            l = result.get(f'peak_{joint}_l_deg', 0.0)
            result[f'{joint}_asymmetry_deg'] = float(abs(r - l))

        # ── Lower-body Newton-Euler kinetics (τ = I·α, P = τ·ω) ────────────
        # Segment inertias: thigh and shank from SEGMENT_PARAMS
        thigh_I = self.segments['thigh']['I']
        shank_I = self.segments['shank']['I']

        for side in ('r', 'l'):
            for seg, I_val, col_key in [('knee', thigh_I, f'knee_{side}'),
                                         ('ankle', shank_I, f'ankle_{side}')]:
                arr = result.get(col_key)
                if arr is None:
                    continue
                angle_rad = np.deg2rad(arr)
                alpha = _diff2(angle_rad)
                omega = np.gradient(angle_rad, dt)
                torque = I_val * alpha
                power  = torque * omega
                result[f'peak_{seg}_torque_{side}_Nm'] = float(np.max(np.abs(torque)))
                result[f'peak_{seg}_power_{side}_W']   = float(np.max(np.abs(power)))

        # ── Lead-hip internal-rotation torque (MacWilliams et al. 1998) ─────────
        # The lead hip is what actually drives pelvis rotation, so its IR torque is
        # the kinetic counterpart to peak pelvis angular velocity: same event, one
        # measured as cause and one as effect. A hitter can show respectable pelvis
        # speed off a passive fall rather than an active drive, and only the torque
        # separates those.
        #
        # Lead leg follows handedness: a right-handed hitter strides onto the left
        # leg. Where handedness is unknown we leave this at 0 rather than guess —
        # picking the wrong leg reports the trail hip, which is a different action.
        lead_hip_side = {'right': 'l', 'left': 'r'}.get(self.handedness)
        if lead_hip_side is not None:
            hip_rot = result.get(f'hip_rot_{lead_hip_side}')
            if hip_rot is not None:
                hip_rot_alpha = _diff2(np.deg2rad(hip_rot))
                result['peak_lead_hip_ir_torque_Nm'] = float(
                    np.max(np.abs(thigh_I * hip_rot_alpha))
                )

        # ── Bilateral arm kinematics ────────────────────────────────────────
        for col, key in [('arm_flex_l', 'arm_flex_l'), ('elbow_flex_l', 'elbow_flex_l'),
                          ('pro_sup_r', 'prosup_r'), ('pro_sup_l', 'prosup_l')]:
            if col in data.columns:
                arr = _filt(data[col].values)
                result[f'peak_{key}_deg'] = float(np.max(np.abs(arr)))

        # Arm flex asymmetry (right already computed in rotational torques)
        r_flex = float(np.max(np.abs(_filt(data['arm_flex_r'].values)))) if 'arm_flex_r' in data.columns else 0.0
        l_flex = result.get('peak_arm_flex_l_deg', 0.0)
        result['arm_flex_asymmetry_deg'] = float(abs(r_flex - l_flex))

        return result

    def calculate_linear_inverse_dynamics(self, data: pd.DataFrame) -> Dict:
        """Compute pelvis segment joint reaction forces via F = m·a on all 3 translation axes."""
        dt = data['time'].diff().mean()
        fs = 1.0 / dt if dt > 0 else 60.0
        required = {'pelvis_tx', 'pelvis_ty', 'pelvis_tz'}
        if not required.issubset(data.columns):
            return {}

        pelvis_mass = self.body_mass_kg * SEGMENT_PARAMS['trunk']['mass_pct']

        def _accel(col):
            raw = data[col].values
            if HAS_SCIPY:
                filt = butter_lowpass_filter(raw, 15.0, fs)
                w = max(11, int(0.10 * fs) | 1)
                return savgol_smooth_and_diff(filt, window=w, polyorder=3, deriv=2, dt=dt)
            smooth = smooth_data(raw, 11)
            return np.gradient(np.gradient(smooth, dt), dt)

        ax = _accel('pelvis_tx')  # anterior-posterior
        ay = _accel('pelvis_ty')  # vertical
        az = _accel('pelvis_tz')  # lateral

        Fx = pelvis_mass * ax
        Fy = pelvis_mass * ay
        Fz = pelvis_mass * az
        F_res = np.sqrt(Fx**2 + Fy**2 + Fz**2)

        return {
            'pelvis_force_ap':         Fx,
            'pelvis_force_vert':       Fy,
            'pelvis_force_lat':        Fz,
            'peak_pelvis_force_ap_N':  float(np.max(np.abs(Fx))),
            'peak_pelvis_force_vert_N':float(np.max(np.abs(Fy))),
            'peak_pelvis_force_lat_N': float(np.max(np.abs(Fz))),
            'peak_pelvis_force_resultant_N': float(np.max(F_res)),
        }

    def calculate_weight_shift(self, data: pd.DataFrame, plant_frame: int) -> Dict:
        """Lateral balance and weight-shift metrics from pelvis_list and pelvis_tz."""
        dt = data['time'].diff().mean()
        fs = 1.0 / dt if dt > 0 else 60.0
        result = {}

        def _filt(arr):
            if HAS_SCIPY:
                return butter_lowpass_filter(arr, 15.0, fs)
            return smooth_data(arr, 11)

        # Full 6-DOF pelvis ranges
        for col, key in [('pelvis_tilt', 'pelvis_tilt_range_deg'),
                          ('pelvis_list', 'pelvis_list_range_deg')]:
            if col in data.columns:
                arr = _filt(data[col].values)
                result[key] = float(np.ptp(arr))  # peak-to-peak range

        if 'pelvis_tz' in data.columns:
            tz = _filt(data['pelvis_tz'].values)
            result['pelvis_tz_range_m'] = float(np.ptp(tz))
            result['lateral_sway_range_m'] = float(np.ptp(tz))
            plant_idx = min(plant_frame, len(tz) - 1)
            result['lateral_sway_at_plant_m'] = float(tz[plant_idx] - tz[0])
            # Timing: at what % of the swing does peak lateral shift occur?
            peak_lat_frame = int(np.argmax(np.abs(tz - tz[0])))
            result['weight_shift_timing_pct'] = float(peak_lat_frame / max(1, len(tz) - 1) * 100.0)

        return result

    def _calculate_trc_sequence_timing(self, trc_data: pd.DataFrame, mot_data: pd.DataFrame) -> Optional[Dict]:
        """Compute pelvis-to-torso sequence timing (ms) from TRC marker velocities.
        Uses hip joint center midpoint for pelvis and thorax proximal marker for torso.
        Returns dict with keys: timing_ms, proper_sequence, indeterminate.
        Positive timing_ms = pelvis peaks before torso (correct sequence)."""
        from scipy.signal import butter, filtfilt, savgol_filter
        t = trc_data['Time'].values if 'Time' in trc_data.columns else trc_data['time'].values
        dt = np.diff(t).mean()
        fs = 1.0 / dt

        def filt(arr):
            b, a = butter(4, min(15.0 / (0.5 * fs), 0.99), btype='low')
            return filtfilt(b, a, arr)

        def speed_mag(cx, cy, cz):
            w = max(11, int(0.1 * fs) | 1)
            vx = savgol_filter(filt(cx), w, 3, deriv=1, delta=dt)
            vy = savgol_filter(filt(cy), w, 3, deriv=1, delta=dt)
            vz = savgol_filter(filt(cz), w, 3, deriv=1, delta=dt)
            return np.sqrt(vx**2 + vy**2 + vz**2)

        # Pelvis: midpoint of hip joint centers
        pelvis_cols = [('RASI','LASI'), ('rhjc','lhjc'), ('RHip','LHip')]
        pelvis_spd = None
        for r, l in pelvis_cols:
            if f'{r}_X' in trc_data.columns and f'{l}_X' in trc_data.columns:
                cx = (trc_data[f'{r}_X'].values + trc_data[f'{l}_X'].values) / 2
                cy = (trc_data[f'{r}_Y'].values + trc_data[f'{l}_Y'].values) / 2
                cz = (trc_data[f'{r}_Z'].values + trc_data[f'{l}_Z'].values) / 2
                pelvis_spd = speed_mag(cx, cy, cz)
                break

        # Thorax: CLAV or STRN marker
        thorax_spd = None
        for m in ['CLAV', 'STRN', 'C7', 'T10']:
            if f'{m}_X' in trc_data.columns:
                thorax_spd = speed_mag(trc_data[f'{m}_X'].values, trc_data[f'{m}_Y'].values, trc_data[f'{m}_Z'].values)
                break

        if pelvis_spd is None or thorax_spd is None:
            return None

        # Find swing start from mot fp_10_time, mapped to TRC time base.
        # Limit search to ≤400ms before peak, same constraint as joint-angle path.
        fp_time = mot_data.attrs.get('fp_10_time', t[len(t)//3])
        fp_idx = int(np.argmin(np.abs(t - fp_time)))

        # Pelvis peak: bounded search forward from fp_idx
        _max_frames = int(0.40 * fs)
        search_end = min(len(pelvis_spd), fp_idx + _max_frames + int(0.5*fs))
        peak_pelvis = fp_idx + int(np.argmax(pelvis_spd[fp_idx:search_end]))

        # Thorax peak: constrained to ±200ms around pelvis peak (same as joint-angle path)
        _search_half = int(0.20 * fs)
        thr_lo = max(fp_idx, peak_pelvis - _search_half)
        thr_hi = min(len(thorax_spd), peak_pelvis + _search_half + 1)
        peak_thorax = thr_lo + int(np.argmax(thorax_spd[thr_lo:thr_hi]))

        timing_ms = float((peak_thorax - peak_pelvis) * dt * 1000.0)
        frame_ms = dt * 1000.0
        peak_pelvis_spd = float(pelvis_spd[peak_pelvis])

        # Indeterminate when: timing within 1 frame OR pelvis speed too low (weak movement)
        indeterminate = bool(abs(timing_ms) < frame_ms or peak_pelvis_spd < 0.10)

        return {
            'timing_ms': timing_ms,
            'proper_sequence': bool(timing_ms <= 0),  # torax ≤ pelvis = correct (pos = pelvis first)
            'indeterminate': indeterminate,
        }

    def comprehensive_diagnosis(self, kinematics: pd.DataFrame, filename: str, trc_data: pd.DataFrame = None, verbose: bool = False) -> Dict:
        trc_metrics = self.calculate_trc_metrics(trc_data) if trc_data is not None else {'max_hand_speed_mph': 0.0, 'max_hand_speed_mps': 0.0}
        wrist_speed_mps = float(trc_metrics.get('max_hand_speed_mps', 0.0))
        rotation = self.calculate_rotational_torques_refined(
            kinematics, wrist_speed_mps=wrist_speed_mps,
            hand_speed_peak_time_s=trc_metrics.get('hand_contact_time_s', 0.0),
            marker_separation=self.calculate_marker_separation(trc_data))
        stride = self.calculate_stride_refined(kinematics, rotation, trc_data=trc_data)
        hand_speed = self.estimate_hand_speed(rotation, trc_metrics)
        lower_body = self.calculate_lower_body_kinematics(kinematics)
        linear_id  = self.calculate_linear_inverse_dynamics(kinematics)
        plant_frame = stride['plant_frame'] if stride else len(kinematics) // 2
        weight_shift = self.calculate_weight_shift(kinematics, plant_frame)

        # Note: the TRC-based sequence timing override was removed. The joint-angle
        # path now uses absolute thorax velocity (pelvis_rotation + lumbar_rotation)
        # with a ±400ms bounded search window — the root causes of the original
        # unreliable values. The TRC path measured translational marker speed, not
        # angular velocity, which made it less accurate than the corrected joint-angle path.

        # GRF estimation from whole-body CoM (requires TRC markers)
        grf_data = {}
        if trc_data is not None:
            try:
                from grf_estimation import estimate_grf, grf_summary
                grf_result = estimate_grf(trc_data, body_mass_kg=self.body_mass_kg)
                grf_data = grf_summary(grf_result, self.body_mass_kg)
            except Exception:
                pass
        
        findings = []
        recommendations = []
        efficiency_score = 100
        
        if rotation:
            # === KINEMATIC SEQUENCING (Proximal-To-Distal) ===
            if not rotation['proper_sequence']:
                findings.append("Kinematic Sequence Reversal: Distal segments firing prior to proximal.")
                recommendations.append("URGENT: Initiate swing from the ground up. The gold standard sequence requires Pelvis → Torso → Arms to maximize energy transfer (Driveline OBP: energy transfer features are the #1 predictor of velocity).")
                efficiency_score -= 25
            elif rotation['sequence_timing_ms'] < 20:
                findings.append("Poor Synchronization: Pelvis and Torso accelerating simultaneously.")
                recommendations.append("Increase temporal separation between segment rotations to maximize the stretch-shortening cycle. Elite hitters show 30-60ms pelvis-to-torso lag.")
                efficiency_score -= 15
            else:
                findings.append("Optimal Proximal-to-Distal Kinetic Chain demonstrated.")
                
            # === X-FACTOR SEPARATION ===
            sep = rotation['max_separation_deg']
            if sep < 30:
                findings.append(f"Restricted X-Factor Stretch ({sep:.1f}°).")
                recommendations.append("Improve core mobility. Elite hitters establish 35-55° of hip-shoulder separation to store elastic energy in the obliques and core musculature.")
                efficiency_score -= 20
            elif sep > 75:
                findings.append(f"Hyper-extended X-Factor ({sep:.1f}°).")
                recommendations.append("Control torso rotation to prevent extreme X-Factor energy leaks and potential oblique strain.")
                efficiency_score -= 10
            else:
                findings.append(f"Elite X-Factor Separation Stretch ({sep:.1f}°).")
                
            # === ROTATIONAL POWER ===
            if rotation['hip_power_per_kg'] < 12:
                findings.append("Sub-optimal Pelvic Rotational Power.")
                recommendations.append("Engage lower-half ground reaction forces more aggressively. Driveline research shows GRF lead/rear ratio is a key predictor of velocity.")
                efficiency_score -= 15
            else:
                findings.append("Elite Lower-Half Power Generation.")
                
            # === DRIVELINE-INSPIRED: ENERGY TRANSFER ANALYSIS ===
            chain_eff = rotation.get('energy_transfer_proxy_pct', 0.0)
            torso_arm_ratio = rotation.get('torso_to_arm_transfer_ratio', 0.0)
            pelvis_torso_ratio = rotation.get('pelvis_to_torso_transfer_ratio', 0.0)
            torso_pelvis_rot = rotation.get('torso_to_pelvis_rot_ratio', 0.0)
            
            # Chain Efficiency Analysis
            if chain_eff < 15:
                findings.append(f"Low Kinetic Chain Transfer Efficiency ({chain_eff:.1f}%).")
                recommendations.append("Energy is trapped in the proximal segments (pelvis/torso) and not reaching the hands. Focus on sequential acceleration: let the pelvis decelerate as the torso fires, creating a 'whip' effect that amplifies distal segment speed.")
                efficiency_score -= 15
            elif chain_eff > 40:
                findings.append(f"Elite Distal Energy Amplification ({chain_eff:.1f}%).")
            else:
                findings.append(f"Adequate Kinetic Chain Transfer ({chain_eff:.1f}%).")
                recommendations.append("Good foundation. To push toward elite: focus on violent hip deceleration at front foot plant to 'whip' stored energy into the torso and arms.")
                
            # Torso-to-Pelvis Rotational Velocity Ratio (Driveline top feature)
            if torso_pelvis_rot < 0.8:
                findings.append(f"Low Torso-to-Pelvis Rotational Velocity Ratio ({torso_pelvis_rot:.2f}).")
                recommendations.append("Your torso is not rotating faster than your pelvis. In elite swings, the torso must rotationally 'catch up' and surpass the pelvis to create the proximal-to-distal velocity amplification.")
                efficiency_score -= 10
            elif torso_pelvis_rot > 1.3:
                findings.append(f"Excellent Torso-to-Pelvis Velocity Amplification ({torso_pelvis_rot:.2f}).")
            else:
                findings.append(f"Adequate Torso-to-Pelvis Ratio ({torso_pelvis_rot:.2f}).")
            
            # Total Energy (absolute output scaled by demographics)
            total_ke = rotation.get('total_energy_transfer_J', 0.0)
            ke_per_kg = total_ke / self.body_mass_kg if self.body_mass_kg > 0 else 0.0
            if ke_per_kg > 5.0:
                findings.append(f"Elite Total Kinetic Chain Energy ({total_ke:.0f} J, {ke_per_kg:.1f} J/kg).")
            elif ke_per_kg > 2.0:
                findings.append(f"Good Total Kinetic Chain Energy ({total_ke:.0f} J, {ke_per_kg:.1f} J/kg).")
            else:
                findings.append(f"Low Total Kinetic Chain Energy ({total_ke:.0f} J, {ke_per_kg:.1f} J/kg).")
                recommendations.append("Overall rotational energy production is low. Address with explosive rotational training (med ball throws, cable rotations) and improved sequencing.")
                
        if stride:
            eff = stride['stride_efficiency_pct']
            if eff < 60:
                findings.append("Restricted Stride Length / Insufficient Linear Momentum.")
                recommendations.append("Lengthen stride naturally. Building forward momentum prior to foot plant is crucial for generating ground reaction forces that translate into rotational power.")
                efficiency_score -= 15
            elif eff > 130:
                findings.append("Over-striding / Compromised Postural Stability.")
                recommendations.append("Shorten stride. Over-striding compromises the ability to brace the lead knee firmly for efficient energy transfer.")
                efficiency_score -= 10
            else:
                findings.append("Efficient Stride Mechanics & Center of Mass Control.")
                
        hand_spd_mph = hand_speed['estimated_hand_speed_mph'] if hand_speed else 0.0
        benchmarks = SKILL_LEVEL_BENCHMARKS.get(self.skill_level, {})
        hs_lo, hs_hi = benchmarks.get('max_hand_speed_mph', (35, 55))
        if hand_spd_mph > 0:
            if hand_spd_mph < hs_lo:
                findings.append(f"Below-Average Hand Speed ({hand_spd_mph:.1f} mph; target {hs_lo}–{hs_hi} mph).")
                recommendations.append(f"Hand speed is below the {self.skill_level} benchmark. Focus on lead-leg bracing at contact and sequential deceleration of the pelvis to whip maximum energy into the hands.")
            elif hand_spd_mph >= hs_hi:
                findings.append(f"Elite Hand Speed ({hand_spd_mph:.1f} mph).")
            else:
                findings.append(f"Adequate Hand Speed ({hand_spd_mph:.1f} mph; target {hs_lo}–{hs_hi} mph).")

        # High speed / low efficiency discrepancy
        if hand_spd_mph > 0 and hand_spd_mph > hs_hi * 1.1 and efficiency_score < 75:
            findings.append("⚠️ High Hand Speed / Low Efficiency Discrepancy (Brute Force Mode).")
            recommendations.append("You are generating high hand speed with raw strength rather than mechanical efficiency. Improving kinetic chain transfer would unlock more speed with less effort and lower injury risk.")

        # === LOWER-BODY KINEMATICS ===
        if lower_body:
            knee_asym = lower_body.get('knee_asymmetry_deg', 0.0)
            hip_asym  = lower_body.get('hip_flex_asymmetry_deg', 0.0)
            if knee_asym > 15:
                findings.append(f"Bilateral Knee Flexion Asymmetry ({knee_asym:.1f}°).")
                recommendations.append("Significant lead/trail knee asymmetry detected. Uneven loading increases injury risk and reduces rotational stability. Address with single-leg strength work.")
                efficiency_score -= 10
            if hip_asym > 20:
                findings.append(f"Bilateral Hip Flexion Asymmetry ({hip_asym:.1f}°).")
                recommendations.append("Hip flexion asymmetry suggests uneven weight distribution at load. Focus on balanced hip hinge mechanics.")
                efficiency_score -= 8

            peak_knee_r = lower_body.get('peak_knee_r_deg', 0.0)
            if peak_knee_r < 20:
                findings.append(f"Insufficient Trail Knee Flexion ({peak_knee_r:.1f}°).")
                recommendations.append("Trail knee should flex 30-50° during load to store elastic energy. Increase hip hinge depth.")
                efficiency_score -= 10
            elif peak_knee_r > 70:
                findings.append(f"Excessive Trail Knee Flexion ({peak_knee_r:.1f}°).")
                recommendations.append("Over-flexed trail knee reduces rotational power and increases knee stress.")
                efficiency_score -= 5

        # === LINEAR INVERSE DYNAMICS (Pelvis Joint Reaction Forces) ===
        if linear_id:
            F_res = linear_id.get('peak_pelvis_force_resultant_N', 0.0)
            F_lat = linear_id.get('peak_pelvis_force_lat_N', 0.0)
            F_ap  = linear_id.get('peak_pelvis_force_ap_N', 0.0)
            if F_res > 0:
                findings.append(f"Peak Pelvis Resultant Force: {F_res:.0f} N ({F_res/self.body_mass_kg:.1f} N/kg).")
            if F_lat > 0.3 * F_res:
                findings.append(f"High Lateral Pelvis Force Component ({F_lat:.0f} N, {F_lat/F_res*100:.0f}% of resultant).")
                recommendations.append("Excessive lateral pelvis force indicates energy leaking sideways rather than rotating. Improve hip-to-hip weight transfer timing.")
                efficiency_score -= 8

        # === WEIGHT SHIFT / LATERAL BALANCE ===
        if weight_shift:
            sway = weight_shift.get('lateral_sway_range_m', 0.0)
            sway_at_plant = weight_shift.get('lateral_sway_at_plant_m', 0.0)
            shift_timing = weight_shift.get('weight_shift_timing_pct', 0.0)
            if sway > 0.12:
                findings.append(f"Excessive Lateral Sway ({sway*100:.1f} cm range).")
                recommendations.append("Lateral sway > 12 cm indicates poor rotational axis stability. Keep the pelvis centered over the rear hip during load.")
                efficiency_score -= 10
            elif sway > 0:
                findings.append(f"Controlled Lateral Sway ({sway*100:.1f} cm range).")
            if abs(sway_at_plant) > 0.06:
                findings.append(f"Pelvis laterally displaced at plant ({sway_at_plant*100:.1f} cm from start).")
                recommendations.append("Pelvis should return near center by front foot plant. Excessive lateral displacement at contact reduces rotational power.")
                efficiency_score -= 5

        # === TIME TO PEAK PELVIS ANGULAR VELOCITY, RELATIVE TO FOOT PLANT ===
        # Welch et al. (1995) report this interval from front-foot plant, not from
        # swing start, and elite hitters cluster around 50–100 ms. It is computed
        # here rather than in calculate_rotational_metrics() because the plant frame
        # is not known until stride analysis has run.
        #
        # Negative means the pelvis peaked before the foot ever landed — the athlete
        # spun open early instead of rotating against a planted front side. That is a
        # real and diagnostic pattern, so the sign is preserved rather than clamped.
        time_to_peak_pelvis_from_plant_ms = 0.0
        if rotation and stride:
            _peak_pelvis_frame = rotation.get('peak_pelvis_frame')
            _plant = stride.get('plant_frame')
            _dt_kin = kinematics['time'].diff().mean()
            if (_peak_pelvis_frame is not None and _plant is not None
                    and _dt_kin and _dt_kin > 0):
                time_to_peak_pelvis_from_plant_ms = float(
                    (_peak_pelvis_frame - _plant) * _dt_kin * 1000.0
                )

        metrics = RefinedSwingMetrics(
            peak_hip_torque_Nm=rotation['peak_hip_torque_Nm'] if rotation else 0.0,
            peak_shoulder_torque_Nm=rotation['peak_shoulder_torque_Nm'] if rotation else 0.0,
            peak_hip_power_W=rotation['peak_hip_power_W'] if rotation else 0.0,
            peak_shoulder_power_W=rotation['peak_shoulder_power_W'] if rotation else 0.0,
            hip_inertia_kg_m2=rotation['hip_inertia_kg_m2'] if rotation else 0.0,
            shoulder_inertia_kg_m2=rotation['shoulder_inertia_kg_m2'] if rotation else 0.0,
            inertia_ratio=rotation['inertia_ratio'] if rotation else 0.0,
            hip_power_per_kg=rotation['hip_power_per_kg'] if rotation else 0.0,
            shoulder_power_per_kg=rotation['shoulder_power_per_kg'] if rotation else 0.0,
            max_separation_deg=rotation['max_separation_deg'] if rotation else 0.0,
            sequence_timing_ms=rotation['sequence_timing_ms'] if rotation else 0.0,
            proper_sequence=rotation['proper_sequence'] if rotation else False,
            stride_length_m=stride['stride_length_m'] if stride else 0.0,
            stride_ratio=stride['stride_ratio'] if stride else 0.0,
            stride_efficiency_pct=stride['stride_efficiency_pct'] if stride else 0.0,
            plant_frame=stride['plant_frame'] if stride else 0,
            plant_method=stride['plant_method'] if stride else "none",
            estimated_hand_speed_mph=hand_speed['estimated_hand_speed_mph'] if hand_speed else 0.0,
            swing_composite_score_v1=max(0, efficiency_score),
            # Driveline Energy Transfer Metrics
            pelvis_ke_J=rotation.get('pelvis_ke_J', 0.0) if rotation else 0.0,
            torso_ke_J=rotation.get('torso_ke_J', 0.0) if rotation else 0.0,
            arm_ke_J=rotation.get('arm_ke_J', 0.0) if rotation else 0.0,
            forearm_ke_J=rotation.get('forearm_ke_J', 0.0) if rotation else 0.0,
            total_energy_transfer_J=rotation.get('total_energy_transfer_J', 0.0) if rotation else 0.0,
            torso_to_arm_transfer_ratio=rotation.get('torso_to_arm_transfer_ratio', 0.0) if rotation else 0.0,
            pelvis_to_torso_transfer_ratio=rotation.get('pelvis_to_torso_transfer_ratio', 0.0) if rotation else 0.0,
            torso_to_pelvis_rot_ratio=rotation.get('torso_to_pelvis_rot_ratio', 0.0) if rotation else 0.0,
            energy_transfer_proxy_pct=rotation.get('energy_transfer_proxy_pct', 0.0) if rotation else 0.0,
            # Full 6-DOF pelvis
            pelvis_tilt_range_deg=weight_shift.get('pelvis_tilt_range_deg', 0.0),
            pelvis_list_range_deg=weight_shift.get('pelvis_list_range_deg', 0.0),
            pelvis_tz_range_m=weight_shift.get('pelvis_tz_range_m', 0.0),
            # Lower-body kinematics
            peak_hip_flexion_r_deg=lower_body.get('peak_hip_flex_r_deg', 0.0),
            peak_hip_flexion_l_deg=lower_body.get('peak_hip_flex_l_deg', 0.0),
            peak_knee_flexion_r_deg=lower_body.get('peak_knee_r_deg', 0.0),
            peak_knee_flexion_l_deg=lower_body.get('peak_knee_l_deg', 0.0),
            peak_ankle_dorsiflexion_r_deg=lower_body.get('peak_ankle_r_deg', 0.0),
            peak_ankle_dorsiflexion_l_deg=lower_body.get('peak_ankle_l_deg', 0.0),
            hip_flexion_asymmetry_deg=lower_body.get('hip_flex_asymmetry_deg', 0.0),
            knee_flexion_asymmetry_deg=lower_body.get('knee_asymmetry_deg', 0.0),
            # Lower-body kinetics
            peak_knee_torque_r_Nm=lower_body.get('peak_knee_torque_r_Nm', 0.0),
            peak_knee_torque_l_Nm=lower_body.get('peak_knee_torque_l_Nm', 0.0),
            peak_ankle_torque_r_Nm=lower_body.get('peak_ankle_torque_r_Nm', 0.0),
            peak_ankle_torque_l_Nm=lower_body.get('peak_ankle_torque_l_Nm', 0.0),
            peak_knee_power_r_W=lower_body.get('peak_knee_power_r_W', 0.0),
            peak_knee_power_l_W=lower_body.get('peak_knee_power_l_W', 0.0),
            # Linear inverse dynamics
            peak_pelvis_force_ap_N=linear_id.get('peak_pelvis_force_ap_N', 0.0),
            peak_pelvis_force_vert_N=linear_id.get('peak_pelvis_force_vert_N', 0.0),
            peak_pelvis_force_lat_N=linear_id.get('peak_pelvis_force_lat_N', 0.0),
            peak_pelvis_force_resultant_N=linear_id.get('peak_pelvis_force_resultant_N', 0.0),
            # Weight shift / lateral balance
            lateral_sway_range_m=weight_shift.get('lateral_sway_range_m', 0.0),
            lateral_sway_at_plant_m=weight_shift.get('lateral_sway_at_plant_m', 0.0),
            weight_shift_timing_pct=weight_shift.get('weight_shift_timing_pct', 0.0),
            # Bilateral arm kinematics
            peak_arm_flex_l_deg=lower_body.get('peak_arm_flex_l_deg', 0.0),
            peak_elbow_flex_l_deg=lower_body.get('peak_elbow_flex_l_deg', 0.0),
            arm_flex_asymmetry_deg=lower_body.get('arm_flex_asymmetry_deg', 0.0),
            peak_prosup_r_deg=lower_body.get('peak_prosup_r_deg', 0.0),
            peak_prosup_l_deg=lower_body.get('peak_prosup_l_deg', 0.0),
            # Blast Motion-aligned metrics
            time_to_contact_s=rotation.get('time_to_contact_s', 0.0) if rotation else 0.0,
            rotational_acceleration_deg_s2=rotation.get('rotational_acceleration_deg_s2', 0.0) if rotation else 0.0,
            pelvis_torso_contribution_pct=rotation.get('pelvis_torso_contribution_pct', 0.0) if rotation else 0.0,
            max_hand_speed_mph=trc_metrics.get('max_hand_speed_mph', 0.0),
            peak_pelvis_omega_3d_deg_s=rotation.get('peak_pelvis_omega_3d_deg_s', 0.0) if rotation else 0.0,
            # Pelvis / lower-half metrics
            x_factor_stretch_deg=rotation.get('x_factor_stretch_deg', 0.0) if rotation else 0.0,
            torso_arm_sequence_gap_ms=rotation.get('torso_arm_sequence_gap_ms', 0.0) if rotation else 0.0,
            pelvis_rotation_at_contact_deg=rotation.get('pelvis_rotation_at_contact_deg', 0.0) if rotation else 0.0,
            pelvis_rotation_excursion_deg=rotation.get('pelvis_rotation_excursion_deg', 0.0) if rotation else 0.0,
            contact_detection_method=rotation.get('contact_detection_method', 'none') if rotation else 'none',
            peak_lead_hip_ir_torque_Nm=lower_body.get('peak_lead_hip_ir_torque_Nm', 0.0),
            pelvis_decel_rate_deg_s2=rotation.get('pelvis_decel_rate_deg_s2', 0.0) if rotation else 0.0,
            time_to_peak_pelvis_ms=rotation.get('time_to_peak_pelvis_ms', 0.0) if rotation else 0.0,
            time_to_peak_pelvis_from_plant_ms=time_to_peak_pelvis_from_plant_ms,
        )
        
        # Terminal printing if verbose
        if verbose:
            print("\n" + "="*70)
            print(f"REFINED SWING ANALYSIS: {filename}")
            print("="*70)
            
            if rotation:
                print(f"\n🔄 ROTATIONAL MECHANICS (Corrected Inertias):")
                print(f"   Hip Inertia:      {rotation['hip_inertia_kg_m2']:.4f} kg·m²")
                print(f"   Shoulder Inertia: {rotation['shoulder_inertia_kg_m2']:.4f} kg·m² ({rotation['inertia_ratio']:.1f}× hip)")
                print(f"   Peak Hip Torque:      {rotation['peak_hip_torque_Nm']:.1f} N·m")
                print(f"   Peak Shoulder Torque: {rotation['peak_shoulder_torque_Nm']:.1f} N·m")
                print(f"   Peak Hip Power:       {rotation['peak_hip_power_W']:.0f} W ({rotation['hip_power_per_kg']:.1f} W/kg)")
                print(f"   Peak Shoulder Power:  {rotation['peak_shoulder_power_W']:.0f} W ({rotation['shoulder_power_per_kg']:.1f} W/kg)")
                print(f"   Max Separation:       {rotation['max_separation_deg']:.1f}°")
                print(f"   Sequence Timing:      {rotation['sequence_timing_ms']:.0f} ms")
                print(f"   Proper Sequence:      {'YES ✅' if rotation['proper_sequence'] else 'NO ❌'}")
            
            if stride:
                print(f"\n🦵 STRIDE (Event Detection):")
                print(f"   Plant Frame: {stride['plant_frame']} at t={stride['plant_time']:.2f}s ({stride['plant_method']})")
                print(f"   Stride Length:    {stride['stride_length_ft']:.2f} ft ({stride['stride_ratio']:.2f} × height)")
                print(f"   Stride Efficiency: {stride['stride_efficiency_pct']:.0f}%")
                
            if hand_speed:
                src = hand_speed.get('source', '')
                print(f"\n🦾 EST. HAND SPEED: {hand_speed['estimated_hand_speed_mph']:.1f} mph ({src})")

            if lower_body:
                print(f"\n🦵 LOWER-BODY KINEMATICS:")
                print(f"   Hip Flex R/L:   {lower_body.get('peak_hip_flex_r_deg',0):.1f}° / {lower_body.get('peak_hip_flex_l_deg',0):.1f}°  (asym {lower_body.get('hip_flex_asymmetry_deg',0):.1f}°)")
                print(f"   Knee Flex R/L:  {lower_body.get('peak_knee_r_deg',0):.1f}° / {lower_body.get('peak_knee_l_deg',0):.1f}°  (asym {lower_body.get('knee_asymmetry_deg',0):.1f}°)")
                print(f"   Ankle R/L:      {lower_body.get('peak_ankle_r_deg',0):.1f}° / {lower_body.get('peak_ankle_l_deg',0):.1f}°")
                print(f"   Knee Torque R/L:{lower_body.get('peak_knee_torque_r_Nm',0):.1f} / {lower_body.get('peak_knee_torque_l_Nm',0):.1f} N·m")
                print(f"   Knee Power  R/L:{lower_body.get('peak_knee_power_r_W',0):.0f} / {lower_body.get('peak_knee_power_l_W',0):.0f} W")

            if linear_id:
                print(f"\n⚡ LINEAR INVERSE DYNAMICS (Pelvis F=ma):")
                print(f"   AP Force:   {linear_id.get('peak_pelvis_force_ap_N',0):.0f} N")
                print(f"   Vert Force: {linear_id.get('peak_pelvis_force_vert_N',0):.0f} N")
                print(f"   Lat Force:  {linear_id.get('peak_pelvis_force_lat_N',0):.0f} N")
                print(f"   Resultant:  {linear_id.get('peak_pelvis_force_resultant_N',0):.0f} N ({linear_id.get('peak_pelvis_force_resultant_N',0)/self.body_mass_kg:.1f} N/kg)")

            if weight_shift:
                print(f"\n⚖️  WEIGHT SHIFT / LATERAL BALANCE:")
                print(f"   Pelvis Tilt Range:  {weight_shift.get('pelvis_tilt_range_deg',0):.1f}°")
                print(f"   Pelvis List Range:  {weight_shift.get('pelvis_list_range_deg',0):.1f}°")
                print(f"   Lateral Sway:       {weight_shift.get('lateral_sway_range_m',0)*100:.1f} cm")
                print(f"   Sway at Plant:      {weight_shift.get('lateral_sway_at_plant_m',0)*100:.1f} cm")
                print(f"   Shift Timing:       {weight_shift.get('weight_shift_timing_pct',0):.0f}% of swing")

            print(f"\n" + "="*70)
            print(f"OVERALL EFFICIENCY: {max(0, efficiency_score)}/100")
            print("="*70)
            for finding in findings:
                print(f"   {finding}")

        phase_report = self.build_phase_report(rotation, stride, trc_metrics, lower_body)

        # Score the capture itself, separately from the swing, so a check swing or
        # a mistracked trial can be held out of multi-swing averages with a reason.
        _dt = kinematics['time'].diff().mean() if 'time' in kinematics.columns else None
        capture_quality = assess_capture_quality(
            rotation, phase_report.get('dimensions', {}),
            has_markers=trc_data is not None, dt=_dt,
        )

        return {
            # Internal handle so the API can draw the kinematic-sequence chart from
            # the same segment velocities the metrics use. Contains numpy arrays —
            # callers MUST pop it before JSON serialisation.
            "_rotation": rotation,
            "capture_quality": capture_quality,
            "metrics": asdict(metrics),
            "findings": findings,
            "recommendations": recommendations,
            "efficiency_score": max(0, efficiency_score),
            "reference_values": SKILL_LEVEL_BENCHMARKS.get(self.skill_level, SKILL_LEVEL_BENCHMARKS['high_school']),
            "phase_report": phase_report,
            "swing_score": phase_report['swing_score'],
            "lower_body": {k: v for k, v in lower_body.items() if not isinstance(v, np.ndarray)},
            "linear_inverse_dynamics": {k: v for k, v in linear_id.items() if not isinstance(v, np.ndarray)},
            "weight_shift": weight_shift,
            "grf_estimation": grf_data,
            "data_quality": _build_data_quality(trc_metrics, has_grf=bool(grf_data)),
            # Per-metric evidence tier and whether THIS capture supports it. Kept
            # separate so a good citation can never mask data that cannot carry
            # the number — see backend/metric_evidence.py.
            "metric_evidence": _metric_evidence_block(
                asdict(metrics),
                {'separation_signal_saturated': rotation.get('separation_signal_saturated') if rotation else False,
                 'separation_source': rotation.get('separation_source', 'model_lumbar_rotation') if rotation else 'model_lumbar_rotation',
                 'separation_raw_peak_deg': rotation.get('separation_raw_peak_deg', 0.0) if rotation else 0.0}),
            # Equipment / session context
            **({"instrument": self.instrument} if self.instrument else {}),
            **({"instrument_note": self.instrument_note} if self.instrument_note else {}),
        }

    def _rate_dimension(self, key: str, value: float, invert: bool = False) -> int:
        """Rate a single dimension 1-5 based on per-skill-level threshold table.
        For 'invert=True' dimensions (like direction-at-contact), lower value is better.
        Returns 1-5."""
        thresholds = DIMENSION_THRESHOLDS.get(key, {}).get(self.skill_level, [])
        if not thresholds:
            return 3  # fallback
        
        if invert:
            # Iterate from highest threshold downward — smaller is better
            for threshold, stars in sorted(thresholds, key=lambda x: x[0], reverse=True):
                if value <= threshold:
                    return stars
            return 1
        else:
            rating = 1
            for threshold, stars in sorted(thresholds, key=lambda x: x[0]):
                if value >= threshold:
                    rating = stars
            return rating

    def _rating_to_badge(self, stars: int) -> str:
        """Convert a 1-5 star rating to a color badge label."""
        if stars >= 5:
            return 'excellent'
        elif stars >= 3:
            return 'satisfactory'
        else:
            return 'off_target'

    def calculate_lead_leg_block(self, lower_body: Dict, rotation: Dict, stride: Dict) -> Dict:
        """Lead-leg block: how much the front knee EXTENDS (straightens) from front-foot
        plant to contact. A firm, extending lead leg posts up and redirects momentum into
        rotation — a top bat-speed correlate in the Driveline OpenBiomechanics dataset.

        The lead leg is taken from the athlete's stated handedness when known — a
        right-handed hitter strides onto the left leg, and vice versa. Only when
        handedness is unknown does it fall back to inferring the lead leg as the one
        less flexed at contact, which misfires on check swings and short captures.
        Returns {} if knee arrays are unavailable.
        """
        knee_r = lower_body.get('knee_r') if lower_body else None
        knee_l = lower_body.get('knee_l') if lower_body else None
        if knee_r is None or knee_l is None or rotation is None:
            return {}
        # Work in flexion magnitude so the result is independent of the model's sign convention.
        fr = np.abs(np.asarray(knee_r, dtype=float))
        fl = np.abs(np.asarray(knee_l, dtype=float))
        n = min(len(fr), len(fl))
        if n < 3:
            return {}

        plant = stride.get('plant_frame', n // 3) if stride else n // 3
        plant = int(min(max(plant, 0), n - 2))

        # Contact proxy = peak pelvis angular velocity (consistent with the rest of the report).
        pom = rotation.get('pelvis_omega')
        if pom is not None and len(pom) >= n:
            contact = int(np.argmax(np.abs(np.asarray(pom[:n]))))
        else:
            contact = min(n - 1, plant + max(1, (n - plant) // 2))
        if contact <= plant:
            contact = min(n - 1, plant + max(1, (n - plant) // 2))

        # A right-handed hitter strides onto the left leg; a lefty onto the right.
        if self.handedness == 'right':
            lead, lead_source = 'l', 'handedness'
        elif self.handedness == 'left':
            lead, lead_source = 'r', 'handedness'
        else:
            # Fallback: the straighter (less flexed) leg at contact.
            lead = 'l' if fl[contact] < fr[contact] else 'r'
            lead_source = 'inferred_from_pose'
        lead_flex = fl if lead == 'l' else fr

        # Block = how much the lead knee STRAIGHTENS from its deepest flexion in the
        # plant→contact window into contact. Measuring from peak flexion (not raw plant)
        # keeps the metric robust when plant is detected slightly early (still loading)
        # and guarantees the "extension" is non-negative, as it physically should be.
        window = lead_flex[plant:contact + 1]
        peak_flex = float(np.max(window)) if len(window) else float(lead_flex[plant])
        contact_flex = float(lead_flex[contact])
        ext = max(0.0, peak_flex - contact_flex)
        return {
            'lead_leg_block_deg': round(ext, 1),
            'lead_side': lead,
            'lead_side_source': lead_source,
            'handedness': self.handedness,
            'lead_knee_peak_flex_deg': round(peak_flex, 1),
            'lead_knee_flex_at_contact_deg': round(contact_flex, 1),
        }

    def _empirical_percentile(self, key: str, value: float) -> Tuple[Optional[float], Optional[int]]:
        """Percentile of `value` against the user's own library of swings at this
        level (built by build_cohort.py). Returns (percentile, n) or (None, None)
        when no sufficiently-populated cohort exists for this level/dimension."""
        model = get_cohort_model()
        if not model:
            return (None, None)
        entry = (model.get(self.skill_level) or {}).get(key)
        if not entry:
            return (None, None)
        vals = entry.get('values') or []
        n = int(entry.get('n', len(vals)))
        if n < COHORT_MIN_N or not vals:
            return (None, None)
        import bisect
        # Rank in "goodness" space so lower-is-better dimensions rank correctly.
        invert = key in INVERT_DIMS
        g = sorted(-v for v in vals) if invert else sorted(vals)
        x = -float(value) if invert else float(value)
        lo = bisect.bisect_left(g, x)
        hi = bisect.bisect_right(g, x)
        rank = (lo + hi) / 2.0
        pct = max(1.0, min(99.0, round(rank / len(g) * 100.0, 0)))
        return (pct, n)

    def _percentile_for(self, key: str, value: float) -> Optional[float]:
        """Translate a raw dimension value into an approximate cohort percentile (1-99)
        by interpolating the value across the level's 5 star-boundary thresholds."""
        thr = DIMENSION_THRESHOLDS.get(key, {}).get(self.skill_level, [])
        if not thr:
            return None
        vals = [t[0] for t in sorted(thr, key=lambda x: x[1])]  # thresholds in star order 1..5
        anchors = list(PERCENTILE_ANCHORS)
        if key in INVERT_DIMS:
            # Lower value is better → flip axis so "more good" maps to higher percentile.
            pts = sorted([(-vals[i], anchors[i]) for i in range(len(vals))], key=lambda p: p[0])
            x = -float(value)
        else:
            pts = sorted([(vals[i], anchors[i]) for i in range(len(vals))], key=lambda p: p[0])
            x = float(value)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if x <= xs[0]:
            if xs[1] > xs[0]:
                frac = (x - xs[0]) / (xs[1] - xs[0])
                return round(max(1.0, ys[0] + frac * (ys[1] - ys[0])), 0)
            return ys[0]
        if x >= xs[-1]:
            if xs[-1] > xs[-2]:
                frac = (x - xs[-1]) / (xs[-1] - xs[-2])
                return round(min(99.0, ys[-1] + frac * (ys[-1] - ys[-2])), 0)
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1] and xs[i + 1] > xs[i]:
                frac = (x - xs[i]) / (xs[i + 1] - xs[i])
                return round(ys[i] + frac * (ys[i + 1] - ys[i]), 0)
        return 50.0

    def build_phase_report(self, rotation: Dict, stride: Dict, trc_metrics: Dict, lower_body: Dict = None) -> Dict:
        """Build a 12-dimension report from computed physics outputs.
        Returns a structured dict with phases, dimension detail, and an aggregate Swing Score."""
        dims = {}

        # ------------------------------------------------------------------
        # PHASE 1: BALANCE & LOAD
        # ------------------------------------------------------------------
        # Negative Move: peak backward shift of pelvis before stride
        # We derive from rotation['pelvis_angle'] pre-plant: use a proxy —
        # the change in pelvis position. If we have stride data, use
        # stride_length as forward move proxy and approximate negative move
        # from a fixed ratio typical of the sport. Without pelvis_tx timeseries
        # here, we use a heuristic from stride ratio.
        if stride:
            # Negative move correlates strongly with stride initiation quality.
            # Use stride_ratio as primary signal: good stride implies good load back.
            neg_move_proxy = stride['stride_length_m'] * 0.15  # ~15% of stride is backward
            neg_move_stars = self._rate_dimension('negative_move', neg_move_proxy)
        else:
            neg_move_proxy = 0.0
            neg_move_stars = 2

        dims['negative_move'] = {
            'label': DIMENSION_LABELS['negative_move'],
            'stars': neg_move_stars,
            'badge': self._rating_to_badge(neg_move_stars),
            'value': round(neg_move_proxy, 3),
            'unit': 'm',
            'description': 'Initial weight shift rearward to load energy before the stride.',
        }

        # Pelvis Load: Pelvis KE during the load phase
        pelvis_ke = rotation.get('pelvis_ke_J', 0.0) if rotation else 0.0
        # Pelvis Load and Upper Torso Load: KE at swing onset (load phase),
        # not peak swing KE. Peak KE reflects swing speed, not load quality.
        # Use pelvis_omega and lumbar_omega at swing_start_frame as the load proxy.
        if rotation and 'pelvis_angle' in rotation:
            sw = rotation.get('swing_start_frame', 0)
            # Recompute KE at onset using the stored omega arrays
            # pelvis_ke_load = 0.5 * I * omega_at_onset²
            # We use peak_pelvis_omega scaled by the fraction of swing completed at onset
            # Simpler: use the stored pelvis_ke but scale by (onset_omega/peak_omega)²
            # Best: use hip_power_per_kg as the load proxy (already swing-window corrected)
            pelvis_ke_load = rotation.get('pelvis_ke_J', 0.0)
            torso_ke_load  = rotation.get('torso_ke_J', 0.0)
        else:
            pelvis_ke_load = 0.0
            torso_ke_load  = 0.0

        pl_stars = self._rate_dimension('pelvis_load', pelvis_ke_load)
        dims['pelvis_load'] = {
            'label': DIMENSION_LABELS['pelvis_load'],
            'stars': pl_stars,
            'badge': self._rating_to_badge(pl_stars),
            'value': round(pelvis_ke_load, 1),
            'unit': 'J',
            'description': 'Hip rotational energy during the swing (proxy for hip coil power).',
        }

        # Upper Torso Load: the omega cap (2× pelvis) was applied upstream in torque_refined,
        # so torso_ke_J is already artifact-cleaned. Apply a matching 4× pelvis_ke guard here
        # (equivalent to 2× omega) as a belt-and-suspenders check against any residual artifacts.
        torso_ke_capped = min(torso_ke_load, pelvis_ke_load * 4.0) if pelvis_ke_load > 0 else torso_ke_load
        utl_stars = self._rate_dimension('upper_torso_load', torso_ke_capped)
        dims['upper_torso_load'] = {
            'label': DIMENSION_LABELS['upper_torso_load'],
            'stars': utl_stars,
            'badge': self._rating_to_badge(utl_stars),
            'value': round(torso_ke_capped, 1),
            'unit': 'J',
            'description': 'Trunk rotational energy during the swing (proxy for shoulder coil power).',
        }

        # ------------------------------------------------------------------
        # PHASE 2: STRIDE
        # ------------------------------------------------------------------
        # Stride length and forward move both depend on actually seeing the stride.
        # When the capture doesn't contain one, report them as not measured rather
        # than scoring a confident "off target" for something we cannot observe.
        stride_ok = bool(stride and stride.get('stride_detected'))
        stride_note = (stride or {}).get('stride_reason') or 'stride not measurable in this capture'

        stride_ratio = stride['stride_ratio'] if stride else 0.0
        sl_stars = self._rate_dimension('stride_length', stride_ratio)
        dims['stride_length'] = {
            'label': DIMENSION_LABELS['stride_length'],
            'stars': sl_stars if stride_ok else 0,
            'badge': self._rating_to_badge(sl_stars) if stride_ok else 'unavailable',
            'value': round(stride_ratio, 2) if stride_ok else None,
            'unit': '× height',
            'description': 'Forward step distance relative to body height. Elite target ~75-90% of height.',
            'available': stride_ok,
            **({} if stride_ok else {'unavailable_reason': stride_note}),
        }

        stride_eff = stride['stride_efficiency_pct'] if stride else 0.0
        # Penalize over-striding (>115%) as well as under-striding
        fm_val = min(stride_eff, 115.0) if stride_eff <= 115.0 else max(0, 115.0 - (stride_eff - 115.0))
        fm_stars = self._rate_dimension('forward_move', fm_val)
        dims['forward_move'] = {
            'label': DIMENSION_LABELS['forward_move'],
            'stars': fm_stars if stride_ok else 0,
            'badge': self._rating_to_badge(fm_stars) if stride_ok else 'unavailable',
            'value': round(stride_eff, 1) if stride_ok else None,
            'unit': '%',
            'description': 'Controlled forward momentum of the stride, stopping at front foot plant.',
            'available': stride_ok,
            **({} if stride_ok else {'unavailable_reason': stride_note}),
        }

        # ------------------------------------------------------------------
        # PHASE 3: POWER MOVE
        # ------------------------------------------------------------------
        sep_deg = rotation.get('max_separation_deg', 0.0) if rotation else 0.0
        # Penalty for hyper-extension (>65°) — same symmetrical penalty as under-separation
        if sep_deg > 65:
            sep_score_val = max(0, 65.0 - (sep_deg - 65.0))
        else:
            sep_score_val = sep_deg
        mhs_stars = self._rate_dimension('max_hip_shoulder_separation', sep_score_val)
        dims['max_hip_shoulder_separation'] = {
            'label': DIMENSION_LABELS['max_hip_shoulder_separation'],
            'stars': mhs_stars,
            'badge': self._rating_to_badge(mhs_stars),
            'value': round(sep_deg, 1),
            'unit': '°',
            'description': 'Maximum angle between hips and shoulders. Stores elastic energy (X-Factor).',
        }

        # Pelvis Total Rotation Range — from swing onset to peak pelvis omega only.
        # Excludes follow-through which inflates the range to 200°+.
        if rotation and 'pelvis_angle' in rotation:
            pelvis_ang = rotation['pelvis_angle']
            sw  = rotation.get('swing_start_frame', 0)
            pk  = int(np.argmax(np.abs(rotation.get('pelvis_omega', np.zeros(len(pelvis_ang))))))
            pelvis_ang_window = pelvis_ang[sw:pk + 1] if pk > sw else pelvis_ang[sw:]
            pelvis_rot_range = float(np.abs(np.max(pelvis_ang_window) - np.min(pelvis_ang_window)) * 180.0 / np.pi)
        else:
            pelvis_rot_range = 0.0
        prr_stars = self._rate_dimension('pelvis_rotation_range', pelvis_rot_range)
        dims['pelvis_rotation_range'] = {
            'label': DIMENSION_LABELS['pelvis_rotation_range'],
            'stars': prr_stars,
            'badge': self._rating_to_badge(prr_stars),
            'value': round(pelvis_rot_range, 1),
            'unit': '°',
            'description': 'Total hip rotation from load through contact.',
        }

        # Upper Torso Total Rotation Range — use lumbar_angle range within swing window.
        # This is the actual trunk twist (shoulder relative to pelvis), which is what
        # coaches mean by "shoulder rotation range". Capped at 120° (physically plausible max).
        if rotation and 'pelvis_angle' in rotation:
            sw = rotation.get('swing_start_frame', 0)
            # lumbar_omega_sw gives us the relative twist rate; integrate to get range
            # Simpler: use pelvis_rotation_range as base and add lumbar contribution
            # from the X-Factor (max separation already computed from lumbar_angle)
            sep = dims['max_hip_shoulder_separation']['value']  # lumbar range in swing
            torso_rot_range = min(pelvis_rot_range + sep, 120.0)
        else:
            torso_rot_range = 0.0
        utrr_stars = self._rate_dimension('upper_torso_rotation_range', torso_rot_range)
        dims['upper_torso_rotation_range'] = {
            'label': DIMENSION_LABELS['upper_torso_rotation_range'],
            'stars': utrr_stars,
            'badge': self._rating_to_badge(utrr_stars),
            'value': round(torso_rot_range, 1),
            'unit': '°',
            'description': 'Total shoulder rotation from load through contact.',
        }

        # ------------------------------------------------------------------
        # PHASE 4: CONTACT & FOLLOW-THROUGH
        # ------------------------------------------------------------------
        # Pelvis Direction at Contact — how close to 90° (square) at plant frame
        if rotation and 'pelvis_angle' in rotation and stride:
            plant_idx = min(stride['plant_frame'], len(rotation['pelvis_angle']) - 1)
            pelvis_at_contact_deg = float(np.abs(rotation['pelvis_angle'][plant_idx]) * 180.0 / np.pi)
            # Deviation from "square" — 90° is ideal so deviation = |90 - angle|
            pelvis_dev = abs(90.0 - pelvis_at_contact_deg)
        else:
            pelvis_dev = 45.0
        pdc_stars = self._rate_dimension('pelvis_direction_at_contact', pelvis_dev, invert=True)
        dims['pelvis_direction_at_contact'] = {
            'label': DIMENSION_LABELS['pelvis_direction_at_contact'],
            'stars': pdc_stars,
            'badge': self._rating_to_badge(pdc_stars),
            'value': round(pelvis_dev, 1),
            'unit': '° off-square',
            'description': 'Hip alignment at contact. Hips should be square (90°) to the pitcher.',
        }

        # Upper Torso Direction at Contact
        if rotation:
            torso_dev = pelvis_dev * (1.0 / max(0.5, rotation.get('torso_to_pelvis_rot_ratio', 1.0)))
        else:
            torso_dev = 50.0
        utdc_stars = self._rate_dimension('upper_torso_direction_at_contact', float(torso_dev), invert=True)
        dims['upper_torso_direction_at_contact'] = {
            'label': DIMENSION_LABELS['upper_torso_direction_at_contact'],
            'stars': utdc_stars,
            'badge': self._rating_to_badge(utdc_stars),
            'value': round(torso_dev, 1),
            'unit': '° off-square',
            'description': 'Shoulder alignment at contact for optimal barrel control and plate coverage.',
        }

        # Lead-Leg Block — front-knee extension from plant to contact (OBP bat-speed correlate)
        llb = self.calculate_lead_leg_block(lower_body or {}, rotation, stride)
        llb_val = llb.get('lead_leg_block_deg', 0.0)
        llb_stars = self._rate_dimension('lead_leg_block', llb_val)
        dims['lead_leg_block'] = {
            'label': DIMENSION_LABELS['lead_leg_block'],
            'stars': llb_stars,
            'badge': self._rating_to_badge(llb_stars),
            'value': round(llb_val, 1),
            'unit': '° ext',
            'description': 'Lead (front) knee straightening from foot plant to contact. A firm, '
                           'extending front leg posts up and whips energy into the barrel (Driveline OBP).',
        }

        # Kinetic Chain Efficiency
        chain_eff = rotation.get('energy_transfer_proxy_pct', 0.0) if rotation else 0.0
        kce_stars = self._rate_dimension('kinetic_chain_efficiency', chain_eff)
        dims['kinetic_chain_efficiency'] = {
            'label': DIMENSION_LABELS['kinetic_chain_efficiency'],
            'stars': kce_stars,
            'badge': self._rating_to_badge(kce_stars),
            'value': round(chain_eff, 1),
            'unit': '%',
            'description': 'Percentage of total body energy that reaches the hands/bat.',
        }

        # Sequence Quality — computed directly (not threshold lookup)
        if rotation:
            proper = rotation.get('proper_sequence', False)
            indeterminate = rotation.get('sequence_indeterminate', False)
            timing_ms = rotation.get('sequence_timing_ms', 0.0)
            benchmarks = SKILL_LEVEL_BENCHMARKS.get(self.skill_level, {})
            t_lo, t_hi = benchmarks.get('sequence_timing_ms', (20, 60))
            if indeterminate:
                # Cannot determine which segment peaks first at 60Hz — honest 3/5.
                # Don't penalize what the capture resolution can't measure.
                sq_stars = 3
            elif proper and t_lo <= timing_ms <= t_hi:
                sq_stars = 5
            elif proper and timing_ms > 0:
                sq_stars = 4 if abs(timing_ms - (t_lo + t_hi) / 2) < 15 else 3
            elif proper:
                sq_stars = 3
            else:
                sq_stars = 1
        else:
            sq_stars = 2
        dims['sequence_quality'] = {
            'label': DIMENSION_LABELS['sequence_quality'],
            'stars': sq_stars,
            'badge': self._rating_to_badge(sq_stars),
            'value': round(rotation.get('sequence_timing_ms', 0.0) if rotation else 0.0, 1),
            'unit': 'ms lag',
            'description': 'Proximal-to-distal sequencing: Pelvis → Torso → Arms in correct order and timing.',
            **(({'indeterminate': True,
                 'indeterminate_reason': 'Timing difference is within 1-frame resolution (≤17ms at 60Hz). Sequence direction cannot be determined from this capture rate.'}
                if rotation and rotation.get('sequence_indeterminate') else {})),
        }

        # Hand / Bat Speed — most reliable output metric
        # Priority: TRC wrist markers > angular velocity estimate
        _trc_hs = trc_metrics.get('max_hand_speed_mph', 0.0) if trc_metrics else 0.0
        _trc_source = bool(_trc_hs > 0)
        hand_spd = _trc_hs
        if hand_spd == 0.0:
            hs_result = self.estimate_hand_speed(rotation, trc_metrics or {})
            hand_spd = hs_result.get('estimated_hand_speed_mph', 0.0) if hs_result else 0.0
        hs_stars = self._rate_dimension('hand_speed', hand_spd)
        dims['hand_speed'] = {
            'label': DIMENSION_LABELS['hand_speed'],
            'stars': hs_stars,
            'badge': self._rating_to_badge(hs_stars),
            'value': round(hand_spd, 1),
            'unit': 'mph',
            'description': 'Peak hand/wrist speed — primary output metric of the kinetic chain.',
            'source': 'trc_marker' if _trc_source else 'angular_velocity_estimate',
            # Monocular captures have high geometric sensitivity for hand speed: accuracy
            # degrades when the wrist trajectory has a large depth component relative to
            # the camera. Within-session trends are valid; cross-session comparisons are not.
            **(({'monocular_caveat': True,
                 'monocular_note': 'Monocular wrist speed: valid within this session, but sensitive to camera angle. Cross-session and cross-athlete comparisons unreliable.'}
                if _trc_source else {})),
        }

        # Follow-Through Quality: pelvis deceleration arc after contact (peak pelvis omega)
        # Measured as degrees of continued pelvis rotation from peak omega to end of trial.
        # A complete follow-through = 30-65° of continued rotation post-contact.
        ft_range = 0.0
        if rotation and 'pelvis_angle' in rotation and 'pelvis_omega' in rotation:
            pelvis_ang = rotation['pelvis_angle']
            pelvis_om  = rotation['pelvis_omega']
            pk = int(np.argmax(np.abs(pelvis_om)))
            if pk < len(pelvis_ang) - 1:
                # Rotation from peak to the point where pelvis reverses or trial ends
                post_peak = pelvis_ang[pk:]
                # Find where pelvis stops rotating in the same direction (sign change)
                peak_sign = np.sign(pelvis_om[pk])
                end_idx = len(post_peak) - 1
                for j in range(1, len(post_peak)):
                    if np.sign(pelvis_om[pk + j] if pk + j < len(pelvis_om) else 0) != peak_sign:
                        end_idx = j
                        break
                ft_range = float(np.abs(post_peak[end_idx] - post_peak[0]) * 180.0 / np.pi)
        ft_stars = self._rate_dimension('follow_through_quality', ft_range)
        dims['follow_through_quality'] = {
            'label': DIMENSION_LABELS['follow_through_quality'],
            'stars': ft_stars,
            'badge': self._rating_to_badge(ft_stars),
            'value': round(ft_range, 1),
            'unit': '°',
            'description': 'Pelvis deceleration arc after contact. Complete follow-through = 30-65° continued rotation.',
        }

        # ------------------------------------------------------------------
        # SWING SCORE (0-100): Weighted average of dimension star ratings
        # Stars range 1-5. Normalise to 0-100 as (stars-1)/4 * 100, then weight.
        # ------------------------------------------------------------------
        # Star-boundary percentile fallback for dimensions without a threshold table.
        _STAR_PCT = {1: 8.0, 2: 30.0, 3: 50.0, 4: 72.0, 5: 92.0}

        total_weight = 0.0
        weighted_sum = 0.0
        pct_weight = 0.0
        pct_weighted_sum = 0.0
        n_empirical = 0
        for dim_key, weight in DIMENSION_WEIGHTS.items():
            # A dimension the capture couldn't measure carries no weight — scoring
            # it would penalise the athlete for a limitation of the data.
            if dims[dim_key].get('available') is False:
                continue
            stars = dims[dim_key]['stars']
            normalized = ((stars - 1) / 4.0) * 100.0
            weighted_sum += normalized * weight
            total_weight += weight

            val = dims[dim_key]['value']
            # Research-guided benchmark percentile — always computed; it is the prior.
            bench_pct = self._percentile_for(dim_key, val)
            if bench_pct is None:
                bench_pct = _STAR_PCT.get(stars, 50.0)
            dims[dim_key]['percentile_benchmark'] = int(round(bench_pct))

            # Empirical percentile from the user's own level cohort, if populated.
            emp_pct, emp_n = self._empirical_percentile(dim_key, val)
            if emp_pct is not None:
                # Empirical-Bayes shrinkage toward the research benchmark: cohort weight
                # grows with n, so research still anchors small cohorts (never abandoned).
                w = emp_n / (emp_n + COHORT_SHRINKAGE)
                pct = w * emp_pct + (1.0 - w) * bench_pct
                dims[dim_key]['percentile_n'] = int(emp_n)
                dims[dim_key]['percentile_library'] = int(round(emp_pct))
                dims[dim_key]['percentile_weight'] = round(w, 2)
                n_empirical += 1
            else:
                pct = bench_pct

            dims[dim_key]['percentile'] = int(round(pct))
            pct_weighted_sum += pct * weight
            pct_weight += weight

        swing_score = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0
        overall_percentile = int(round(pct_weighted_sum / pct_weight)) if pct_weight > 0 else 50
        percentile_basis = 'blended' if n_empirical > 0 else 'benchmark'

        # ------------------------------------------------------------------
        # PRESCRIPTION ENGINE — rank weak dimensions by bat-speed impact and
        # map each to a specific cue + drill (Driveline-style constraint coaching).
        # impact = dimension weight × star deficit (how far below a 5-star it is).
        # ------------------------------------------------------------------
        prescriptions = []
        for dim_key, weight in DIMENSION_WEIGHTS.items():
            if dims[dim_key].get('available') is False:
                continue  # never prescribe a drill for something we didn't measure
            stars = dims[dim_key]['stars']
            if stars >= 4:
                continue  # only prescribe for below-target dimensions
            drill = DRILL_LIBRARY.get(dim_key)
            if not drill:
                continue
            prescriptions.append({
                'key': dim_key,
                'label': DIMENSION_LABELS.get(dim_key, dim_key),
                'stars': stars,
                'percentile': dims[dim_key].get('percentile', 50),
                'impact': round(weight * (5 - stars), 4),
                'cue': drill['cue'],
                'drill': drill['drill'],
                'why': drill['why'],
            })
        prescriptions.sort(key=lambda p: p['impact'], reverse=True)
        for i, p in enumerate(prescriptions):
            p['priority'] = i + 1

        # ------------------------------------------------------------------
        # Assemble phases
        # ------------------------------------------------------------------
        phases = {}
        for phase_key, phase_meta in SWING_PHASES.items():
            phase_dims = [{**dims[d], 'key': d} for d in phase_meta['dimensions'] if d in dims]
            rated = [d for d in phase_dims if d.get('available') is not False]
            phase_avg_stars = sum(d['stars'] for d in rated) / max(1, len(rated))
            phases[phase_key] = {
                'label': phase_meta['label'],
                'icon': phase_meta['icon'],
                'avg_stars': round(phase_avg_stars, 1),
                'badge': self._rating_to_badge(round(phase_avg_stars)),
                'dimensions': phase_dims,
            }

        return {
            'swing_score': swing_score,
            'overall_percentile': overall_percentile,
            'percentile_basis': percentile_basis,      # 'library' (your own swings) or 'benchmark' (research)
            'percentile_library_dims': n_empirical,     # how many of the 15 used your library
            'skill_level': self.skill_level,
            'phases': phases,
            'dimensions': dims,
            'prescriptions': prescriptions,
            'lead_leg_block': llb,
        }

def find_mot_files() -> List[str]:
    """Auto-find .mot files"""
    current_dir = os.getcwd()
    local_files = glob.glob("*.mot")
    downloads_path = os.path.expanduser("~/Downloads")
    downloads_files = glob.glob(os.path.join(downloads_path, "*.mot"))
    all_files = local_files + downloads_files
    # Only pick those labeled as swing (optional: modify if files aren't labeled 'swing')
    swing_files = list(set([f for f in all_files if f.endswith('.mot')])) 
    return swing_files

def main():
    print("="*70)
    print("REFINED HITTING OPTIMIZATION SYSTEM")
    print("With Critical Biomechanics Refinements (OpenCap Optimized)")
    print("="*70)
    print("\nREFINEMENTS:")
    print("  ✅ FIX #1: Shoulder inertia = trunk + 2×arms + bat")
    print("  ✅ FIX #2: Savitzky-Golay filter + Butterworth Low-Pass filter for markerless noise")
    print("  ✅ FIX #3: Angle Unwrapping to prevent +180/-180 flips")
    print("  ✅ FIX #4: Dynamic event thresholds and window sizes based on framerate")
    print("="*70)
    
    swing_files = find_mot_files()
    if not swing_files:
        print("\n❌ No swing .mot files found in Downloads or current directory")
        return
        
    print(f"\n✅ Found {len(swing_files)} .mot files")
    
    body_mass_kg = 82
    body_height_m = 1.83
    
    optimizer = RefinedHittingOptimizer(body_mass_kg, body_height_m)
    
    all_metrics = []
    for filepath in swing_files:
        filename = os.path.basename(filepath)
        kinematics = optimizer.load_mot_file(filepath)
        if kinematics is None or len(kinematics) == 0:
            print(f"⚠️ Empty or invalid data for {filename}")
            continue
            
        trc_filepath = filepath.replace('Kinematics', 'MarkerData').replace('.mot', '.trc')
        trc_data = None
        if os.path.exists(trc_filepath):
            trc_data = optimizer.load_trc_file(trc_filepath)
            if trc_data is not None:
                print(f"✅ Loaded matching TRC file: {os.path.basename(trc_filepath)}")
            
        diagnosis = optimizer.comprehensive_diagnosis(kinematics, filename, trc_data=trc_data, verbose=True)
        metrics = diagnosis['metrics']
        
        all_metrics.append({
            'file': filename,
            'score': diagnosis['efficiency_score'],
            'inertia_ratio': metrics['inertia_ratio'],
            'hip_power_W_kg': metrics['hip_power_per_kg'],
            'plant_method': metrics['plant_method'],
            'hand_speed_mph': metrics['estimated_hand_speed_mph']
        })
        
    if len(all_metrics) > 1:
        df = pd.DataFrame(all_metrics)
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        df.to_csv('refined_swing_comparison.csv', index=False)
        print("\n✅ Saved: refined_swing_comparison.csv")
        
    print("\n" + "="*70)
    print("✅ REFINED ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
