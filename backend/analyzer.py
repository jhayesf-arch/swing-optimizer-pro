import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List

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
# SWINGAI-INSPIRED 12-DIMENSION THRESHOLD TABLE
# Maps each of SwingAI's 4 phases / 12 dimensions onto our
# computed physics metrics, with per-skill-level corridors
# that yield a 1-5 star rating (5 = elite / Excellent,
# 3-4 = Satisfactory, 1-2 = Off-Target).
# ============================================================
SWINGAI_THRESHOLDS = {
    # PHASE 1 — BALANCE & LOAD
    'negative_move': {
        # How far the pelvis shifts BACKWARD before stride (m)
        # Derived from pelvis_tx: min value before plant vs starting position
        'youth':        [(-0.02,1),(-0.01,2),(0.01,3),(0.03,4),(0.05,5)],
        'high_school':  [(-0.02,1),(-0.01,2),(0.02,3),(0.04,4),(0.06,5)],
        'college':      [(-0.02,1),(0.00,2),(0.03,3),(0.05,4),(0.08,5)],
        'professional': [(-0.02,1),(0.00,2),(0.03,3),(0.06,4),(0.10,5)],
    },
    'pelvis_load': {
        # Pelvis KE during load phase (J) — proxy for hip coil
        'youth':        [(0,1),(5,2),(15,3),(30,4),(50,5)],
        'high_school':  [(0,1),(10,2),(25,3),(50,4),(80,5)],
        'college':      [(0,1),(15,2),(40,3),(80,4),(120,5)],
        'professional': [(0,1),(25,2),(60,3),(110,4),(160,5)],
    },
    'upper_torso_load': {
        # Torso KE during load (J) — shoulder coil tension
        'youth':        [(0,1),(5,2),(12,3),(25,4),(40,5)],
        'high_school':  [(0,1),(8,2),(18,3),(40,4),(65,5)],
        'college':      [(0,1),(12,2),(28,3),(60,4),(95,5)],
        'professional': [(0,1),(18,2),(40,3),(85,4),(130,5)],
    },
    # PHASE 2 — STRIDE
    'stride_length': {
        # stride_ratio (stride / height). Elite = ~0.7-0.9
        'youth':        [(0.0,1),(0.3,2),(0.5,3),(0.65,4),(0.80,5)],
        'high_school':  [(0.0,1),(0.35,2),(0.55,3),(0.70,4),(0.85,5)],
        'college':      [(0.0,1),(0.40,2),(0.60,3),(0.75,4),(0.90,5)],
        'professional': [(0.0,1),(0.45,2),(0.65,3),(0.78,4),(0.92,5)],
    },
    'forward_move': {
        # stride_efficiency_pct. Target 75-110%.
        'youth':        [(0,1),(40,2),(65,3),(90,4),(115,5)],
        'high_school':  [(0,1),(45,2),(70,3),(95,4),(115,5)],
        'college':      [(0,1),(50,2),(75,3),(98,4),(115,5)],
        'professional': [(0,1),(50,2),(75,3),(100,4),(115,5)],
    },
    # PHASE 3 — POWER MOVE
    'max_hip_shoulder_separation': {
        # max_separation_deg. Elite = 40-55 degrees.
        'youth':        [(0,1),(15,2),(28,3),(38,4),(48,5)],
        'high_school':  [(0,1),(18,2),(30,3),(40,4),(52,5)],
        'college':      [(0,1),(22,2),(33,3),(42,4),(54,5)],
        'professional': [(0,1),(25,2),(35,3),(44,4),(56,5)],
    },
    'pelvis_rotation_range': {
        # Total pelvis rotation from load to contact (degrees).
        # Derived from pelvis_angle range (rad->deg).
        'youth':        [(0,1),(20,2),(35,3),(50,4),(65,5)],
        'high_school':  [(0,1),(25,2),(40,3),(55,4),(70,5)],
        'college':      [(0,1),(30,2),(45,3),(60,4),(75,5)],
        'professional': [(0,1),(35,2),(50,3),(65,4),(80,5)],
    },
    'upper_torso_rotation_range': {
        # Total shoulder rotation from load to contact (degrees).
        'youth':        [(0,1),(30,2),(50,3),(70,4),(90,5)],
        'high_school':  [(0,1),(35,2),(55,3),(80,4),(100,5)],
        'college':      [(0,1),(40,2),(65,3),(90,4),(110,5)],
        'professional': [(0,1),(45,2),(70,3),(95,4),(115,5)],
    },
    # PHASE 4 — CONTACT & FOLLOW-THROUGH
    'pelvis_direction_at_contact': {
        # Absolute deviation of pelvis from 90° (square to pitcher) at plant frame.
        # Lower is better. Score 1-5 inversely.
        'youth':        [(90,1),(60,2),(40,3),(20,4),(8,5)],
        'high_school':  [(90,1),(55,2),(35,3),(18,4),(6,5)],
        'college':      [(90,1),(50,2),(30,3),(15,4),(5,5)],
        'professional': [(90,1),(45,2),(25,3),(12,4),(4,5)],
    },
    'upper_torso_direction_at_contact': {
        # Same but for torso angle at contact.
        'youth':        [(90,1),(65,2),(45,3),(25,4),(10,5)],
        'high_school':  [(90,1),(60,2),(40,3),(22,4),(8,5)],
        'college':      [(90,1),(55,2),(35,3),(18,4),(6,5)],
        'professional': [(90,1),(50,2),(30,3),(15,4),(5,5)],
    },
    'kinetic_chain_efficiency': {
        # kinetic_chain_efficiency_pct. Higher = better distal energy amplification.
        'youth':        [(0,1),(8,2),(18,3),(30,4),(42,5)],
        'high_school':  [(0,1),(10,2),(22,3),(35,4),(48,5)],
        'college':      [(0,1),(12,2),(25,3),(40,4),(55,5)],
        'professional': [(0,1),(15,2),(28,3),(44,4),(60,5)],
    },
    'sequence_quality': {
        # Combined rating of proper_sequence + sequence_timing_ms
        # (computed directly, not threshold lookup)
        'youth':        [],
        'high_school':  [],
        'college':      [],
        'professional': [],
    },
}

# Dimension weights for Swing Score (must sum to 1.0)
SWINGAI_WEIGHTS = {
    'negative_move': 0.06,
    'pelvis_load': 0.08,
    'upper_torso_load': 0.06,
    'stride_length': 0.07,
    'forward_move': 0.07,
    'max_hip_shoulder_separation': 0.14,
    'pelvis_rotation_range': 0.08,
    'upper_torso_rotation_range': 0.08,
    'pelvis_direction_at_contact': 0.08,
    'upper_torso_direction_at_contact': 0.08,
    'kinetic_chain_efficiency': 0.10,
    'sequence_quality': 0.10,
}

SWINGAI_LABELS = {
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
    'kinetic_chain_efficiency': 'Kinetic Chain Efficiency',
    'sequence_quality': 'Sequence Quality',
}

SWINGAI_PHASES = {
    'balance_load': {
        'label': 'Balance & Load',
        'icon': '⚖️',
        'dimensions': ['negative_move', 'pelvis_load', 'upper_torso_load'],
    },
    'stride': {
        'label': 'Stride',
        'icon': '👣',
        'dimensions': ['stride_length', 'forward_move'],
    },
    'power_move': {
        'label': 'Power Move',
        'icon': '💥',
        'dimensions': ['max_hip_shoulder_separation', 'pelvis_rotation_range', 'upper_torso_rotation_range'],
    },
    'contact': {
        'label': 'Contact & Follow-Through',
        'icon': '🎯',
        'dimensions': ['pelvis_direction_at_contact', 'upper_torso_direction_at_contact', 'kinetic_chain_efficiency', 'sequence_quality'],
    },
}

SKILL_LEVEL_BENCHMARKS = {
    'youth': {
        'power_range_W': (900, 2500),
        'hip_power_per_kg_elite': 8.0,
        'ke_per_kg_elite': 2.0,
        'chain_efficiency_elite': 25.0,
        'torso_pelvis_ratio_optimal': (0.8, 1.1),
        'x_factor_optimal': (25, 40),
        'sequence_timing_ms': (20, 50),
        'max_hand_speed_mph': (35, 55)
    },
    'high_school': {
        'power_range_W': (2300, 4300),
        'hip_power_per_kg_elite': 12.0,
        'ke_per_kg_elite': 3.5,
        'chain_efficiency_elite': 35.0,
        'torso_pelvis_ratio_optimal': (1.0, 1.2),
        'x_factor_optimal': (30, 50),
        'sequence_timing_ms': (30, 60),
        'max_hand_speed_mph': (45, 65)
    },
    'college': {
        'power_range_W': (2750, 4750),
        'hip_power_per_kg_elite': 16.0,
        'ke_per_kg_elite': 5.0,
        'chain_efficiency_elite': 40.0,
        'torso_pelvis_ratio_optimal': (1.1, 1.3),
        'x_factor_optimal': (35, 55),
        'sequence_timing_ms': (30, 60),
        'max_hand_speed_mph': (55, 75)
    },
    'professional': {
        'power_range_W': (3650, 5650),
        'hip_power_per_kg_elite': 20.0,
        'ke_per_kg_elite': 7.0,
        'chain_efficiency_elite': 45.0,
        'torso_pelvis_ratio_optimal': (1.2, 1.5),
        'x_factor_optimal': (40, 60),
        'sequence_timing_ms': (40, 60),
        'max_hand_speed_mph': (65, 85)
    }
}

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
    negative_move_m: float
    plant_frame: int
    plant_method: str
    estimated_hand_speed_mph: float
    overall_efficiency: int
    # Driveline-inspired Energy Transfer Metrics
    pelvis_ke_J: float = 0.0
    torso_ke_J: float = 0.0
    arm_ke_J: float = 0.0
    elbow_ke_J: float = 0.0
    total_energy_transfer_J: float = 0.0
    torso_to_arm_transfer_ratio: float = 0.0
    pelvis_to_torso_transfer_ratio: float = 0.0
    torso_to_pelvis_rot_ratio: float = 0.0
    kinetic_chain_efficiency_pct: float = 0.0
    # Advanced Driveline-inspired segment velocity & braking metrics
    peak_pelvis_omega_deg_s: float = 0.0
    peak_shoulder_omega_deg_s: float = 0.0
    peak_arm_omega_deg_s: float = 0.0
    pelvis_decel_at_contact_pct: float = 0.0
    velocity_amplification_ratio: float = 0.0
    swing_duration_ms: float = 0.0

class RefinedHittingOptimizer:
    def __init__(self, body_mass_kg: float, body_height_m: float, skill_level: str = 'high_school'):
        self.body_mass_kg = float(body_mass_kg)
        self.body_height_m = float(body_height_m)
        self.skill_level = skill_level if skill_level in SKILL_LEVEL_BENCHMARKS else 'high_school'
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
        for i, line in enumerate(lines):
            if 'endheader' in line.lower():
                header_end = i + 1
                break
        data = pd.read_csv(filepath, sep='\t', skiprows=header_end, skipinitialspace=True)
        data.columns = data.columns.str.strip()
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
        # Find whichever wrist is moving faster (proxy for bat speed)
        wrist_markers = ['r_mwrist_study', 'L_mwrist_study', 'r_lwrist_study', 'L_lwrist_study', 'RWrist', 'LWrist']
        
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
                    
        metrics = {
            'max_hand_speed_mps': float(max_hand_speed),
            'max_hand_speed_mph': float(max_hand_speed) * 2.23694
        }
        return metrics
        
    def calculate_rotational_torques_refined(self, data: pd.DataFrame) -> Dict:
        dt = data['time'].diff().mean()
        fs = 1.0 / dt if dt > 0 else 60.0
        
        # OpenCap generates jumping angles across -180/180 boundaries. We must UNWRAP them!
        if 'pelvis_rotation' not in data.columns or 'lumbar_rotation' not in data.columns:
            return None
            
        pelvis_angle_raw = np.deg2rad(data['pelvis_rotation'].values)
        pelvis_angle_unwrapped = np.unwrap(pelvis_angle_raw)
        
        lumbar_angle_raw = np.deg2rad(data['lumbar_rotation'].values)
        lumbar_angle_unwrapped = np.unwrap(lumbar_angle_raw)
        
        shoulder_angle_unwrapped = pelvis_angle_unwrapped + lumbar_angle_unwrapped
        
        if HAS_SCIPY:
            # 1. Zero-lag Butterworth filter (Cutoff: 15Hz) to remove OpenCap high-frequency noise
            cutoff_hz = 15.0
            pelvis_angle = butter_lowpass_filter(pelvis_angle_unwrapped, cutoff_hz, fs)
            shoulder_angle = butter_lowpass_filter(shoulder_angle_unwrapped, cutoff_hz, fs)
            
            # 2. Dynamic Savitzky-Golay window based on actual framerate (~100ms window)
            window_size = int(0.10 * fs)
            if window_size % 2 == 0: 
                window_size += 1
            window_size = max(11, window_size) # Default minimum
            
            pelvis_omega = savgol_smooth_and_diff(pelvis_angle, window=window_size, polyorder=3, deriv=1, dt=dt)
            pelvis_alpha = savgol_smooth_and_diff(pelvis_angle, window=window_size, polyorder=3, deriv=2, dt=dt)
            
            shoulder_omega = savgol_smooth_and_diff(shoulder_angle, window=window_size, polyorder=3, deriv=1, dt=dt)
            shoulder_alpha = savgol_smooth_and_diff(shoulder_angle, window=window_size, polyorder=3, deriv=2, dt=dt)
        else:
            pelvis_angle = pelvis_angle_unwrapped
            shoulder_angle = shoulder_angle_unwrapped
            
            pelvis_smooth = smooth_data(pelvis_angle, window=11)
            pelvis_omega = np.gradient(pelvis_smooth, dt)
            pelvis_alpha = np.gradient(pelvis_omega, dt)
            
            shoulder_smooth = smooth_data(shoulder_angle, window=11)
            shoulder_omega = np.gradient(shoulder_smooth, dt)
            shoulder_alpha = np.gradient(shoulder_omega, dt)
        
        trunk_I = self.segments['trunk']['I']
        thigh_I = self.segments['thigh']['I']
        upper_arm_I = self.segments['upper_arm']['I']
        forearm_I = self.segments['forearm']['I']

        # Pelvis-lumbar complex: lower trunk fraction (~30%) + bilateral thighs rotating
        # about the vertical pelvis axis (de Leva 1996 trunk partition).
        hip_inertia = trunk_I * 0.30 + 2 * thigh_I
        hip_torque = hip_inertia * pelvis_alpha

        bat_mass = 0.91
        # Bat COM at ~57% of 86 cm standard bat length from handle end -> 0.49 m
        bat_com_dist = 0.49
        bat_I = bat_mass * bat_com_dist ** 2

        # Upper body shoulder system: upper+mid trunk fraction (~70%) + bilateral arms + bat
        shoulder_inertia = trunk_I * 0.70 + 2 * (upper_arm_I + forearm_I) + bat_I
        shoulder_torque = shoulder_inertia * shoulder_alpha
        
        inertia_ratio = shoulder_inertia / hip_inertia
        hip_power = hip_torque * pelvis_omega
        shoulder_power = shoulder_torque * shoulder_omega
        
        peak_hip_torque = np.max(np.abs(hip_torque))
        peak_shoulder_torque = np.max(np.abs(shoulder_torque))
        peak_hip_power = np.max(np.abs(hip_power))
        peak_shoulder_power = np.max(np.abs(shoulder_power))
        
        hip_power_per_kg = peak_hip_power / self.body_mass_kg
        shoulder_power_per_kg = peak_shoulder_power / self.body_mass_kg
        
        separation = (shoulder_angle - pelvis_angle) * 180.0/np.pi
        max_separation = np.max(np.abs(separation))
        
        # Arm angular velocity: vector magnitude across all contributing DOF.
        # Internal rotation (arm_rot_r) and horizontal adduction (arm_add_r) are the
        # dominant contributors to hand speed in hitting; flexion alone is a poor proxy.
        arm_omega_sq = np.zeros_like(pelvis_omega)
        for col in ['arm_flex_r', 'arm_add_r', 'arm_rot_r']:
            if col in data.columns:
                ang = np.unwrap(np.deg2rad(data[col].values))
                if HAS_SCIPY:
                    ang = butter_lowpass_filter(ang, cutoff_hz, fs)
                    omega_dof = savgol_smooth_and_diff(ang, window=window_size, polyorder=3, deriv=1, dt=dt)
                else:
                    omega_dof = np.gradient(smooth_data(ang, window=11), dt)
                arm_omega_sq += omega_dof ** 2
        arm_omega = np.sqrt(arm_omega_sq)
                
        # Elbow Extension Whip
        elb_omega = np.zeros_like(pelvis_omega)
        if 'elbow_flex_r' in data.columns:
            elb_r_unwrapped = np.unwrap(np.deg2rad(data['elbow_flex_r'].values))
            if HAS_SCIPY:
                elb_r_filtered = butter_lowpass_filter(elb_r_unwrapped, cutoff_hz, fs)
                elb_omega = savgol_smooth_and_diff(elb_r_filtered, window=window_size, polyorder=3, deriv=1, dt=dt)
            else:
                elb_omega = np.gradient(smooth_data(elb_r_unwrapped, window=11), dt)
                
        # Proximal-to-Distal Sequencing using absolute max velocities
        peak_hip_frame = np.argmax(np.abs(pelvis_omega))
        peak_shoulder_frame = np.argmax(np.abs(shoulder_omega))
        peak_arm_frame = np.argmax(np.abs(arm_omega)) if np.sum(np.abs(arm_omega)) > 0 else peak_shoulder_frame + 1
        
        sequence_timing_ms = float((peak_shoulder_frame - peak_hip_frame) * dt * 1000.0)
        # True Proximal-to-Distal: Pelvis -> Torso -> Arms
        proper_sequence = bool(peak_hip_frame < peak_shoulder_frame and peak_shoulder_frame <= peak_arm_frame)
        
        # =========================================================================
        # DRIVELINE-INSPIRED: Segmental Kinetic Energy Transfer Analysis
        # Ref: drivelineresearch/openbiomechanics + autoresearch-claude-code
        # The #1 predictive feature class for velocity is ENERGY TRANSFER RATIOS
        # between adjacent kinematic chain segments, NOT raw angular velocities.
        # =========================================================================
        eps = 1e-6  # prevent division by zero
        
        peak_pelvis_w = float(np.max(np.abs(pelvis_omega)))
        peak_shoulder_w = float(np.max(np.abs(shoulder_omega)))
        peak_arm_w_val = float(np.max(np.abs(arm_omega)))
        peak_elb_w_val = float(np.max(np.abs(elb_omega)))
        
        # Segmental Kinetic Energy: KE = 0.5 * I * omega^2
        pelvis_ke = 0.5 * hip_inertia * (peak_pelvis_w ** 2)
        torso_ke = 0.5 * trunk_I * (peak_shoulder_w ** 2)
        arm_ke = 0.5 * (upper_arm_I + forearm_I) * 2 * (peak_arm_w_val ** 2)  # bilateral
        elbow_ke = 0.5 * forearm_I * 2 * (peak_elb_w_val ** 2)  # bilateral forearm
        
        total_energy_transfer = pelvis_ke + torso_ke + arm_ke + elbow_ke
        
        # Driveline's top transfer ratios (proximal-to-distal energy flow)
        torso_to_arm_ratio = torso_ke / (arm_ke + eps)
        pelvis_to_torso_ratio = pelvis_ke / (torso_ke + eps)
        torso_to_pelvis_rot_ratio = peak_shoulder_w / (peak_pelvis_w + eps)
        
        # Kinetic Chain Efficiency: how well does energy amplify distally?
        # In an ideal proximal-to-distal sequence, each distal segment should
        # have HIGHER angular velocity (but lower inertia) than its proximal neighbor.
        # Efficiency = (distal KE sum) / (total KE) — higher = better transfer
        distal_ke = arm_ke + elbow_ke
        chain_efficiency = (distal_ke / (total_energy_transfer + eps)) * 100.0
        
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
            'peak_arm_omega_rad_s': float(peak_arm_w_val),
            'peak_elb_omega_rad_s': float(peak_elb_w_val),
            'peak_shoulder_omega_rad_s': float(peak_shoulder_w),
            'peak_pelvis_omega_rad_s': float(peak_pelvis_w),
            'pelvis_omega': pelvis_omega,
            'pelvis_angle': pelvis_angle,
            'shoulder_angle': shoulder_angle,
            # Driveline Energy Transfer Metrics
            'pelvis_ke_J': float(pelvis_ke),
            'torso_ke_J': float(torso_ke),
            'arm_ke_J': float(arm_ke),
            'elbow_ke_J': float(elbow_ke),
            'total_energy_transfer_J': float(total_energy_transfer),
            'torso_to_arm_transfer_ratio': float(torso_to_arm_ratio),
            'pelvis_to_torso_transfer_ratio': float(pelvis_to_torso_ratio),
            'torso_to_pelvis_rot_ratio': float(torso_to_pelvis_rot_ratio),
            'kinetic_chain_efficiency_pct': float(chain_efficiency)
        }
        
    def calculate_stride_refined(self, data: pd.DataFrame, rotation: Dict = None) -> Dict:
        if 'pelvis_tx' not in data.columns:
            return None

        # Optional: apply low-pass filter to positions if scipy is available
        dt = data['time'].diff().mean()
        fs = 1.0 / dt if dt > 0 else 60.0

        # Stride is a horizontal-plane measurement: forward (tx) + lateral (tz).
        # pelvis_ty is the vertical axis in OpenSim/OpenCap and must NOT be included.
        pelvis_x = data['pelvis_tx'].values
        has_tz = 'pelvis_tz' in data.columns
        pelvis_z = data['pelvis_tz'].values if has_tz else np.zeros_like(pelvis_x)

        if HAS_SCIPY:
            pelvis_x = butter_lowpass_filter(pelvis_x, 15.0, fs)
            if has_tz:
                pelvis_z = butter_lowpass_filter(pelvis_z, 15.0, fs)
        
        # Event Detection: finding plant frame robustly
        if rotation and 'pelvis_omega' in rotation:
            pelvis_omega = rotation['pelvis_omega']
            pelvis_omega_abs_deg = np.abs(pelvis_omega) * 180.0/np.pi
            
            # Robust Threshold: 40% of peak velocity (instead of hardcoded 100 deg/s)
            peak_omega_deg = np.max(pelvis_omega_abs_deg)
            threshold = 0.40 * peak_omega_deg
            
            plant_candidates = np.where(pelvis_omega_abs_deg > threshold)[0]
            if len(plant_candidates) > 0:
                plant_frame = plant_candidates[0]
                plant_method = "velocity_threshold_40%"
            else:
                plant_frame = np.argmax(pelvis_omega_abs_deg)
                plant_method = "peak_rotation"
        else:
            plant_frame = len(data) // 2
            plant_method = "fallback_midframe"
        
        # Stride length in the horizontal plane (sagittal + lateral displacement).
        start_pos = np.array([pelvis_x[0], pelvis_z[0]])
        plant_pos = np.array([pelvis_x[plant_frame], pelvis_z[plant_frame]])
        stride_length = np.linalg.norm(plant_pos - start_pos)

        # Negative move: actual backward displacement of pelvis_tx before plant.
        pre_plant_x = pelvis_x[:plant_frame + 1]
        neg_move_m = float(max(0.0, pelvis_x[0] - np.min(pre_plant_x)))
        
        stride_ratio = stride_length / self.body_height_m
        optimal_stride_ratio = 0.75
        stride_efficiency_pct = (stride_ratio / optimal_stride_ratio) * 100.0
        
        return {
            'stride_length_m': float(stride_length),
            'stride_length_ft': float(stride_length * 3.28084),
            'stride_ratio': float(stride_ratio),
            'stride_efficiency_pct': float(stride_efficiency_pct),
            'plant_frame': int(plant_frame),
            'plant_time': float(data['time'].iloc[plant_frame]),
            'plant_method': plant_method,
            'negative_move_m': neg_move_m,
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
                # Elbow extension contributes additively to the hand's linear velocity
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
        
    def comprehensive_diagnosis(self, kinematics: pd.DataFrame, filename: str, trc_data: pd.DataFrame = None, verbose: bool = False) -> Dict:
        rotation = self.calculate_rotational_torques_refined(kinematics)
        stride = self.calculate_stride_refined(kinematics, rotation)
        trc_metrics = self.calculate_trc_metrics(trc_data) if trc_data is not None else {'max_hand_speed_mph': 0.0, 'max_hand_speed_mps': 0.0}
        hand_speed = self.estimate_hand_speed(rotation, trc_metrics)
        
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
            chain_eff = rotation.get('kinetic_chain_efficiency_pct', 0.0)
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

        # =========================================================================
        # ADVANCED DRIVELINE-INSPIRED METRICS
        # Ref: Driveline openbiomechanics dataset, Cohen et al. (2014) kinematic
        # sequence analysis, Fleisig & Escamilla (1996) arm/trunk velocity data.
        # =========================================================================
        _eps = 1e-6
        peak_pelvis_omega_deg_s = rotation.get('peak_pelvis_omega_rad_s', 0.0) * (180.0 / np.pi) if rotation else 0.0
        peak_shoulder_omega_deg_s = rotation.get('peak_shoulder_omega_rad_s', 0.0) * (180.0 / np.pi) if rotation else 0.0
        peak_arm_omega_deg_s = rotation.get('peak_arm_omega_rad_s', 0.0) * (180.0 / np.pi) if rotation else 0.0

        pelvis_decel_at_contact_pct = 0.0
        if rotation and stride and 'pelvis_omega' in rotation:
            _plant_idx = min(stride['plant_frame'], len(rotation['pelvis_omega']) - 1)
            _peak_pv = rotation.get('peak_pelvis_omega_rad_s', 0.0)
            if _peak_pv > _eps:
                _contact_pv = abs(float(rotation['pelvis_omega'][_plant_idx]))
                pelvis_decel_at_contact_pct = (_contact_pv / _peak_pv) * 100.0

        velocity_amplification_ratio = 0.0
        if rotation:
            _p = rotation.get('peak_pelvis_omega_rad_s', 0.0)
            _a = rotation.get('peak_arm_omega_rad_s', 0.0)
            if _p > _eps and _a > 0.0:
                velocity_amplification_ratio = _a / _p

        swing_duration_ms = float((kinematics['time'].iloc[-1] - kinematics['time'].iloc[0]) * 1000.0)

        if rotation and peak_pelvis_omega_deg_s > 10:
            if peak_pelvis_omega_deg_s >= 450:
                findings.append(f"Elite Peak Pelvis Rotational Velocity ({peak_pelvis_omega_deg_s:.0f} °/s; Driveline elite ≥450 °/s).")
            elif peak_pelvis_omega_deg_s >= 300:
                findings.append(f"Adequate Peak Pelvis Velocity ({peak_pelvis_omega_deg_s:.0f} °/s; elite target ≥450 °/s).")
                recommendations.append("Increase explosive hip rotation speed through overload/underload rotational training and improved lower-half loading patterns.")
            else:
                findings.append(f"Below-Average Peak Pelvis Velocity ({peak_pelvis_omega_deg_s:.0f} °/s; elite target ≥450 °/s).")
                recommendations.append("Insufficient hip rotational velocity. Prioritize explosive rotational med ball throws, hip mobility protocols, and ground-reaction force training.")
                efficiency_score -= 10

        if pelvis_decel_at_contact_pct > 0:
            if pelvis_decel_at_contact_pct <= 55:
                findings.append(f"Effective Pelvis Deceleration at Contact ({pelvis_decel_at_contact_pct:.0f}% of peak): front-leg brace efficiently redirects energy up the chain.")
            elif pelvis_decel_at_contact_pct <= 78:
                findings.append(f"Partial Pelvis Braking at Contact ({pelvis_decel_at_contact_pct:.0f}% of peak).")
                recommendations.append("Sharpen front-leg bracing. A faster pelvis deceleration at foot plant amplifies the energy whip into the torso and arms.")
            else:
                findings.append(f"Poor Pelvis Braking at Contact ({pelvis_decel_at_contact_pct:.0f}% of peak still active): pelvis not decelerating — energy stalls proximally.")
                recommendations.append("Lead leg must brace hard at foot plant. The pelvis must decelerate sharply to transfer energy up the chain (proximal-to-distal whip effect — Welch et al. 1995).")
                efficiency_score -= 12

        if velocity_amplification_ratio > 0:
            if velocity_amplification_ratio >= 2.5:
                findings.append(f"Elite Distal Velocity Amplification ({velocity_amplification_ratio:.1f}×): arm segments effectively multiplying pelvis angular speed.")
            elif velocity_amplification_ratio >= 1.5:
                findings.append(f"Good Velocity Amplification ({velocity_amplification_ratio:.1f}×): kinetic chain successfully amplifying arm speed from pelvis base.")
            else:
                findings.append(f"Low Distal Velocity Amplification ({velocity_amplification_ratio:.1f}×): arm speed not amplifying sufficiently above pelvis velocity.")
                recommendations.append("Keep hands and forearms relaxed during load. Tension in distal segments prevents them from being whipped by proximal deceleration (Hirashima et al. 2008).")

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
            negative_move_m=stride['negative_move_m'] if stride else 0.0,
            plant_frame=stride['plant_frame'] if stride else 0,
            plant_method=stride['plant_method'] if stride else "none",
            estimated_hand_speed_mph=hand_speed['estimated_hand_speed_mph'] if hand_speed else 0.0,
            overall_efficiency=max(0, efficiency_score),
            # Driveline Energy Transfer Metrics
            pelvis_ke_J=rotation.get('pelvis_ke_J', 0.0) if rotation else 0.0,
            torso_ke_J=rotation.get('torso_ke_J', 0.0) if rotation else 0.0,
            arm_ke_J=rotation.get('arm_ke_J', 0.0) if rotation else 0.0,
            elbow_ke_J=rotation.get('elbow_ke_J', 0.0) if rotation else 0.0,
            total_energy_transfer_J=rotation.get('total_energy_transfer_J', 0.0) if rotation else 0.0,
            torso_to_arm_transfer_ratio=rotation.get('torso_to_arm_transfer_ratio', 0.0) if rotation else 0.0,
            pelvis_to_torso_transfer_ratio=rotation.get('pelvis_to_torso_transfer_ratio', 0.0) if rotation else 0.0,
            torso_to_pelvis_rot_ratio=rotation.get('torso_to_pelvis_rot_ratio', 0.0) if rotation else 0.0,
            kinetic_chain_efficiency_pct=rotation.get('kinetic_chain_efficiency_pct', 0.0) if rotation else 0.0,
            peak_pelvis_omega_deg_s=peak_pelvis_omega_deg_s,
            peak_shoulder_omega_deg_s=peak_shoulder_omega_deg_s,
            peak_arm_omega_deg_s=peak_arm_omega_deg_s,
            pelvis_decel_at_contact_pct=pelvis_decel_at_contact_pct,
            velocity_amplification_ratio=velocity_amplification_ratio,
            swing_duration_ms=swing_duration_ms,
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
                
            print(f"\n" + "="*70)
            print(f"OVERALL EFFICIENCY: {max(0, efficiency_score)}/100")
            print("="*70)
            for finding in findings:
                print(f"   {finding}")
        
        swingai_report = self.build_swingai_report(rotation, stride, trc_metrics)
        
        return {
            "metrics": asdict(metrics),
            "findings": findings,
            "recommendations": recommendations,
            "efficiency_score": max(0, efficiency_score),
            "reference_values": SKILL_LEVEL_BENCHMARKS.get(self.skill_level, SKILL_LEVEL_BENCHMARKS['high_school']),
            "swingai_report": swingai_report,
            "swing_score": swingai_report['swing_score'],
        }

    def _rate_dimension(self, key: str, value: float, invert: bool = False) -> int:
        """Rate a single dimension 1-5 based on per-skill-level threshold table.
        For 'invert=True' dimensions (like direction-at-contact), lower value is better.
        Returns 1-5."""
        thresholds = SWINGAI_THRESHOLDS.get(key, {}).get(self.skill_level, [])
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
        """Convert a 1-5 star rating to a SwingAI-style color badge label."""
        if stars >= 5:
            return 'excellent'
        elif stars >= 3:
            return 'satisfactory'
        else:
            return 'off_target'

    def build_swingai_report(self, rotation: Dict, stride: Dict, trc_metrics: Dict) -> Dict:
        """Build a SwingAI-mirrored 12-dimension report from computed physics outputs.
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
            # Use actual measured backward pelvis displacement before plant frame.
            neg_move_m = stride.get('negative_move_m', 0.0)
            neg_move_stars = self._rate_dimension('negative_move', neg_move_m)
        else:
            neg_move_m = 0.0
            neg_move_stars = 2

        dims['negative_move'] = {
            'label': SWINGAI_LABELS['negative_move'],
            'stars': neg_move_stars,
            'badge': self._rating_to_badge(neg_move_stars),
            'value': round(neg_move_m, 3),
            'unit': 'm',
            'description': 'Initial weight shift rearward to load energy before the stride.',
        }

        # Pelvis Load: Pelvis KE during the load phase
        pelvis_ke = rotation.get('pelvis_ke_J', 0.0) if rotation else 0.0
        pl_stars = self._rate_dimension('pelvis_load', pelvis_ke)
        dims['pelvis_load'] = {
            'label': SWINGAI_LABELS['pelvis_load'],
            'stars': pl_stars,
            'badge': self._rating_to_badge(pl_stars),
            'value': round(pelvis_ke, 1),
            'unit': 'J',
            'description': 'Hip coil energy storage during the load phase.',
        }

        # Upper Torso Load: Torso KE
        torso_ke = rotation.get('torso_ke_J', 0.0) if rotation else 0.0
        utl_stars = self._rate_dimension('upper_torso_load', torso_ke)
        dims['upper_torso_load'] = {
            'label': SWINGAI_LABELS['upper_torso_load'],
            'stars': utl_stars,
            'badge': self._rating_to_badge(utl_stars),
            'value': round(torso_ke, 1),
            'unit': 'J',
            'description': 'Shoulder coil tension built during the load phase.',
        }

        # ------------------------------------------------------------------
        # PHASE 2: STRIDE
        # ------------------------------------------------------------------
        stride_ratio = stride['stride_ratio'] if stride else 0.0
        sl_stars = self._rate_dimension('stride_length', stride_ratio)
        dims['stride_length'] = {
            'label': SWINGAI_LABELS['stride_length'],
            'stars': sl_stars,
            'badge': self._rating_to_badge(sl_stars),
            'value': round(stride_ratio, 2),
            'unit': '× height',
            'description': 'Forward step distance relative to body height. Elite target ~75-90% of height.',
        }

        stride_eff = stride['stride_efficiency_pct'] if stride else 0.0
        # Penalize over-striding (>115%) as well as under-striding
        fm_val = min(stride_eff, 115.0) if stride_eff <= 115.0 else max(0, 115.0 - (stride_eff - 115.0))
        fm_stars = self._rate_dimension('forward_move', fm_val)
        dims['forward_move'] = {
            'label': SWINGAI_LABELS['forward_move'],
            'stars': fm_stars,
            'badge': self._rating_to_badge(fm_stars),
            'value': round(stride_eff, 1),
            'unit': '%',
            'description': 'Controlled forward momentum of the stride, stopping at front foot plant.',
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
            'label': SWINGAI_LABELS['max_hip_shoulder_separation'],
            'stars': mhs_stars,
            'badge': self._rating_to_badge(mhs_stars),
            'value': round(sep_deg, 1),
            'unit': '°',
            'description': 'Maximum angle between hips and shoulders. Stores elastic energy (X-Factor).',
        }

        # Pelvis Total Rotation Range — total radians traversed converted to degrees
        if rotation and 'pelvis_angle' in rotation:
            pelvis_ang = rotation['pelvis_angle']
            pelvis_rot_range = float(np.abs(np.max(pelvis_ang) - np.min(pelvis_ang)) * 180.0 / np.pi)
        else:
            pelvis_rot_range = 0.0
        prr_stars = self._rate_dimension('pelvis_rotation_range', pelvis_rot_range)
        dims['pelvis_rotation_range'] = {
            'label': SWINGAI_LABELS['pelvis_rotation_range'],
            'stars': prr_stars,
            'badge': self._rating_to_badge(prr_stars),
            'value': round(pelvis_rot_range, 1),
            'unit': '°',
            'description': 'Total hip rotation from load through contact.',
        }

        # Upper Torso Total Rotation Range — computed directly from shoulder_angle timeseries.
        if rotation and 'shoulder_angle' in rotation:
            shoulder_ang = rotation['shoulder_angle']
            torso_rot_range = float(np.abs(np.max(shoulder_ang) - np.min(shoulder_ang)) * 180.0 / np.pi)
        elif rotation:
            # Fallback: scale pelvis range by omega ratio
            torso_rot_range = pelvis_rot_range * rotation.get('torso_to_pelvis_rot_ratio', 1.0)
        else:
            torso_rot_range = 0.0
        utrr_stars = self._rate_dimension('upper_torso_rotation_range', torso_rot_range)
        dims['upper_torso_rotation_range'] = {
            'label': SWINGAI_LABELS['upper_torso_rotation_range'],
            'stars': utrr_stars,
            'badge': self._rating_to_badge(utrr_stars),
            'value': round(torso_rot_range, 1),
            'unit': '°',
            'description': 'Total shoulder rotation from load through contact.',
        }

        # ------------------------------------------------------------------
        # PHASE 4: CONTACT & FOLLOW-THROUGH
        # ------------------------------------------------------------------
        # Pelvis Direction at Contact — rotation accumulated from initial position to plant.
        # Target: hips have rotated ~90° (square to pitcher) by front foot plant.
        if rotation and 'pelvis_angle' in rotation and stride:
            plant_idx = min(stride['plant_frame'], len(rotation['pelvis_angle']) - 1)
            pelvis_rot_at_contact = (rotation['pelvis_angle'][plant_idx] - rotation['pelvis_angle'][0]) * 180.0 / np.pi
            # Deviation from the ideal 90° of rotation at contact
            pelvis_dev = abs(90.0 - abs(pelvis_rot_at_contact))
        else:
            pelvis_dev = 45.0
        pdc_stars = self._rate_dimension('pelvis_direction_at_contact', pelvis_dev, invert=True)
        dims['pelvis_direction_at_contact'] = {
            'label': SWINGAI_LABELS['pelvis_direction_at_contact'],
            'stars': pdc_stars,
            'badge': self._rating_to_badge(pdc_stars),
            'value': round(pelvis_dev, 1),
            'unit': '° off-square',
            'description': 'Hip alignment at contact. Hips should be square (90°) to the pitcher.',
        }

        # Upper Torso Direction at Contact — use shoulder_angle directly when available.
        if rotation and 'shoulder_angle' in rotation and stride:
            plant_idx = min(stride['plant_frame'], len(rotation['shoulder_angle']) - 1)
            shoulder_rot_at_contact = (rotation['shoulder_angle'][plant_idx] - rotation['shoulder_angle'][0]) * 180.0 / np.pi
            torso_dev = abs(90.0 - abs(shoulder_rot_at_contact))
        elif rotation:
            torso_dev = pelvis_dev * (1.0 / max(0.5, rotation.get('torso_to_pelvis_rot_ratio', 1.0)))
        else:
            torso_dev = 50.0
        utdc_stars = self._rate_dimension('upper_torso_direction_at_contact', float(torso_dev), invert=True)
        dims['upper_torso_direction_at_contact'] = {
            'label': SWINGAI_LABELS['upper_torso_direction_at_contact'],
            'stars': utdc_stars,
            'badge': self._rating_to_badge(utdc_stars),
            'value': round(torso_dev, 1),
            'unit': '° off-square',
            'description': 'Shoulder alignment at contact for optimal barrel control and plate coverage.',
        }

        # Kinetic Chain Efficiency
        chain_eff = rotation.get('kinetic_chain_efficiency_pct', 0.0) if rotation else 0.0
        kce_stars = self._rate_dimension('kinetic_chain_efficiency', chain_eff)
        dims['kinetic_chain_efficiency'] = {
            'label': SWINGAI_LABELS['kinetic_chain_efficiency'],
            'stars': kce_stars,
            'badge': self._rating_to_badge(kce_stars),
            'value': round(chain_eff, 1),
            'unit': '%',
            'description': 'Percentage of total body energy that reaches the hands/bat.',
        }

        # Sequence Quality — computed directly (not threshold lookup)
        if rotation:
            proper = rotation.get('proper_sequence', False)
            timing_ms = rotation.get('sequence_timing_ms', 0.0)
            benchmarks = SKILL_LEVEL_BENCHMARKS.get(self.skill_level, {})
            t_lo, t_hi = benchmarks.get('sequence_timing_ms', (20, 60))
            if proper and t_lo <= timing_ms <= t_hi:
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
            'label': SWINGAI_LABELS['sequence_quality'],
            'stars': sq_stars,
            'badge': self._rating_to_badge(sq_stars),
            'value': round(rotation.get('sequence_timing_ms', 0.0) if rotation else 0.0, 0),
            'unit': 'ms lag',
            'description': 'Proximal-to-distal sequencing: Pelvis → Torso → Arms in correct order and timing.',
        }

        # ------------------------------------------------------------------
        # SWING SCORE (0-100): Weighted average of dimension star ratings
        # Stars range 1-5. Normalise to 0-100 as (stars-1)/4 * 100, then weight.
        # ------------------------------------------------------------------
        total_weight = 0.0
        weighted_sum = 0.0
        for dim_key, weight in SWINGAI_WEIGHTS.items():
            stars = dims[dim_key]['stars']
            normalized = ((stars - 1) / 4.0) * 100.0
            weighted_sum += normalized * weight
            total_weight += weight
        swing_score = round(weighted_sum / total_weight, 1) if total_weight > 0 else 0.0

        # ------------------------------------------------------------------
        # Assemble phases
        # ------------------------------------------------------------------
        phases = {}
        for phase_key, phase_meta in SWINGAI_PHASES.items():
            phase_dims = [dims[d] for d in phase_meta['dimensions'] if d in dims]
            phase_avg_stars = sum(d['stars'] for d in phase_dims) / max(1, len(phase_dims))
            phases[phase_key] = {
                'label': phase_meta['label'],
                'icon': phase_meta['icon'],
                'avg_stars': round(phase_avg_stars, 1),
                'badge': self._rating_to_badge(round(phase_avg_stars)),
                'dimensions': phase_dims,
            }

        return {
            'swing_score': swing_score,
            'skill_level': self.skill_level,
            'phases': phases,
            'dimensions': dims,
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
