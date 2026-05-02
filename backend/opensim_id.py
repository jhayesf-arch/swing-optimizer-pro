"""
opensim_id.py
-------------
Runs OpenSim's Inverse Dynamics tool on a swing .mot file using the
subject-scaled LaiUhlrich2022 model from OpenCap.

Improvements over baseline:
  1. Trims .mot to swing window (±0.5s around peak pelvis omega) before ID
     — prevents pre-swing standing from contaminating the filter IC.
  2. Tightens lumbar_rotation joint limits to ±50° in a temp model copy
     — prevents IK from pushing lumbar to its ±90° hard limit.
  3. Adds bat mass/inertia to hand_r body
     — makes upper-extremity torques physically accurate.
"""
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
import math
import numpy as np
import pandas as pd

try:
    import opensim as osim
    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

DEFAULT_MODEL = os.path.expanduser(
    "~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae"
    "/OpenSimData/Model/LaiUhlrich2022_scaled.osim"
)

# Lumbar rotation limit for baseball swing (±90° default is too loose)
LUMBAR_ROT_LIMIT_RAD = math.radians(50.0)


def run_inverse_dynamics(mot_path: str, model_path: str = DEFAULT_MODEL,
                          lowpass_hz: float = 15.0,
                          bat_mass_kg: float = 0.0,
                          bat_length_m: float = 0.0) -> dict:
    """
    Run OpenSim Inverse Dynamics on a swing .mot file.

    Parameters
    ----------
    bat_mass_kg   : bat mass in kg (0 = no bat added)
    bat_length_m  : bat length in metres, used to compute moment of inertia
    """
    if not HAS_OPENSIM:
        raise RuntimeError("opensim not found. Activate the opencap-processing conda env.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"Motion file not found: {mot_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── 1. Trim .mot to swing window ────────────────────────────────────
        trimmed_mot = os.path.join(tmpdir, "swing_trimmed.mot")
        t_start, t_end = _trim_to_swing_window(mot_path, trimmed_mot)

        # ── 2. Build modified model (lumbar limits + optional bat) ──────────
        mod_model_path = os.path.join(tmpdir, "model_modified.osim")
        _build_modified_model(model_path, mod_model_path, bat_mass_kg, bat_length_m)

        # ── 3. Run ID ────────────────────────────────────────────────────────
        sto_path = os.path.join(tmpdir, "ID_results.sto")
        model = osim.Model(mod_model_path)
        model.initSystem()

        id_tool = osim.InverseDynamicsTool()
        id_tool.setModel(model)
        id_tool.setCoordinatesFileName(trimmed_mot)
        id_tool.setLowpassCutoffFrequency(lowpass_hz)
        id_tool.setStartTime(t_start)
        id_tool.setEndTime(t_end)
        id_tool.setOutputGenForceFileName(sto_path)
        id_tool.setResultsDir(tmpdir)
        force_set = osim.ArrayStr()
        force_set.append("Muscles")
        id_tool.setExcludedForces(force_set)
        id_tool.run()

        if not os.path.exists(sto_path):
            raise RuntimeError("ID tool produced no output.")

        df = _load_sto(sto_path)
        out_sto = mot_path.replace(".mot", "_ID.sto")
        shutil.copy(sto_path, out_sto)

    joints = _extract_peak_kinetics(df)
    return {'sto_path': out_sto, 'joints': joints, 'dataframe': df,
            'swing_t_start': t_start, 'swing_t_end': t_end}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_mot_df(mot_path: str) -> pd.DataFrame:
    with open(mot_path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(mot_path, sep='\t', skiprows=header_end, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df, lines[:header_end]


def _trim_to_swing_window(mot_path: str, out_path: str,
                           window_s: float = 0.5) -> tuple:
    """
    Detect swing onset by walking backward from peak pelvis omega.
    Write a trimmed .mot covering [onset - 0.1s, onset + window_s].
    Returns (t_start, t_end).
    """
    df, header_lines = _read_mot_df(mot_path)
    time = df['time'].values
    dt = float(np.diff(time).mean())

    if 'pelvis_rotation' not in df.columns:
        # Can't detect swing — use full trial
        shutil.copy(mot_path, out_path)
        return float(time[0]), float(time[-1])

    try:
        from scipy.signal import butter, filtfilt, savgol_filter
        pelvis_rad = np.unwrap(np.deg2rad(df['pelvis_rotation'].values))
        nyq = 0.5 / dt
        b, a = butter(4, min(15.0 / nyq, 0.99), btype='low')
        pelvis_f = filtfilt(b, a, pelvis_rad)
        w = max(11, int(0.10 / dt) | 1)
        pelvis_omega = savgol_filter(pelvis_f, w, 3, deriv=1, delta=dt)
    except Exception:
        pelvis_omega = np.gradient(np.unwrap(np.deg2rad(df['pelvis_rotation'].values)), dt)

    peak_frame = int(np.argmax(np.abs(pelvis_omega)))
    onset_frame = 0
    for i in range(peak_frame, -1, -1):
        if abs(pelvis_omega[i]) * 180 / np.pi < 50.0:
            onset_frame = i
            break

    t_onset = time[onset_frame]
    t_start = max(time[0], t_onset - 0.1)
    t_end   = min(time[-1], t_onset + window_s)

    mask = (time >= t_start) & (time <= t_end)
    df_trim = df[mask].copy()

    # Rewrite .mot with same header
    with open(out_path, 'w') as f:
        for line in header_lines:
            # Update nRows
            if line.strip().startswith('nRows'):
                f.write(f'nRows={len(df_trim)}\n')
            else:
                f.write(line)
        df_trim.to_csv(f, sep='\t', index=False)

    return float(t_start), float(t_end)


def _build_modified_model(src_path: str, dst_path: str,
                           bat_mass_kg: float, bat_length_m: float):
    """
    Copy the .osim model and apply:
      - Tighten lumbar_rotation range to ±LUMBAR_ROT_LIMIT_RAD
      - Add bat mass/inertia to hand_r body (if bat_mass_kg > 0)
    Uses string replacement to avoid full XML re-serialisation (which
    can corrupt the large .osim file).
    """
    with open(src_path, 'r') as f:
        content = f.read()

    # ── Tighten lumbar_rotation limits ──────────────────────────────────────
    # The model has: <range>-1.5707963300000001 1.5707963300000001</range>
    # immediately after <Coordinate name="lumbar_rotation">
    lim = f'{LUMBAR_ROT_LIMIT_RAD:.10f}'
    old_range = '-1.5707963300000001 1.5707963300000001'
    new_range = f'-{lim} {lim}'
    # Only replace the one inside lumbar_rotation coordinate block
    marker = '<Coordinate name="lumbar_rotation">'
    idx = content.find(marker)
    if idx != -1:
        block_end = content.find('</Coordinate>', idx) + len('</Coordinate>')
        block = content[idx:block_end]
        block_new = block.replace(old_range, new_range, 1)
        content = content[:idx] + block_new + content[block_end:]

    # ── Add bat mass/inertia to hand_r ──────────────────────────────────────
    if bat_mass_kg > 0 and bat_length_m > 0:
        # Bat modelled as uniform rod: I_perp = (1/12)*m*L^2 (about CoM)
        # CoM at L/2 from handle (grip end), so offset from hand origin ≈ L/2
        bat_I_perp = (1.0 / 12.0) * bat_mass_kg * bat_length_m ** 2
        bat_I_long = 0.001 * bat_mass_kg  # negligible axial inertia

        hand_r_marker = '<Body name="hand_r">'
        idx = content.find(hand_r_marker)
        if idx != -1:
            # Find existing mass tag in this body block
            mass_tag_start = content.find('<mass>', idx)
            mass_tag_end   = content.find('</mass>', idx) + len('</mass>')
            old_mass_str   = content[mass_tag_start:mass_tag_end]
            old_mass = float(old_mass_str.replace('<mass>', '').replace('</mass>', '').strip())
            new_mass = old_mass + bat_mass_kg
            content = content[:mass_tag_start] + f'<mass>{new_mass}</mass>' + content[mass_tag_end:]

            # Find existing inertia tag and add bat contribution
            inertia_start = content.find('<inertia>', idx)
            inertia_end   = content.find('</inertia>', idx) + len('</inertia>')
            old_inertia_str = content[inertia_start:inertia_end]
            vals = [float(v) for v in old_inertia_str.replace('<inertia>', '').replace('</inertia>', '').split()]
            # Ixx, Iyy (long axis), Izz — bat swings primarily in Ixx/Izz plane
            vals[0] += bat_I_perp
            vals[1] += bat_I_long
            vals[2] += bat_I_perp
            new_inertia = '<inertia>' + ' '.join(f'{v:.10f}' for v in vals) + '</inertia>'
            content = content[:inertia_start] + new_inertia + content[inertia_end:]

    with open(dst_path, 'w') as f:
        f.write(content)


def _load_sto(sto_path: str) -> pd.DataFrame:
    with open(sto_path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(sto_path, sep='\t', skiprows=header_end, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def _extract_peak_kinetics(df: pd.DataFrame) -> dict:
    joints = {}
    for col in [c for c in df.columns if c != 'time']:
        joints[col] = {'peak_torque_Nm': round(float(np.max(np.abs(df[col].values))), 2)}
    return joints


def summarize_id_results(id_result: dict) -> dict:
    j = id_result['joints']

    def _peak(*keys):
        vals = []
        for k in keys:
            for suffix in ('', '_moment', '_force'):
                full = k + suffix
                if full in j:
                    vals.append(j[full]['peak_torque_Nm'])
        return max(vals) if vals else 0.0

    return {
        'pelvis_residual_force_N':   _peak('pelvis_tx', 'pelvis_ty', 'pelvis_tz'),
        'pelvis_residual_torque_Nm': _peak('pelvis_tilt', 'pelvis_list', 'pelvis_rotation'),
        'peak_lumbar_torque_Nm':     _peak('lumbar_extension', 'lumbar_bending', 'lumbar_rotation'),
        'peak_hip_torque_r_Nm':      _peak('hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r'),
        'peak_hip_torque_l_Nm':      _peak('hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'),
        'peak_knee_torque_r_Nm':     _peak('knee_angle_r'),
        'peak_knee_torque_l_Nm':     _peak('knee_angle_l'),
        'peak_ankle_torque_r_Nm':    _peak('ankle_angle_r'),
        'peak_ankle_torque_l_Nm':    _peak('ankle_angle_l'),
        'peak_shoulder_torque_r_Nm': _peak('arm_flex_r', 'arm_add_r', 'arm_rot_r'),
        'peak_shoulder_torque_l_Nm': _peak('arm_flex_l', 'arm_add_l', 'arm_rot_l'),
        'peak_elbow_torque_r_Nm':    _peak('elbow_flex_r'),
        'peak_elbow_torque_l_Nm':    _peak('elbow_flex_l'),
        'peak_prosup_torque_r_Nm':   _peak('pro_sup_r'),
        'peak_prosup_torque_l_Nm':   _peak('pro_sup_l'),
    }


if __name__ == '__main__':
    import sys
    mot = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        '~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae'
        '/OpenSimData/Kinematics/swing_lower_first.mot')
    bat_mass   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.88   # kg (~31oz)
    bat_length = float(sys.argv[3]) if len(sys.argv) > 3 else 0.864  # m  (34 inch)

    print(f'Running OpenSim ID: {os.path.basename(mot)}  bat={bat_mass}kg {bat_length}m')
    result = run_inverse_dynamics(mot, bat_mass_kg=bat_mass, bat_length_m=bat_length)
    summary = summarize_id_results(result)
    print(f'Swing window: {result["swing_t_start"]:.2f}s – {result["swing_t_end"]:.2f}s')
    print(f'Saved: {result["sto_path"]}')
    print('\n=== JOINT TORQUE SUMMARY ===')
    for k, v in summary.items():
        print(f'  {k:<35} {v:>8.1f} N·m')
