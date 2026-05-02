"""
opensim_id.py
-------------
Runs OpenSim's Inverse Dynamics tool on a swing .mot file using the
subject-scaled LaiUhlrich2022 model from OpenCap.

Returns a dict of peak joint torques/powers for every joint in the model,
replacing the manual τ=Iα approximation in analyzer.py.
"""
import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

try:
    import opensim as osim
    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

# Default scaled model — the one OpenCap generates per session
DEFAULT_MODEL = os.path.expanduser(
    "~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae"
    "/OpenSimData/Model/LaiUhlrich2022_scaled.osim"
)


def run_inverse_dynamics(mot_path: str, model_path: str = DEFAULT_MODEL,
                          lowpass_hz: float = 15.0) -> dict:
    """
    Run OpenSim Inverse Dynamics on a .mot file.

    Returns
    -------
    dict with keys:
        'sto_path'   : path to the raw .sto output file
        'joints'     : dict of joint_name -> {'peak_torque_Nm', 'peak_power_W'}
        'dataframe'  : full ID results as a pandas DataFrame (time + all generalized forces)
    """
    if not HAS_OPENSIM:
        raise RuntimeError("opensim package not found. Activate the opencap-processing conda env.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"Motion file not found: {mot_path}")

    # Read time range from .mot header
    t_start, t_end = _get_time_range(mot_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        sto_path = os.path.join(tmpdir, "ID_results.sto")

        # Build and run the ID tool via OpenSim API
        model = osim.Model(model_path)
        model.initSystem()

        id_tool = osim.InverseDynamicsTool()
        id_tool.setModel(model)
        id_tool.setCoordinatesFileName(mot_path)
        id_tool.setLowpassCutoffFrequency(lowpass_hz)
        id_tool.setStartTime(t_start)
        id_tool.setEndTime(t_end)
        id_tool.setOutputGenForceFileName(sto_path)
        id_tool.setResultsDir(tmpdir)
        # Exclude muscles — we want net joint torques only
        force_set = osim.ArrayStr()
        force_set.append("Muscles")
        id_tool.setExcludedForces(force_set)

        id_tool.run()

        if not os.path.exists(sto_path):
            raise RuntimeError("ID tool ran but produced no output file.")

        df = _load_sto(sto_path)
        # Copy out before tmpdir is deleted
        import shutil
        out_sto = mot_path.replace(".mot", "_ID.sto")
        shutil.copy(sto_path, out_sto)

    joints = _extract_peak_kinetics(df)
    return {
        'sto_path': out_sto,
        'joints': joints,
        'dataframe': df,
    }


def _get_time_range(mot_path: str):
    """Read first and last time value from a .mot file."""
    with open(mot_path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(mot_path, sep='\t', skiprows=header_end, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return float(df['time'].iloc[0]), float(df['time'].iloc[-1])


def _load_sto(sto_path: str) -> pd.DataFrame:
    """Parse an OpenSim .sto file into a DataFrame."""
    with open(sto_path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(sto_path, sep='\t', skiprows=header_end, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def _extract_peak_kinetics(df: pd.DataFrame) -> dict:
    """
    From the ID .sto DataFrame, extract peak torque for each joint DOF.
    Also estimates peak power using finite-difference velocity from the torque signal.
    """
    joints = {}
    time = df['time'].values
    dt = np.diff(time).mean()

    torque_cols = [c for c in df.columns if c != 'time']
    for col in torque_cols:
        torque = df[col].values
        peak_torque = float(np.max(np.abs(torque)))
        # Approximate angular velocity via finite difference of torque integral
        # (not ideal but avoids needing the .mot again here)
        # Better: caller can pass omega separately. For now report torque only.
        joints[col] = {
            'peak_torque_Nm': round(peak_torque, 2),
        }
    return joints


def summarize_id_results(id_result: dict) -> dict:
    """
    Collapse the per-DOF ID results into the biomechanically meaningful
    joint groups used by the swing analyzer.
    OpenSim ID output uses _moment / _force suffixes on column names.
    """
    j = id_result['joints']

    def _peak(*keys):
        # Try both bare name and _moment / _force variants
        vals = []
        for k in keys:
            for suffix in ('', '_moment', '_force'):
                full = k + suffix
                if full in j:
                    vals.append(j[full]['peak_torque_Nm'])
        return max(vals) if vals else 0.0

    return {
        # Pelvis residuals — should be near zero for a good IK solution
        'pelvis_residual_force_N':   _peak('pelvis_tx', 'pelvis_ty', 'pelvis_tz'),
        'pelvis_residual_torque_Nm': _peak('pelvis_tilt', 'pelvis_list', 'pelvis_rotation'),
        # Lumbar
        'peak_lumbar_torque_Nm':     _peak('lumbar_extension', 'lumbar_bending', 'lumbar_rotation'),
        # Hip
        'peak_hip_torque_r_Nm':      _peak('hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r'),
        'peak_hip_torque_l_Nm':      _peak('hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'),
        # Knee
        'peak_knee_torque_r_Nm':     _peak('knee_angle_r'),
        'peak_knee_torque_l_Nm':     _peak('knee_angle_l'),
        # Ankle
        'peak_ankle_torque_r_Nm':    _peak('ankle_angle_r'),
        'peak_ankle_torque_l_Nm':    _peak('ankle_angle_l'),
        # Shoulder / arm
        'peak_shoulder_torque_r_Nm': _peak('arm_flex_r', 'arm_add_r', 'arm_rot_r'),
        'peak_shoulder_torque_l_Nm': _peak('arm_flex_l', 'arm_add_l', 'arm_rot_l'),
        # Elbow
        'peak_elbow_torque_r_Nm':    _peak('elbow_flex_r'),
        'peak_elbow_torque_l_Nm':    _peak('elbow_flex_l'),
        # Pronation/supination
        'peak_prosup_torque_r_Nm':   _peak('pro_sup_r'),
        'peak_prosup_torque_l_Nm':   _peak('pro_sup_l'),
    }


if __name__ == '__main__':
    import sys, json

    mot = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.expanduser('~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae'
                           '/OpenSimData/Kinematics/swing_lower_first.mot')

    print(f"Running OpenSim ID on: {mot}")
    result = run_inverse_dynamics(mot)
    summary = summarize_id_results(result)

    print(f"\nSaved raw ID output: {result['sto_path']}")
    print("\n=== JOINT TORQUE SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<35} {v:>8.1f} N·m")
