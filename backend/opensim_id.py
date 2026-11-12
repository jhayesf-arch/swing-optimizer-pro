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
from typing import Optional

try:
    import opensim as osim
    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

# Scaled model is per-athlete (OpenCap session → OpenSimData/Model/*.osim).
# Prefer a model uploaded with the trial; otherwise OPENSIM_MODEL_PATH; the old
# Desktop path stays as the last fallback so an existing local setup still runs.
DEFAULT_MODEL = os.environ.get("OPENSIM_MODEL_PATH") or os.path.expanduser(
    "~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae"
    "/OpenSimData/Model/LaiUhlrich2022_scaled.osim"
)
DEFAULT_SHOULDER_MODEL = os.path.expanduser(
    "~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae"
    "/OpenSimData/Model/LaiUhlrich2022_shoulder_scaled.osim"
)

# Lumbar rotation limit for baseball swing (±90° default is too loose)
LUMBAR_ROT_LIMIT_RAD = math.radians(50.0)


def run_inverse_dynamics(mot_path: str, model_path: str = DEFAULT_MODEL,
                          lowpass_hz: float = 10.0,
                          bat_mass_kg: float = 0.0,
                          bat_length_m: float = 0.0,
                          use_shoulder_model: bool = False,
                          external_loads: Optional[dict] = None,
                          time_range: Optional[tuple] = None,
                          consistent_grf: bool = True) -> dict:
    """
    Run OpenSim Inverse Dynamics on a swing .mot file.

    Parameters
    ----------
    lowpass_hz : coordinate filter before ID differentiates twice. 10 Hz, not the
                 15 Hz used for velocities elsewhere: on OpenCap capture 1001,
                 quiet standing at 15 Hz implied 0.2–1.65 BW vertical and 1.2 BW
                 sideways (impossible); at 10 Hz 0.56–1.28 and 0.45 BW. Provisional
                 until a clean swing is checked against lab force plates.
    use_shoulder_model : if True, use the ISB shoulder model (adds scapular DOFs)
                         instead of the standard ball-and-socket arm model.
    external_loads : per-foot ground reaction forces to apply, as produced by
                     bottom_up_id (keys time, grf_l, grf_r, cop_l, cop_r; ground
                     frame, N and m). Without them ID sees no ground force at all
                     and the lower-limb moments are missing the load the legs
                     carry — the pelvis residuals then absorb it instead.
    consistent_grf : with external_loads, take the TOTAL ground force from the
                     model's own dynamics (a first ID pass without loads) and
                     use the estimated loads only to split it between the feet
                     and place it (CoP). The estimate comes from a simpler body
                     model than OpenSim's, so applying it as-is leaves a pelvis
                     residual of ~1 BW that the leg moments silently absorb.
                     The result also carries grf_check: whether the force the
                     motion requires is physically possible for a swing.
    """
    if use_shoulder_model:
        try:
            from scale_shoulder_model import scale_shoulder_for_session
            search = os.path.dirname(mot_path)
            shoulder_found = False
            for _ in range(5):
                for candidate in [os.path.join(search, "LaiUhlrich2022_scaled.osim"),
                                   os.path.join(search, "Model", "LaiUhlrich2022_scaled.osim")]:
                    if os.path.exists(candidate):
                        model_path = scale_shoulder_for_session(candidate)
                        shoulder_found = True
                        break
                if shoulder_found:
                    break
                search = os.path.dirname(search)
            if not shoulder_found and os.path.exists(DEFAULT_SHOULDER_MODEL):
                model_path = DEFAULT_SHOULDER_MODEL
        except Exception as e:
            print(f"Warning: could not load shoulder model ({e}), using standard model")
    if not HAS_OPENSIM:
        raise RuntimeError("opensim not found. Activate the opencap-processing conda env.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(mot_path):
        raise FileNotFoundError(f"Motion file not found: {mot_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── 1. Trim .mot to swing window ────────────────────────────────────
        trimmed_mot = os.path.join(tmpdir, "swing_trimmed.mot")
        t_start, t_end = _trim_to_swing_window(mot_path, trimmed_mot, time_range=time_range)

        # ── 2. Build modified model (lumbar limits + optional bat) ──────────
        mod_model_path = os.path.join(tmpdir, "model_modified.osim")
        _build_modified_model(model_path, mod_model_path, bat_mass_kg, bat_length_m)

        # ── 3. Run ID ────────────────────────────────────────────────────────
        model = osim.Model(mod_model_path)
        state = model.initSystem()
        weight_N = model.getTotalMass(state) * 9.81

        def _run(loads, name):
            sto = os.path.join(tmpdir, f"{name}.sto")
            id_tool = osim.InverseDynamicsTool()
            id_tool.setModel(model)
            id_tool.setCoordinatesFileName(trimmed_mot)
            id_tool.setLowpassCutoffFrequency(lowpass_hz)
            id_tool.setStartTime(t_start)
            id_tool.setEndTime(t_end)
            id_tool.setOutputGenForceFileName(sto)
            id_tool.setResultsDir(tmpdir)
            force_set = osim.ArrayStr()
            force_set.append("Muscles")
            id_tool.setExcludedForces(force_set)
            if loads is not None:
                load_dir = os.path.join(tmpdir, name)
                os.makedirs(load_dir, exist_ok=True)
                id_tool.setExternalLoadsFileName(write_external_loads(loads, load_dir))
            id_tool.run()
            if not os.path.exists(sto):
                raise RuntimeError("ID tool produced no output.")
            return sto, _load_sto(sto)

        # Pass 1, no ground force: the pelvis translational residuals are then
        # exactly the external force the model's own motion requires
        # (M·a_com − M·g, from the same filtered coordinates ID differentiates).
        grf_check = None
        loads_used = external_loads
        if external_loads is not None and consistent_grf:
            _, df_free = _run(None, "ID_no_loads")
            required = required_ground_force(df_free)
            if required is not None:
                grf_check = check_required_grf(required, weight_N)
                loads_used = consistent_foot_loads(external_loads, df_free['time'].values, required)

        sto_path, df = _run(loads_used, "ID_results")
        out_sto = mot_path.replace(".mot", "_ID.sto")
        shutil.copy(sto_path, out_sto)

    joints = _extract_peak_kinetics(df)
    return {'sto_path': out_sto, 'joints': joints, 'dataframe': df,
            'swing_t_start': t_start, 'swing_t_end': t_end,
            'external_loads_applied': loads_used is not None,
            'grf_source': ('model_consistent' if loads_used is not external_loads
                           else 'estimated' if loads_used is not None else 'none'),
            'grf_check': grf_check, 'body_weight_N': weight_N,
            'external_loads': loads_used, 'lowpass_hz': lowpass_hz,
            'model_path': model_path}


# ── External loads ───────────────────────────────────────────────────────────

# Body each foot's ground reaction is applied to on the LaiUhlrich2022 model.
_FOOT_BODY = {'r': 'calcn_r', 'l': 'calcn_l'}


def write_external_loads(loads: dict, out_dir: str) -> str:
    """Write an OpenSim ExternalLoads pair (.xml + .mot) for per-foot GRF.

    Force and point are both expressed in ground, which is the frame OpenCap's
    .trc markers — and therefore our GRF and CoP — are in. Column names follow
    OpenSim's identifier convention: force_identifier "ground_force_r_v" reads
    columns ground_force_r_vx/vy/vz, and so on. No free torque is modelled, so
    the torque columns are zero.

    Pure file writing — importable and testable without opensim installed.
    Returns the .xml path.
    """
    t = np.asarray(loads['time'], dtype=float)
    cols = {'time': t}
    for side in ('r', 'l'):
        F = np.nan_to_num(np.asarray(loads[f'grf_{side}'], dtype=float))
        P = np.nan_to_num(np.asarray(loads[f'cop_{side}'], dtype=float))
        for j, ax in enumerate('xyz'):
            cols[f'ground_force_{side}_v{ax}'] = F[:, j]
        for j, ax in enumerate('xyz'):
            cols[f'ground_force_{side}_p{ax}'] = P[:, j]
        for ax in 'xyz':
            cols[f'ground_torque_{side}_{ax}'] = np.zeros_like(t)

    mot_path = os.path.join(out_dir, 'external_loads.mot')
    names = list(cols.keys())
    with open(mot_path, 'w') as f:
        f.write('external_loads\nversion=1\n')
        f.write(f'nRows={len(t)}\nnColumns={len(names)}\ninDegrees=no\nendheader\n')
        f.write('\t'.join(names) + '\n')
        data = np.column_stack([cols[n] for n in names])
        for row in data:
            f.write('\t'.join(f'{v:.6f}' for v in row) + '\n')

    root = ET.Element('OpenSimDocument', Version='40000')
    el = ET.SubElement(root, 'ExternalLoads', name='externalloads')
    objs = ET.SubElement(el, 'objects')
    for side in ('r', 'l'):
        ef = ET.SubElement(objs, 'ExternalForce', name=f'grf_{side}')
        for tag, val in [('applied_to_body', _FOOT_BODY[side]),
                         ('force_expressed_in_body', 'ground'),
                         ('point_expressed_in_body', 'ground'),
                         ('force_identifier', f'ground_force_{side}_v'),
                         ('point_identifier', f'ground_force_{side}_p'),
                         ('torque_identifier', f'ground_torque_{side}_'),
                         ('data_source_name', 'Unassigned')]:
            ET.SubElement(ef, tag).text = val
    ET.SubElement(el, 'groups')
    ET.SubElement(el, 'datafile').text = mot_path
    ET.SubElement(el, 'external_loads_model_kinematics_file').text = ''
    ET.SubElement(el, 'lowpass_cutoff_frequency_for_load_kinematics').text = '-1'
    xml_path = os.path.join(out_dir, 'external_loads.xml')
    ET.ElementTree(root).write(xml_path, encoding='UTF-8', xml_declaration=True)
    return xml_path


# ── Model-consistent ground force ────────────────────────────────────────────

# What a swing's ground force can physically be, in body weights. Measured at
# the Cal Poly MBL (trial 261105_1001_0001, force plates): total vertical peak
# 1.71 BW, horizontal under 0.15 BW per foot. The limits are kept generous so a
# hard swing is never rejected; what they catch is a motion whose implied force
# is impossible — the ground cannot pull (vertical < 0), and a batter cannot
# push several body weights sideways. Such forces come from noisy accelerations
# (e.g. an IK trunk flip), and every joint moment computed from them inherits it.
GRF_LIMITS_BW = {'vertical_max': 3.0, 'vertical_min': -0.25, 'horizontal_max': 1.0}
_PELVIS_FORCE_COLS = ('pelvis_tx_force', 'pelvis_ty_force', 'pelvis_tz_force')


def required_ground_force(df_no_loads: pd.DataFrame) -> Optional[np.ndarray]:
    """Total external force (N, ground frame, Y up) the motion requires.

    With no external loads applied, ID's pelvis translational generalized forces
    are M·a_com − M·g: exactly what the ground must supply. OpenCap models put
    the pelvis translations along ground X/Y/Z, so the columns are the vector.
    """
    if not all(c in df_no_loads.columns for c in _PELVIS_FORCE_COLS):
        return None
    return df_no_loads[list(_PELVIS_FORCE_COLS)].values.astype(float)


def check_required_grf(required: np.ndarray, weight_N: float) -> dict:
    """Is the ground force this motion implies physically possible?"""
    F = np.asarray(required, dtype=float) / weight_N
    vert = F[:, 1]
    horiz = np.linalg.norm(F[:, [0, 2]], axis=1)
    out = {'peak_vertical_BW': round(float(vert.max()), 2),
           'min_vertical_BW': round(float(vert.min()), 2),
           'peak_horizontal_BW': round(float(horiz.max()), 2)}
    lim = GRF_LIMITS_BW
    reasons = []
    if out['peak_vertical_BW'] > lim['vertical_max']:
        reasons.append(f"vertical {out['peak_vertical_BW']:.1f} BW (a swing stays under ~{lim['vertical_max']:g})")
    if out['min_vertical_BW'] < lim['vertical_min']:
        reasons.append(f"the ground would have to pull down ({out['min_vertical_BW']:.1f} BW)")
    if out['peak_horizontal_BW'] > lim['horizontal_max']:
        reasons.append(f"sideways {out['peak_horizontal_BW']:.1f} BW (a swing stays under ~{lim['horizontal_max']:g})")
    out['plausible'] = not reasons
    out['reasons'] = reasons
    return out


def consistent_foot_loads(estimated: dict, time: np.ndarray, required: np.ndarray) -> dict:
    """Split the model-required total force between the feet.

    The estimate's vertical share per foot decides the split (equal when the
    estimate has no vertical force), and its CoP places each foot's force.
    The total then matches the model's dynamics exactly, so the pelvis force
    residual is ~0 and the leg moments carry the load the legs really do.
    """
    t_est = np.asarray(estimated['time'], dtype=float)
    time = np.asarray(time, dtype=float)

    def interp(arr):
        a = np.nan_to_num(np.asarray(arr, dtype=float))
        return np.column_stack([np.interp(time, t_est, a[:, j]) for j in range(3)])

    fl = np.clip(interp(estimated['grf_l'])[:, 1], 0, None)
    fr = np.clip(interp(estimated['grf_r'])[:, 1], 0, None)
    tot = fl + fr
    w_l = np.divide(fl, tot, out=np.full_like(tot, 0.5), where=tot > 1e-6)
    req = np.asarray(required, dtype=float)
    return {'time': time,
            'grf_l': req * w_l[:, None], 'grf_r': req * (1 - w_l)[:, None],
            'cop_l': interp(estimated['cop_l']), 'cop_r': interp(estimated['cop_r'])}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_mot_df(mot_path: str) -> pd.DataFrame:
    with open(mot_path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(mot_path, sep='\t', skiprows=header_end, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df, lines[:header_end]


def _trim_to_swing_window(mot_path: str, out_path: str,
                           window_s: float = 0.5, time_range: Optional[tuple] = None) -> tuple:
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

    # Search only the swing the rest of the report analysed (main._crop_to_swing),
    # so a faster turn elsewhere in the recording cannot move the ID window.
    in_range = np.ones(len(time), bool)
    if time_range is not None:
        in_range = (time >= time_range[0]) & (time <= time_range[1])
        if not in_range.any():
            in_range[:] = True
    peak_frame = int(np.flatnonzero(in_range)[np.argmax(np.abs(pelvis_omega[in_range]))])
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
        'peak_shoulder_torque_r_Nm': _peak('arm_flex_r', 'arm_add_r', 'arm_rot_r',
                                            'sh_plane_elev_r', 'sh_elev_r', 'sh_axial_rot_r'),
        'peak_shoulder_torque_l_Nm': _peak('arm_flex_l', 'arm_add_l', 'arm_rot_l',
                                            'sh_plane_elev_l', 'sh_elev_l', 'sh_axial_rot_l'),
        'peak_scapular_force_r_N':   _peak('sh_tx_r', 'sh_ty_r', 'sh_tz_r'),
        'peak_scapular_force_l_N':   _peak('sh_tx_l', 'sh_ty_l', 'sh_tz_l'),
        'peak_elbow_torque_r_Nm':    _peak('elbow_flex_r'),
        'peak_elbow_torque_l_Nm':    _peak('elbow_flex_l'),
        'peak_prosup_torque_r_Nm':   _peak('pro_sup_r'),
        'peak_prosup_torque_l_Nm':   _peak('pro_sup_l'),
    }


def summarize_id_magnitudes(df: pd.DataFrame, body_mass_kg: Optional[float] = None) -> dict:
    """Peak joint-moment magnitudes, shaped to compare against bottom-up ID.

    OpenSim reports one generalized force per coordinate; the bottom-up chain
    reports a 3-D moment vector. The hip has three coordinates, so its magnitude
    is the per-frame norm of all three. The knee has one (flexion), so frontal-
    and transverse-plane knee load that the bottom-up vector includes has no
    coordinate here — expect bottom-up ≥ OpenSim at the knee by construction.
    The ankle combines ankle_angle and subtalar.

    Pelvis residuals are the six forces/moments OpenSim needs at the pelvis to
    balance the motion. With good ground reaction forces applied they should be
    small; large residuals mean the applied GRF and the kinematics disagree.
    """
    def _series(name, suffixes):
        """First existing column for a coordinate, or None."""
        for suf in suffixes:
            if name + suf in df.columns:
                return df[name + suf].values.astype(float)
        return None

    def _norm_peak(names, suffixes):
        arrs = [a for a in (_series(n, suffixes) for n in names) if a is not None]
        if not arrs:
            return 0.0
        return float(np.max(np.sqrt(np.sum(np.vstack(arrs) ** 2, axis=0))))

    def norm_peak(*names):
        return _norm_peak(names, ('_moment', ''))

    def res_peak(*names):
        return _norm_peak(names, ('_force', '_moment', ''))

    out = {
        'hip_moment_l_Nm':   norm_peak('hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'),
        'hip_moment_r_Nm':   norm_peak('hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r'),
        'knee_moment_l_Nm':  norm_peak('knee_angle_l'),
        'knee_moment_r_Nm':  norm_peak('knee_angle_r'),
        'ankle_moment_l_Nm': norm_peak('ankle_angle_l', 'subtalar_angle_l'),
        'ankle_moment_r_Nm': norm_peak('ankle_angle_r', 'subtalar_angle_r'),
        'pelvis_residual_force_N':   res_peak('pelvis_tx', 'pelvis_ty', 'pelvis_tz'),
        'pelvis_residual_moment_Nm': res_peak('pelvis_tilt', 'pelvis_list', 'pelvis_rotation'),
    }
    if body_mass_kg:
        out['pelvis_residual_force_BW'] = out['pelvis_residual_force_N'] / (body_mass_kg * 9.81)
    return out


def _cli():
    """python opensim_id.py trial.mot --trc trial.trc --model scaled.osim [--mass 81]

    With --trc the run matches the web upload: ground force estimated from the
    markers, made consistent with the model, split between the feet. Without it
    ID has no ground force and the leg/pelvis numbers are not meaningful.
    """
    import argparse
    p = argparse.ArgumentParser(description='OpenSim inverse dynamics for one swing.')
    p.add_argument('mot', help='OpenCap IK result (.mot)')
    p.add_argument('--trc', help='OpenCap markers (.trc) — needed for ground forces')
    p.add_argument('--model', default=DEFAULT_MODEL, help='athlete-scaled .osim')
    p.add_argument('--mass', type=float, default=None, help='body mass kg (default: model mass)')
    p.add_argument('--height', type=float, default=1.83, help='body height m')
    p.add_argument('--bat-mass', type=float, default=0.88)
    p.add_argument('--bat-length', type=float, default=0.864)
    a = p.parse_args()

    loads, time_range = None, None
    mass = a.mass
    if mass is None and HAS_OPENSIM:
        m = osim.Model(a.model)
        mass = m.getTotalMass(m.initSystem())
    if a.trc:
        from analyzer import RefinedHittingOptimizer
        from main import _crop_to_swing, _time_range
        opt = RefinedHittingOptimizer(body_mass_kg=mass or 82.0, body_height_m=a.height,
                                      bat_mass_kg=a.bat_mass, bat_length_m=a.bat_length)
        kin = opt.load_mot_file(a.mot)
        kin, trc = _crop_to_swing(kin, opt.load_trc_file(a.trc))
        diag = opt.comprehensive_diagnosis(kin, os.path.basename(a.mot), trc_data=trc,
                                           model_path=a.model)
        loads, time_range = diag.get('_foot_loads'), _time_range(kin)
        if loads is None:
            print('Warning: no ground force could be estimated from the .trc; running without.')
    else:
        print('Warning: no --trc, so no ground force is applied. Leg and pelvis results '
              'are not meaningful; pass the trial .trc.')

    r = run_inverse_dynamics(a.mot, model_path=a.model, bat_mass_kg=a.bat_mass,
                             bat_length_m=a.bat_length, external_loads=loads,
                             time_range=time_range)
    mags = summarize_id_magnitudes(r['dataframe'], r['body_weight_N'] / 9.81)
    print(f"\nSwing window {r['swing_t_start']:.2f}–{r['swing_t_end']:.2f} s   "
          f"ground force: {r['grf_source']}")
    chk = r.get('grf_check')
    if chk:
        print(f"Ground force the motion requires: vertical {chk['min_vertical_BW']}..{chk['peak_vertical_BW']} BW, "
              f"sideways peak {chk['peak_horizontal_BW']} BW -> "
              + ('plausible' if chk['plausible'] else 'NOT plausible: ' + '; '.join(chk['reasons'])))
        if not chk['plausible']:
            print('The motion is too noisy for inverse dynamics; the moments below are not reliable.')
    print(f"Pelvis residual: {mags['pelvis_residual_force_N']:.0f} N "
          f"({mags.get('pelvis_residual_force_BW', 0):.2f} BW), {mags['pelvis_residual_moment_Nm']:.0f} N·m")
    for k in ('hip_moment_l_Nm', 'hip_moment_r_Nm', 'knee_moment_l_Nm', 'knee_moment_r_Nm',
              'ankle_moment_l_Nm', 'ankle_moment_r_Nm'):
        print(f"  {k:<20} {mags[k]:7.0f}")
    print(f"Full results: {r['sto_path']}")


if __name__ == '__main__':
    _cli()
