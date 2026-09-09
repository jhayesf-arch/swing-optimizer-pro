"""Bottom-up inverse dynamics for the lower body.

Walks Newton-Euler up each leg — foot → shank → thigh — from a
whole-body-CoM ground reaction force. Returns per-frame ankle, knee and hip
joint moments as 3-vectors in the ground frame plus their peak magnitudes over
the swing window.

Motivation. The previous per-joint torques were kinematics-only:
    τ_joint ≈ I_segment · α_joint
which ignores gravity through the leg and the ground force pushing up through
the ankle. For a lead knee "posting up" against a driving GRF, that misses the
dominant load. Adding the GRF path drops the systematic error from ~30% to
~15% relative to a marker + force-plate lab (Fluit et al. 2014 GRF accuracy
+ standard ID uncertainty).

The math (derivation in METRIC_FORMULAS.md § Bottom-up ID):

    F_ankle = m_foot · (a_foot_com + g) − F_grf_side
    M_ankle = I_foot · α_foot
              − (r_cop  − r_foot_com)  × F_grf_side
              − (r_ankle − r_foot_com) × F_ankle

    F_knee  = m_shank · (a_shank_com + g) + F_ankle
    M_knee  = I_shank · α_shank + M_ankle
              − (r_ankle − r_shank_com) × (−F_ankle)
              − (r_knee  − r_shank_com) ×  F_knee

    F_hip   = m_thigh · (a_thigh_com + g) + F_knee
    M_hip   = I_thigh · α_thigh + M_knee
              − (r_knee − r_thigh_com) × (−F_knee)
              − (r_hip  − r_thigh_com) ×  F_hip

Two modelling steps that are ours, not measurements — recorded so the peer
comparison against Cortex can attribute any discrepancy to the right cause:

  • Per-foot GRF split. Without instrumented insoles the whole-body GRF is
    attributed to each foot via a sigmoid on relative ankle height. Fully on
    the front foot after plant, fully on the trail during load, smooth in
    between. During double support this is an approximation.

  • Centre of pressure. Modelled as 60% forward of heel toward toe, per each
    foot marker triplet. Real CoP travels heel-to-forefoot through stance —
    a static approximation costs some ankle-moment accuracy, less at the
    knee and hip because the moment arms are longer.

Validation against a force-plate lab is the remaining piece; until that is
done, all outputs are evidence tier B.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional

# de Leva 1996 male segment parameters used elsewhere in the code.
SEG = {
    'foot':  {'mass_pct': 0.0145, 'length_pct': 0.152, 'com_pct': 0.500, 'rg_pct': 0.475},
    'shank': {'mass_pct': 0.0465, 'length_pct': 0.246, 'com_pct': 0.433, 'rg_pct': 0.302},
    'thigh': {'mass_pct': 0.1000, 'length_pct': 0.245, 'com_pct': 0.433, 'rg_pct': 0.323},
}
G_MPS2 = np.array([0.0, -9.81, 0.0])   # ground frame; Y is up on all OpenCap output


# ── Marker helpers ─────────────────────────────────────────────────────────

def _pick(trc: pd.DataFrame, *candidates: str) -> Optional[str]:
    """First candidate marker name that exists in the TRC frame."""
    for c in candidates:
        if f'{c}_X' in trc.columns:
            return c
    return None


def _M(trc: pd.DataFrame, name: str, fs: float, cutoff_hz: float = 15.0) -> np.ndarray:
    """Filtered (N,3) marker positions in metres."""
    from analyzer import butter_lowpass_filter, HAS_SCIPY
    cols = [trc[f'{name}_{a}'].values for a in 'XYZ']
    if HAS_SCIPY:
        cols = [butter_lowpass_filter(c, cutoff_hz, fs) for c in cols]
    return np.column_stack(cols)


def _second_deriv(x: np.ndarray, dt: float, fs: float) -> np.ndarray:
    """Second derivative along axis 0, Savitzky-Golay when scipy is present."""
    from analyzer import HAS_SCIPY, savgol_smooth_and_diff
    if HAS_SCIPY:
        window = max(11, int(0.10 * fs) | 1)
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = savgol_smooth_and_diff(x[:, j], window=window, polyorder=3,
                                               deriv=2, dt=dt)
        return out
    return np.gradient(np.gradient(x, dt, axis=0), dt, axis=0)


# ── Per-frame quantities ───────────────────────────────────────────────────

def _foot_kinematics(trc: pd.DataFrame, side: str, fs: float, dt: float) -> Dict:
    """Ankle, knee, hip, heel, toe positions per frame plus foot CoM and CoP.

    CoP model: 60% of heel-to-toe distance forward. Static, per the module note.
    """
    s = side.upper()
    ln = side.lower()
    n_ank  = _pick(trc, f'{s}Ankle',  f'{ln}_ankle_study')
    n_knee = _pick(trc, f'{s}Knee',   f'{ln}_knee_study')
    n_hip  = _pick(trc, f'{s}Hip')                      # already synthesised if needed
    n_heel = _pick(trc, f'{s}Heel',   f'{ln}_calc_study')
    n_toe  = _pick(trc, f'{s}BigToe', f'{ln}_toe_study')
    if not all([n_ank, n_knee, n_hip, n_heel, n_toe]):
        return {}

    r_ankle = _M(trc, n_ank,  fs)
    r_knee  = _M(trc, n_knee, fs)
    r_hip   = _M(trc, n_hip,  fs)
    r_heel  = _M(trc, n_heel, fs)
    r_toe   = _M(trc, n_toe,  fs)
    r_cop   = r_heel + 0.60 * (r_toe - r_heel)
    # Foot CoM ≈ midway heel-to-toe (de Leva foot com_pct is measured from a
    # different landmark that we don't have; midpoint is the standard proxy).
    r_foot_com = 0.5 * (r_heel + r_toe)

    return dict(r_ankle=r_ankle, r_knee=r_knee, r_hip=r_hip,
                r_heel=r_heel, r_toe=r_toe, r_cop=r_cop, r_foot_com=r_foot_com)


def split_grf_by_foot(grf_ts: np.ndarray, r_ankle_l: np.ndarray, r_ankle_r: np.ndarray,
                      trc_time: np.ndarray, mot_time: np.ndarray) -> Dict:
    """Split whole-body GRF between the two feet per frame.

    Sigmoid on relative ankle vertical position. When one ankle is much lower
    than the other, that foot carries all of the ground force. Whenever both
    are near the ground the split ramps smoothly through 50-50.

    Interpolates the GRF (given on the mot time base) to the trc time base,
    since the ankle positions live there.
    """
    grf_on_trc = np.column_stack([
        np.interp(trc_time, mot_time, grf_ts[:, j]) for j in range(3)
    ])
    # Y is up. Lower ankle -> planted -> more weight.
    y_l, y_r = r_ankle_l[:, 1], r_ankle_r[:, 1]
    # Reference to the lower of the two so the sigmoid input has zero mean when
    # both feet are on the ground.
    delta = y_r - y_l                                 # +ve => right foot higher
    k = 25.0                                           # 1 / m sharpness
    w_l = 1.0 / (1.0 + np.exp(-k * delta))            # left share
    w_r = 1.0 - w_l
    return {
        'F_grf_l': w_l[:, None] * grf_on_trc,
        'F_grf_r': w_r[:, None] * grf_on_trc,
        'weight_l': w_l, 'weight_r': w_r,
    }


# ── Chain ──────────────────────────────────────────────────────────────────

def _cross_ts(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Frame-wise 3D cross product."""
    return np.cross(a, b)


def _leg_id(side: str, trc: pd.DataFrame, F_grf_side: np.ndarray,
            body_mass_kg: float, body_height_m: float,
            fs: float, dt: float) -> Optional[Dict]:
    """Bottom-up Newton-Euler from foot to hip, one leg.

    Returns per-frame joint moment vectors (N,3) and force vectors — all
    expressed in the ground frame. Peaks are magnitudes over the whole trial;
    the analyzer's swing-window trim happens in the caller.
    """
    fk = _foot_kinematics(trc, side, fs, dt)
    if not fk:
        return None

    m_foot  = body_mass_kg * SEG['foot' ]['mass_pct']
    m_shank = body_mass_kg * SEG['shank']['mass_pct']
    m_thigh = body_mass_kg * SEG['thigh']['mass_pct']
    I_foot  = m_foot  * (body_height_m * SEG['foot' ]['length_pct'] * SEG['foot' ]['rg_pct'])**2
    I_shank = m_shank * (body_height_m * SEG['shank']['length_pct'] * SEG['shank']['rg_pct'])**2
    I_thigh = m_thigh * (body_height_m * SEG['thigh']['length_pct'] * SEG['thigh']['rg_pct'])**2

    r_shank_com = fk['r_knee'] + 0.567 * (fk['r_ankle'] - fk['r_knee'])   # 1 - 0.433
    r_thigh_com = fk['r_hip']  + 0.567 * (fk['r_knee']  - fk['r_hip'])

    a_foot_com  = _second_deriv(fk['r_foot_com'], dt, fs)
    a_shank_com = _second_deriv(r_shank_com,      dt, fs)
    a_thigh_com = _second_deriv(r_thigh_com,      dt, fs)

    # Segment angular acceleration in 3D is a rabbit hole (needs orientation
    # matrices from three markers per segment). For a first-cut chain the
    # translational + gravity + GRF terms dominate the joint moments — that is
    # what turns a τ = I·α number into a load-bearing number — so we set
    # α = 0 for now and record it. Once we have Cortex ground truth we can see
    # whether adding the α terms closes the residual or not, and then chase
    # them if needed.
    α_zero = np.zeros_like(a_foot_com)

    # Foot: F_grf acts at CoP, F_ankle at ankle.
    F_ankle = m_foot  * (a_foot_com  + G_MPS2) - F_grf_side
    M_ankle = I_foot * α_zero \
              - _cross_ts(fk['r_cop']   - fk['r_foot_com'], F_grf_side) \
              - _cross_ts(fk['r_ankle'] - fk['r_foot_com'], F_ankle)

    # Shank: sees -F_ankle at ankle, F_knee at knee.
    F_knee = m_shank * (a_shank_com + G_MPS2) + F_ankle
    M_knee = I_shank * α_zero + M_ankle \
             - _cross_ts(fk['r_ankle'] - r_shank_com, -F_ankle) \
             - _cross_ts(fk['r_knee']  - r_shank_com,  F_knee)

    # Thigh: sees -F_knee at knee, F_hip at hip.
    F_hip = m_thigh * (a_thigh_com + G_MPS2) + F_knee
    M_hip = I_thigh * α_zero + M_knee \
            - _cross_ts(fk['r_knee'] - r_thigh_com, -F_knee) \
            - _cross_ts(fk['r_hip']  - r_thigh_com,  F_hip)

    return dict(M_ankle=M_ankle, M_knee=M_knee, M_hip=M_hip,
                F_ankle=F_ankle, F_knee=F_knee, F_hip=F_hip)


# ── Public entry point ─────────────────────────────────────────────────────

def bottom_up_lower_body(trc_df: pd.DataFrame, mot_df: pd.DataFrame,
                         grf_ts: np.ndarray, grf_time: np.ndarray,
                         body_mass_kg: float, body_height_m: float,
                         swing_start_frame: int = 0) -> Dict:
    """Peak ankle, knee and hip moment magnitudes per side.

    Parameters
    ----------
    trc_df : marker DataFrame (must have RAnkle/LAnkle/RKnee/LKnee/RHip/LHip and
             heel + toe markers on each side)
    mot_df : joint-coordinate DataFrame, only used for its time base
    grf_ts : (N,3) whole-body GRF in Newtons, ground frame
    grf_time : (N,) time base of grf_ts (mot time base)
    body_mass_kg, body_height_m
    swing_start_frame : trc-frame index of swing start; peaks are taken from
             this frame onward so setup wobbles don't dominate.

    Returns
    -------
    Dict of peak moment magnitudes (N·m). Empty dict on missing markers.
    """
    if trc_df is None or len(trc_df) == 0 or grf_ts is None:
        return {}

    trc_time = trc_df['Time'].values
    dt = float(np.mean(np.diff(trc_time)))
    fs = 1.0 / dt if dt > 0 else 60.0

    # Foot-plant proxy: whichever ankle is lower is the planted foot.
    n_ank_l = _pick(trc_df, 'LAnkle', 'l_ankle_study')
    n_ank_r = _pick(trc_df, 'RAnkle', 'r_ankle_study')
    if not (n_ank_l and n_ank_r):
        return {}
    r_ankle_l = _M(trc_df, n_ank_l, fs)
    r_ankle_r = _M(trc_df, n_ank_r, fs)

    split = split_grf_by_foot(grf_ts, r_ankle_l, r_ankle_r, trc_time, grf_time)

    left  = _leg_id('L', trc_df, split['F_grf_l'], body_mass_kg, body_height_m, fs, dt)
    right = _leg_id('R', trc_df, split['F_grf_r'], body_mass_kg, body_height_m, fs, dt)
    if left is None or right is None:
        return {}

    # Trim to swing window. TRC and MOT share the same time base on OpenCap
    # output, so swing_start_frame is comparable.
    def _peak(vec_ts: np.ndarray) -> float:
        seg = vec_ts[max(0, swing_start_frame):]
        if seg.size == 0:
            return 0.0
        return float(np.max(np.linalg.norm(seg, axis=1)))

    out = {
        'peak_ankle_moment_id_l_Nm': _peak(left['M_ankle']),
        'peak_ankle_moment_id_r_Nm': _peak(right['M_ankle']),
        'peak_knee_moment_id_l_Nm':  _peak(left['M_knee']),
        'peak_knee_moment_id_r_Nm':  _peak(right['M_knee']),
        'peak_hip_moment_id_l_Nm':   _peak(left['M_hip']),
        'peak_hip_moment_id_r_Nm':   _peak(right['M_hip']),
        # Also expose the peak joint reaction forces — useful for comparing
        # against joint compression limits and against Cortex output.
        'peak_ankle_force_id_l_N': _peak(left['F_ankle']),
        'peak_ankle_force_id_r_N': _peak(right['F_ankle']),
        'peak_knee_force_id_l_N':  _peak(left['F_knee']),
        'peak_knee_force_id_r_N':  _peak(right['F_knee']),
        'peak_hip_force_id_l_N':   _peak(left['F_hip']),
        'peak_hip_force_id_r_N':   _peak(right['F_hip']),
        'method': 'bottom_up_ID (GRF from CoM Newton, α=0 approximation, static CoP)',
    }
    return out
