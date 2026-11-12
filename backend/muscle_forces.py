"""
muscle_forces.py
----------------
Fast static optimization for every analysis (seconds, not minutes).

At each frame it solves the classic static-optimization problem
    minimise  sum(a_i^2) + sum(r_j^2)
    subject   sum_i R_ji(q) * a_i * F0_i + r_j = tau_j     for each leg DOF j
              0 <= a_i <= 1
where R are the muscle moment arms from the athlete's OpenSim model at that
frame's pose, F0 the muscles' maximum isometric forces, tau the inverse-
dynamics moments (opensim_id.run_inverse_dynamics) and r reserve torques with
an optimal force of 1 N·m, so the muscles do the work and any reserve torque
marks a moment the muscles cannot make.

Muscle force is activation x max isometric force. That is OpenSim's own
StaticOptimization with use_muscle_physiology = false; it leaves out the
force-length-velocity curves, which is what makes it fast (no fibre
equilibrium per frame). static_optimization.py runs OpenSim's full version for
comparison.

Only leg muscles: LaiUhlrich2022 has 80, 40 per leg, and none in the trunk or
arms (those are torque-driven). The result is gated like ID itself: a run is
reported only when the ground force was plausible, no leg joint leans on
reserves for more than 25% of its moment, and the pelvis residual moment is
under 1% of body weight x COM height (Hicks et al. 2015). Not yet validated
against force plates or EMG for batting.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import opensim as osim
    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

from static_optimization import (MUSCLE_GROUPS, MAX_RESERVE_SHARE,   # noqa: F401
                                 RESIDUAL_MOMENT_LIMIT_FRAC, _group)

LEG_DOFS = ('hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle',
            'ankle_angle', 'subtalar_angle', 'mtp_angle')
ARM_EVERY = 2          # moment arms every 2nd frame, interpolated between
PENALTY = 1e3          # weight on meeting the joint moments exactly


def solve_frame(A: np.ndarray, tau: np.ndarray, n_muscles: int):
    """min |x|^2 s.t. A x = tau (penalised), 0<=a<=1, reserves free."""
    from scipy.optimize import lsq_linear
    n = A.shape[1]
    M = np.vstack([PENALTY * A, np.eye(n)])
    b = np.concatenate([PENALTY * tau, np.zeros(n)])
    lb = np.r_[np.zeros(n_muscles), -np.inf * np.ones(n - n_muscles)]
    ub = np.r_[np.ones(n_muscles), np.inf * np.ones(n - n_muscles)]
    return lsq_linear(M, b, bounds=(lb, ub), method='bvls', lsmr_tol='auto').x


def _filtered_coordinates(mot_df: pd.DataFrame, cols: List[str], lowpass_hz: float) -> pd.DataFrame:
    from scipy.signal import butter, filtfilt
    t = mot_df['time'].values.astype(float)
    fs = 1.0 / float(np.mean(np.diff(t)))
    b, a = butter(4, min(lowpass_hz / (fs / 2), 0.99))
    out = {'time': t}
    for c in cols:
        v = mot_df[c].values.astype(float)
        out[c] = filtfilt(b, a, v) if len(v) > 15 else v
    return pd.DataFrame(out)


def compute(model_path: str, mot_df: pd.DataFrame, id_df: pd.DataFrame,
            lowpass_hz: float = 10.0, in_degrees: bool = True) -> Dict:
    """Per-frame activations and forces for every leg muscle, plus reserves."""
    if not HAS_OPENSIM:
        raise RuntimeError('opensim not available')
    t0 = time.time()
    model = osim.Model(model_path)
    state = model.initSystem()
    cs = model.getCoordinateSet()
    mus = model.getMuscles()
    names = [mus.get(i).getName() for i in range(mus.getSize())]
    f0 = np.array([mus.get(i).getMaxIsometricForce() for i in range(mus.getSize())])
    dofs = [f'{d}_{s}' for s in ('l', 'r') for d in LEG_DOFS
            if cs.contains(f'{d}_{s}') and f'{d}_{s}_moment' in id_df.columns]
    times = id_df['time'].values.astype(float)
    tau = id_df[[d + '_moment' for d in dofs]].values.astype(float)

    leg_cols = [c for c in mot_df.columns if c != 'time' and cs.contains(c)
                and not cs.get(c).getName().startswith('pelvis')]
    q = _filtered_coordinates(mot_df, leg_cols, lowpass_hz)

    def set_pose(tq):
        for c in leg_cols:
            v = float(np.interp(tq, q['time'].values, q[c].values))
            coord = cs.get(c)
            if in_degrees and coord.getMotionType() == 1:      # rotational
                v = np.deg2rad(v)
            coord.setValue(state, v, False)
        model.assemble(state)          # satisfies the knee (patella) coupler
        model.realizePosition(state)

    # Which muscle crosses which DOF: check a few poses of this swing.
    pairs = set()
    for tq in (times[0], times[len(times) // 2], times[-1]):
        set_pose(tq)
        for j, d in enumerate(dofs):
            c = cs.get(d)
            for i in range(len(names)):
                if abs(mus.get(i).computeMomentArm(state, c)) > 1e-5:
                    pairs.add((i, j))
    pairs = sorted(pairs)

    # Moment arms on every ARM_EVERY-th frame, linearly interpolated.
    idx = list(range(0, len(times), ARM_EVERY))
    if idx[-1] != len(times) - 1:
        idx.append(len(times) - 1)
    R_s = np.zeros((len(idx), len(dofs), len(names)))
    for k, fi in enumerate(idx):
        set_pose(times[fi])
        for i, j in pairs:
            R_s[k, j, i] = mus.get(i).computeMomentArm(state, cs.get(dofs[j]))
    R = np.empty((len(times), len(dofs), len(names)))
    for j in range(len(dofs)):
        for i in range(len(names)):
            R[:, j, i] = np.interp(np.arange(len(times)), idx, R_s[:, j, i])
    t_arms = time.time() - t0

    act = np.zeros((len(times), len(names)))
    res = np.zeros((len(times), len(dofs)))
    for f in range(len(times)):
        A = np.hstack([R[f] * f0[None, :], np.eye(len(dofs))])
        x = solve_frame(A, tau[f], len(names))
        act[f], res[f] = x[:len(names)], x[len(names):]
    act_df = pd.DataFrame(act, columns=names)
    act_df.insert(0, 'time', times)
    force_df = pd.DataFrame(act * f0[None, :], columns=names)
    force_df.insert(0, 'time', times)
    for j, d in enumerate(dofs):
        force_df['reserve_' + d] = res[:, j]
    return {'activations': act_df, 'forces': force_df, 'dofs': dofs,
            'seconds': round(time.time() - t0, 2), 'seconds_moment_arms': round(t_arms, 2)}


def summarize(act: pd.DataFrame, force: pd.DataFrame, id_df: pd.DataFrame,
              handedness: Optional[str] = None) -> Dict:
    from static_optimization import summarize as _summ
    return _summ(act, force, id_df, handedness)


def muscle_report(model_path: str, mot_df: pd.DataFrame, id_result: dict,
                  handedness: Optional[str] = None) -> Dict:
    """Run the fast solve on an ID result and apply the reliability gates.

    Returns {status: 'ok'|'withheld'|'failed', reason, groups, reserves, ...}.
    Numbers are only included when status is 'ok'.
    """
    out = {'method': 'static optimization (activation x max force), leg muscles only',
           'experimental': True}
    chk = id_result.get('grf_check')
    if chk and not chk['plausible']:
        out.update(status='withheld', reason='The tracked motion implies an impossible ground '
                   'force, so the joint moments the muscles would have to make are not reliable.')
        return out
    df = id_result['dataframe']
    try:
        r = compute(model_path, mot_df, df, lowpass_hz=id_result.get('lowpass_hz', 10.0))
    except Exception as e:      # never let this break an analysis
        out.update(status='failed', reason=f'{type(e).__name__}: {e}')
        return out
    s = summarize(r['activations'], r['forces'], df, handedness)
    weight = id_result.get('body_weight_N') or 0.0
    model = osim.Model(model_path)
    com_h = float(model.calcMassCenterPosition(model.initSystem()).get(1))
    limit = RESIDUAL_MOMENT_LIMIT_FRAC * weight * com_h
    cols = [c for c in ('pelvis_tilt_moment', 'pelvis_list_moment', 'pelvis_rotation_moment') if c in df]
    res_m = float(np.max(np.abs(df[cols].values))) if cols else 0.0
    out['seconds'] = r['seconds']
    out['residual_moment_check'] = {'peak_Nm': round(res_m, 1), 'limit_Nm': round(limit, 1),
                                    'ok': res_m <= limit}
    out['reserves'] = s['reserves']
    reasons = []
    if s['joints_not_explained_by_muscles']:
        reasons.append('the leg muscles could not produce the moments at '
                       + ', '.join(s['joints_not_explained_by_muscles']))
    if res_m > limit:
        reasons.append(f'the ground force estimate leaves {res_m:.0f} N·m unbalanced at the pelvis '
                       f'(limit {limit:.0f} N·m, Hicks et al. 2015)')
    if reasons:
        out.update(status='withheld', reason='Muscle loads withheld: ' + '; '.join(reasons) + '.')
        return out
    out.update(status='ok', groups=s['groups'])
    return out
