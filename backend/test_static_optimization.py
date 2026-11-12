"""Tests for the static-optimization summary (no OpenSim needed).
Run: python backend/test_static_optimization.py"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_optimization import _group, summarize, MAX_RESERVE_SHARE  # noqa: E402


def test_muscles_map_to_groups_and_unknowns_do_not():
    assert _group('glmax2_l') == 'Glute max'
    assert _group('addmagIsch_r') == 'Adductors'
    assert _group('semiten_l') == 'Hamstrings'
    assert _group('reserve_knee_angle_l') is None


def test_summary_sides_groups_and_reserve_reliance():
    t = np.linspace(0, 1, 5)
    act = pd.DataFrame({'time': t, 'vaslat_l': [0, .2, .9, .3, 0], 'vasmed_l': [0, .1, .5, .2, 0],
                        'glmax1_r': [0, .1, .2, .1, 0]})
    force = act.copy()
    force[['vaslat_l', 'vasmed_l', 'glmax1_r']] *= 1000
    force['reserve_knee_angle_l'] = [0, 5, 90, 5, 0]
    force['reserve_hip_flexion_r'] = [0, 1, 2, 1, 0]
    force['residual_pelvis_tilt'] = [0, -3, 12, 1, 0]
    id_df = pd.DataFrame({'time': t, 'knee_angle_l_moment': [0, 50, 200, 40, 0],
                          'hip_flexion_r_moment': [0, 30, 100, 20, 0]})
    s = summarize(act, force, id_df, handedness='right')     # right-handed: left leg leads
    assert s['groups']['Lead quadriceps']['peak_activation'] == 0.9
    assert s['groups']['Lead quadriceps']['peak_force_N'] == 1400   # summed per frame, not per muscle
    assert 'Trail glute max' in s['groups']
    assert s['reserves']['knee_angle_l']['share_of_id_moment'] == 0.45 > MAX_RESERVE_SHARE
    assert s['joints_not_explained_by_muscles'] == ['knee_angle_l']
    assert s['pelvis_residuals']['pelvis_tilt'] == 12.0


def test_fast_solver_meets_moments_with_least_effort_and_reserves_last():
    from muscle_forces import solve_frame
    # One joint, two identical muscles (F0 1000 N, moment arm 0.05 m) and a reserve.
    A = np.array([[50.0, 50.0, 1.0]])
    x = solve_frame(A, np.array([40.0]), n_muscles=2)
    assert np.allclose(x[:2], [0.4, 0.4], atol=1e-3)      # shared equally, as min sum a^2 says
    assert abs(x[2]) < 0.05
    # Beyond what the muscles can make, the reserve covers the rest; a stays <= 1.
    x = solve_frame(A, np.array([130.0]), n_muscles=2)
    assert np.all(x[:2] <= 1 + 1e-9) and np.isclose(A @ x, 130, atol=0.5)
    assert x[2] > 25


def test_muscle_report_withholds_on_implausible_ground_force():
    from muscle_forces import muscle_report
    out = muscle_report('unused.osim', None, {'grf_check': {'plausible': False, 'reasons': ['x']}})
    assert out['status'] == 'withheld' and 'groups' not in out


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('✓', name)
