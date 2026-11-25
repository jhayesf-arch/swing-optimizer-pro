"""Regression tests for the ground-force and inverse-dynamics chain.

Runs with pytest, or directly: python backend/test_dynamics.py

The static-stance test exists because the bottom-up chain shipped with body
weight missing from its GRF and the sign of gravity flipped on every segment.
Real-trial numbers still looked plausible, so nothing caught it. A motionless
athlete has an answer known without any biomechanics literature, which makes it
the cheapest possible guard against the same class of bug.
"""
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grf_estimation import estimate_grf                     # noqa: E402
from bottom_up_id import _leg_id, split_grf_by_foot, SEG, segment_params_from_osim  # noqa: E402
from opensim_id import (write_external_loads, summarize_id_magnitudes,  # noqa: E402
                        required_ground_force, check_required_grf, consistent_foot_loads)
from dynamics_compare import build_comparison, reference_stem         # noqa: E402

G = 9.81
M, H, N, FS = 75.0, 1.78, 120, 60.0

# Motionless athlete, Y up, feet level and 0.3 m apart.
_POSE = {
    'Neck': (0, 1.50, 0), 'midHip': (0, 0.95, 0),
    'RShoulder': (0, 1.45, 0.18), 'LShoulder': (0, 1.45, -0.18),
    'RElbow': (0, 1.15, 0.20), 'LElbow': (0, 1.15, -0.20),
    'RWrist': (0, 0.90, 0.20), 'LWrist': (0, 0.90, -0.20),
    'RHip': (0, 0.95, 0.10), 'LHip': (0, 0.95, -0.10),
    'RKnee': (0, 0.52, 0.15), 'LKnee': (0, 0.52, -0.15),
    'RAnkle': (0, 0.08, 0.15), 'LAnkle': (0, 0.08, -0.15),
    'RHeel': (-0.05, 0.03, 0.15), 'LHeel': (-0.05, 0.03, -0.15),
    'RBigToe': (0.18, 0.02, 0.15), 'LBigToe': (0.18, 0.02, -0.15),
}


def _static_trc():
    t = np.arange(N) / FS
    cols = {'Frame': np.arange(N), 'Time': t}
    for k, (x, y, z) in _POSE.items():
        cols[f'{k}_X'], cols[f'{k}_Y'], cols[f'{k}_Z'] = (np.full(N, x), np.full(N, y), np.full(N, z))
    return pd.DataFrame(cols), t


def test_static_grf_equals_body_weight():
    trc, _ = _static_trc()
    grf = estimate_grf(trc, M)['grf_total'][N // 2]
    assert abs(grf[1] - M * G) < 0.5, grf
    assert abs(grf[0]) < 0.5 and abs(grf[2]) < 0.5, grf


def test_static_ankle_carries_half_body_weight_minus_foot():
    trc, t = _static_trc()
    grf = estimate_grf(trc, M)['grf_total']
    ank = lambda side: np.array([_POSE[f'{side}Ankle']] * N)
    split = split_grf_by_foot(grf, ank('L'), ank('R'), t, t)
    leg = _leg_id('L', trc, split['F_grf_l'], M, H, FS, 1 / FS)
    expected = -(M * G / 2 - M * SEG['foot']['mass_pct'] * G)   # shank pushes DOWN on foot
    assert abs(leg['F_ankle'][N // 2][1] - expected) < 0.5, leg['F_ankle'][N // 2]


# A minimal model with the same structure as OpenCap's LaiUhlrich2022_scaled:
# femur origin at the hip, knee 0.48 m below it; tibia origin at the knee,
# ankle 0.49 m below. Thigh CoM at 0.20 m -> 0.4167; shank CoM at 0.23 -> 0.4694.
_OSIM = """<OpenSimDocument Version="40000"><Model name="t"><BodySet><objects>
<Body name="pelvis"><mass>11</mass><mass_center>0 0 0</mass_center></Body>
{bodies}</objects></BodySet><JointSet><objects>{joints}</objects></JointSet></Model></OpenSimDocument>"""


def _osim_text():
    bodies, joints = '', ''
    for s in 'rl':
        bodies += (f'<Body name="femur_{s}"><mass>10</mass><mass_center>0 -0.20 0</mass_center></Body>'
                   f'<Body name="tibia_{s}"><mass>4</mass><mass_center>0 -0.23 0</mass_center></Body>'
                   f'<Body name="talus_{s}"><mass>0.1</mass><mass_center>0 0 0</mass_center></Body>'
                   f'<Body name="calcn_{s}"><mass>1.4</mass><mass_center>0.1 0 0</mass_center></Body>'
                   f'<Body name="toes_{s}"><mass>0.25</mass><mass_center>0 0 0</mass_center></Body>')
        for name, a, ta, b, tb in ((f'hip_{s}', 'pelvis', '0 0 0', f'femur_{s}', '0 0 0'),
                                   (f'knee_{s}', f'femur_{s}', '0 -0.48 0', f'tibia_{s}', '0 0 0'),
                                   (f'ankle_{s}', f'tibia_{s}', '0 -0.49 0', f'talus_{s}', '0 0 0')):
            joints += (f'<CustomJoint name="{name}"><frames>'
                       f'<PhysicalOffsetFrame name="p"><socket_parent>/bodyset/{a}</socket_parent><translation>{ta}</translation></PhysicalOffsetFrame>'
                       f'<PhysicalOffsetFrame name="c"><socket_parent>/bodyset/{b}</socket_parent><translation>{tb}</translation></PhysicalOffsetFrame>'
                       f'</frames></CustomJoint>')
    return _OSIM.format(bodies=bodies, joints=joints)


def test_segment_params_from_osim():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.osim')
        open(path, 'w').write(_osim_text())
        p = segment_params_from_osim(path)
    assert abs(p['mass']['thigh_l'] - 10) < 1e-9 and abs(p['mass']['foot_r'] - 1.75) < 1e-9
    assert abs(p['com_frac']['thigh_r'] - 0.20 / 0.48) < 1e-9
    assert abs(p['com_frac']['shank_l'] - 0.23 / 0.49) < 1e-9
    assert abs(p['total_mass_kg'] - (11 + 2 * 15.75)) < 1e-9
    assert segment_params_from_osim(__file__) is None     # not a model -> fall back


def test_model_masses_reach_the_chain():
    """With a model, the ankle load uses the model's foot mass, not de Leva's."""
    trc, t = _static_trc()
    grf = estimate_grf(trc, M)['grf_total']
    ank = lambda side: np.array([_POSE[f'{side}Ankle']] * N)
    split = split_grf_by_foot(grf, ank('L'), ank('R'), t, t)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'm.osim')
        open(path, 'w').write(_osim_text())
        params = segment_params_from_osim(path)
    leg = _leg_id('L', trc, split['F_grf_l'], M, H, FS, 1 / FS, seg_params=params)
    expected = -(M * G / 2 - 1.75 * G)
    assert abs(leg['F_ankle'][N // 2][1] - expected) < 0.5, leg['F_ankle'][N // 2]


def test_external_loads_files_are_consistent():
    n = 10
    loads = {'time': np.linspace(0, 1, n),
             'grf_l': np.ones((n, 3)), 'grf_r': np.ones((n, 3)),
             'cop_l': np.zeros((n, 3)), 'cop_r': np.zeros((n, 3))}
    with tempfile.TemporaryDirectory() as d:
        xml = write_external_loads(loads, d)
        forces = list(ET.parse(xml).getroot().iter('ExternalForce'))
        assert {f.findtext('applied_to_body') for f in forces} == {'calcn_r', 'calcn_l'}
        lines = open(os.path.join(d, 'external_loads.mot')).read().splitlines()
        header = lines[lines.index('endheader') + 1].split('\t')
        for f in forces:
            for ident in ('force_identifier', 'point_identifier'):
                for ax in 'xyz':
                    assert f.findtext(ident) + ax in header
        assert f'nRows={n}' in lines


def test_opensim_magnitudes():
    df = pd.DataFrame({'time': [0, 1], 'hip_flexion_l_moment': [30, 0],
                       'hip_adduction_l_moment': [40, 0], 'knee_angle_l_moment': [-50, 10],
                       'pelvis_tx_force': [3, 0], 'pelvis_ty_force': [4, 0]})
    s = summarize_id_magnitudes(df, 45.0)
    assert s['hip_moment_l_Nm'] == 50.0          # |(30, 40)|
    assert s['knee_moment_l_Nm'] == 50.0
    assert s['pelvis_residual_force_N'] == 5.0   # |(3, 4)|
    assert s['hip_moment_r_Nm'] == 0.0           # missing columns → 0, not a crash


def test_comparison_needs_two_sources_and_maps_lead():
    metrics = {'peak_hip_moment_id_l_Nm': 300.0, 'peak_hip_moment_id_r_Nm': 100.0}
    assert build_comparison(metrics) is None
    ref = {'label': 'plates', 'peaks': {'hip_moment_l_Nm': 250.0}}
    cmp = build_comparison(metrics, reference=ref, handedness='right')
    first = cmp['rows'][0]
    assert first['quantity'].startswith('Lead') and first['key'] == 'hip_moment_l_Nm'
    assert first['delta_pct'] == 20.0


def test_reference_stem_pairs_common_suffixes():
    for name in ('Emilio_1_10_cortex.json', 'Emilio_1_10_ID.sto', 'emilio_1_10.json'):
        assert reference_stem(name) == 'emilio_1_10'


def test_required_grf_reads_pelvis_translation_residuals():
    df = pd.DataFrame({'time': [0, 1], 'pelvis_tx_force': [1., 2.],
                       'pelvis_ty_force': [700., 710.], 'pelvis_tz_force': [3., 4.]})
    assert required_ground_force(df).tolist() == [[1, 700, 3], [2, 710, 4]]
    # A planar model has no sideways translation: no 3-D check is possible.
    assert required_ground_force(df.drop(columns='pelvis_tz_force')) is None


def test_grf_check_passes_standing_and_rejects_impossible_forces():
    W = M * G
    standing = np.tile([0.0, W, 0.0], (10, 1))
    assert check_required_grf(standing, W)['plausible']
    # The ground pushes; it cannot pull. Nor can a batter push 2 BW sideways.
    pulled = standing.copy(); pulled[4, 1] = -0.5 * W
    sideways = standing.copy(); sideways[4, 0] = 2.0 * W
    for F in (pulled, sideways):
        chk = check_required_grf(F, W)
        assert not chk['plausible'] and chk['reasons']


def test_consistent_loads_sum_to_required_and_follow_estimated_split():
    t = np.linspace(0, 1, 5)
    est = {'time': t,
           'grf_l': np.tile([0, 300.0, 0], (5, 1)), 'grf_r': np.tile([0, 100.0, 0], (5, 1)),
           'cop_l': np.tile([0.1, 0, -0.2], (5, 1)), 'cop_r': np.tile([0.1, 0, 0.2], (5, 1))}
    req = np.tile([40.0, 800.0, -20.0], (5, 1))
    out = consistent_foot_loads(est, t, req)
    assert np.allclose(out['grf_l'] + out['grf_r'], req)
    assert np.allclose(out['grf_l'][:, 1], 600.0)          # 75 % share, as estimated
    assert np.allclose(out['cop_r'], est['cop_r'])
    # No vertical force in the estimate (both feet unloaded): split evenly.
    est['grf_l'][:] = 0; est['grf_r'][:] = 0
    out = consistent_foot_loads(est, t, req)
    assert np.allclose(out['grf_l'], req / 2)


_DATA = os.environ.get('SWING_DATA', '/tmp/claude-0/-home-user-swing-optimizer-pro/'
                       '715982cf-832c-5b43-aaf0-1a0a354478dd/scratchpad/cuesta/raw')


def test_dynamics_withheld_on_unusable_capture_or_slow_pelvis():
    from analyzer import RefinedHittingOptimizer
    o = RefinedHittingOptimizer(80, 1.83, skill_level='college')
    good = {'usable': True}
    assert o._dynamics_gate(good, {'peak_pelvis_omega_3d_deg_s': 650.0}) is None
    r = o._dynamics_gate({'usable': False, 'reasons': ['very little pelvis rotation']},
                         {'peak_pelvis_omega_3d_deg_s': 650.0})
    assert r and 'not usable' in r, r
    r = o._dynamics_gate(good, {'peak_pelvis_omega_3d_deg_s': 240.0})
    assert r and '240' in r and '300' in r, r


def test_opensim_id_is_not_run_when_the_capture_gate_withholds_dynamics():
    import main
    called = []
    saved = main.HAS_OPENSIM_ID, main.run_inverse_dynamics
    main.HAS_OPENSIM_ID = True
    main.run_inverse_dynamics = lambda *a, **k: called.append(1) or {}
    try:
        reason = 'Joint loads withheld: this capture is not usable as a swing (check swing).'
        diag = {'metrics': {'peak_hip_moment_id_l_Nm': 0.0},
                'joint_loads_status': {'withheld': True, 'capture_gate': True, 'reason': reason}}
        out = main._run_id('x.mot', os.path.abspath(__file__), 0.88, 0.86, diag)
    finally:
        main.HAS_OPENSIM_ID, main.run_inverse_dynamics = saved
    assert out is None and not called
    st = diag['opensim_id_status']
    assert st['ran'] is False and st['reliable'] is False and st['reason'] == reason
    assert 'opensim_id' not in diag and diag['muscle_loads']['status'] == 'withheld'


def test_practice_move_gets_no_joint_loads():
    # Eric 1004_120fps_2 is a practice move (pelvis ~34 deg/s, capture unusable),
    # and it was the one trial of 132 whose joint loads were shown.
    mot = os.path.join(_DATA, 'Eric', '1004_120fps_2.mot')
    if not os.path.exists(mot):
        print('skip: real trial not available')
        return
    import main
    from analyzer import RefinedHittingOptimizer
    o = RefinedHittingOptimizer(86, 1.833, skill_level='college', handedness='right')
    k, t = main._crop_to_swing(o.load_mot_file(mot), o.load_trc_file(mot[:-4] + '.trc'))
    d = o.comprehensive_diagnosis(k, 'x.mot', trc_data=t,
                                  model_path=os.path.join(_DATA, 'Eric', 'LaiUhlrich2022_shoulder_scaled.osim'))
    assert d['capture_quality']['usable'] is False
    jl = d['joint_loads_status']
    assert jl.get('withheld') is True and jl.get('capture_gate') is True, jl
    assert d['metrics']['peak_hip_moment_id_l_Nm'] == 0.0


def test_bottom_up_peaks_are_taken_over_the_id_window():
    # Bottom-up took its peaks from swing start to the end of the crop (~1 s after
    # contact) while OpenSim ID covers onset - 0.1 s to + 0.5 s, so the comparison
    # table set mismatched windows side by side (Keller_1 lead hip 644 vs 435 N m).
    from bottom_up_id import bottom_up_lower_body
    trc, t = _static_trc()
    grf = estimate_grf(trc, M)['grf_total'].copy()
    grf[110:115, 1] *= 3.0                       # a late spike, after the ID window
    mot = pd.DataFrame({'time': t})
    win = (t[20], t[90])
    out = bottom_up_lower_body(trc, mot, grf_ts=grf, grf_time=t, body_mass_kg=M,
                               body_height_m=H, window_s=win)
    assert abs(out['peak_grf_vert_l_N'] - M * G / 2) < 5.0, out['peak_grf_vert_l_N']
    whole = bottom_up_lower_body(trc, mot, grf_ts=grf, grf_time=t, body_mass_kg=M, body_height_m=H)
    assert whole['peak_grf_vert_l_N'] > 1.4 * M * G / 2          # without a window the spike counts


def test_opensim_trim_uses_the_given_window():
    from opensim_id import _trim_to_swing_window
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, 'a.mot'), os.path.join(d, 'b.mot')
        t = np.arange(240) / 120.0
        rows = '\n'.join(f'{x:.6f}\t{30 * np.sin(4 * x):.4f}' for x in t)
        open(src, 'w').write('a\nversion=1\nnRows=240\nnColumns=2\ninDegrees=yes\nendheader\n'
                             'time\tpelvis_rotation\n' + rows + '\n')
        t0, t1 = _trim_to_swing_window(src, dst, window=(0.5, 1.1))
        assert abs(t0 - 0.5) < 1e-6 and abs(t1 - 1.1) < 1e-6, (t0, t1)


def test_run_id_hands_the_analyzer_window_to_opensim():
    import main
    seen = {}
    saved = main.HAS_OPENSIM_ID, main.run_inverse_dynamics
    main.HAS_OPENSIM_ID = True
    main.run_inverse_dynamics = lambda *a, **k: seen.update(k) or {'external_loads_applied': False}
    try:
        diag = {'metrics': {}, 'joint_loads_status': {}, 'dynamics_window_s': (1.2, 1.8)}
        main._run_id('x.mot', os.path.abspath(__file__), 0.88, 0.86, diag, time_range=(0.0, 3.0))
    finally:
        main.HAS_OPENSIM_ID, main.run_inverse_dynamics = saved
    assert seen.get('window') == (1.2, 1.8), seen


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'✓ {fn.__name__}')
    print(f'{len(tests)} passed')
