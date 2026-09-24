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
from opensim_id import write_external_loads, summarize_id_magnitudes  # noqa: E402
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


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'✓ {fn.__name__}')
    print(f'{len(tests)} passed')
