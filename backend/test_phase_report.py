"""Regression tests for the 2026-11-16 verification fixes (analyzer.py).

Run: python backend/test_phase_report.py

Each test builds a synthetic .mot with a known answer, so the checks hold
without any capture file.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import RefinedHittingOptimizer, PUBLISHED_NORMS  # noqa: E402

FS = 120.0


def _mot(pelvis_deg, lumbar_deg, extra=None):
    n = len(pelvis_deg)
    t = np.arange(n) / FS
    df = pd.DataFrame({'time': t, 'pelvis_rotation': pelvis_deg, 'lumbar_rotation': lumbar_deg,
                       'pelvis_tilt': np.zeros(n), 'pelvis_list': np.zeros(n),
                       'pelvis_tx': np.zeros(n), 'pelvis_ty': np.full(n, 0.95), 'pelvis_tz': np.zeros(n),
                       'lumbar_bending': np.zeros(n), 'lumbar_extension': np.zeros(n)})
    for c in ('arm_flex_r', 'arm_flex_l', 'elbow_flex_r', 'elbow_flex_l', 'arm_add_r', 'arm_add_l',
              'arm_rot_r', 'arm_rot_l', 'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
              'ankle_angle_r', 'ankle_angle_l'):
        df[c] = 0.0
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def _swing(n=360, pelvis_amp=60.0, lumbar_amp=20.0):
    """Pelvis turns 0 -> pelvis_amp over 0.3 s starting at 1.0 s; lumbar twists
    to lumbar_amp slightly later, so the absolute torso rate exceeds the pelvis."""
    t = np.arange(n) / FS
    ramp = lambda t0, w: 0.5 * (1 - np.cos(np.clip((t - t0) / w, 0, 1) * np.pi))
    pelvis = pelvis_amp * ramp(1.0, 0.30)
    lumbar = lumbar_amp * ramp(1.05, 0.25)
    return pelvis, lumbar


def test_torso_to_pelvis_ratio_uses_absolute_torso_rate():
    pelvis, lumbar = _swing()
    o = RefinedHittingOptimizer(81, 1.83, skill_level='college')
    r = o.calculate_rotational_torques_refined(_mot(pelvis, lumbar))
    # Absolute torso rate = pelvis + lumbar, so the ratio must exceed 1; the old
    # code divided the lumbar twist rate alone by the pelvis rate (~0.35 here).
    assert r['torso_to_pelvis_rot_ratio'] > 1.05, r['torso_to_pelvis_rot_ratio']
    assert np.degrees(r['peak_shoulder_omega_rad_s']) > np.degrees(r['peak_pelvis_omega_rad_s'])


def test_lumbar_joint_limit_flip_does_not_reach_torso_signals():
    pelvis, lumbar = _swing()
    flipped = lumbar.copy()
    flipped[150:156] = 89.9           # IK pinned at the +90 deg stop, then snaps back
    o = RefinedHittingOptimizer(81, 1.83, skill_level='college')
    clean = o.calculate_rotational_torques_refined(_mot(pelvis, lumbar))
    bad = o.calculate_rotational_torques_refined(_mot(pelvis, flipped))
    # Repaired torso series must not carry the snap (which would read >2000 deg/s).
    assert bad['torso_omega_abs_deg_s'].max() < 1.5 * clean['torso_omega_abs_deg_s'].max()


def test_direction_at_contact_is_measured_at_contact_from_the_top():
    # Pelvis turns exactly 90 deg from the top to contact: square, 0 deg off.
    pelvis, lumbar = _swing(pelvis_amp=90.0, lumbar_amp=0.0)
    o = RefinedHittingOptimizer(81, 1.83, skill_level='college')
    rot = o.calculate_rotational_torques_refined(_mot(pelvis, lumbar))
    rot['contact_frame'] = int(np.argmax(pelvis >= 89.9))   # contact when the turn completes
    stride = {'plant_frame': int(np.argmax(pelvis >= 5.0)), 'stride_ratio': 0.48, 'stride_length_m': 0.88,
              'stride_efficiency_pct': 100.0, 'stride_detected': True}
    rep = o.build_phase_report(rot, stride, {}, {})
    d = rep['dimensions']['pelvis_direction_at_contact']
    assert d.get('available') is not False and d['value'] < 3.0, d
    # At the plant frame the hips had turned ~5 deg; the old code scored that
    # (85 deg off-square) as contact.
    assert d['stars'] == 5


def test_direction_at_contact_unavailable_without_contact_frame():
    o = RefinedHittingOptimizer(81, 1.83, skill_level='college')
    rep = o.build_phase_report(None, None, {}, {})
    for k in ('pelvis_direction_at_contact', 'upper_torso_direction_at_contact', 'lead_leg_block'):
        d = rep['dimensions'][k]
        assert d['available'] is False and d['stars'] is None and d['value'] is None, (k, d)


def test_missing_knee_data_does_not_score_as_a_zero_block():
    pelvis, lumbar = _swing()
    o = RefinedHittingOptimizer(81, 1.83, skill_level='college')
    rot = o.calculate_rotational_torques_refined(_mot(pelvis, lumbar))
    stride = {'plant_frame': 110, 'stride_ratio': 0.48, 'stride_length_m': 0.88,
              'stride_efficiency_pct': 100.0, 'stride_detected': True}
    rep = o.build_phase_report(rot, stride, {}, lower_body={})
    d = rep['dimensions']['lead_leg_block']
    assert d['available'] is False
    assert all(p['key'] != 'lead_leg_block' for p in rep.get('priorities', []))


def test_percentile_at_published_mean_is_fiftieth():
    o = RefinedHittingOptimizer(81, 1.83, skill_level='college')
    for key, (mean, sd) in ((k, v['college']) for k, v in PUBLISHED_NORMS.items()):
        assert o._percentile_for(key, mean) == 50.0, key
        assert o._percentile_for(key, mean - sd) == 16.0, key
        assert o._percentile_for(key, mean + 2 * sd) == 98.0, key
    # Relative-benchmark tables keep the anchor interpolation.
    assert o._percentile_for('max_hip_shoulder_separation', 46) > 50


def _trc_df(fs, n, ankle_lift_frames=None, units_scale=1.0):
    """Minimal .trc DataFrame: standing body, left ankle lifts and lands."""
    t = np.arange(n) / fs
    cols = {'Frame': np.arange(1, n + 1), 'Time': t}
    pose = {'Neck': (0, 1.5, 0), 'midHip': (0, 0.95, 0), 'RShoulder': (0, 1.45, .18), 'LShoulder': (0, 1.45, -.18),
            'RElbow': (0, 1.15, .2), 'LElbow': (0, 1.15, -.2), 'RWrist': (0, .9, .2), 'LWrist': (0, .9, -.2),
            'RHip': (0, .95, .1), 'LHip': (0, .95, -.1), 'RKnee': (0, .52, .15), 'LKnee': (0, .52, -.15),
            'RAnkle': (0, .08, .15), 'LAnkle': (0, .08, -.45), 'RBigToe': (.18, .02, .15), 'LBigToe': (.18, .02, -.45)}
    for m, (x, y, z) in pose.items():
        for ax, v in zip('XYZ', (x, y, z)):
            cols[f'{m}_{ax}'] = np.full(n, float(v)) * units_scale
    if ankle_lift_frames:
        a, b = ankle_lift_frames
        lift = 0.10 * np.sin(np.linspace(0, np.pi, b - a))
        cols['LAnkle_Y'][a:b] += lift * units_scale
    return pd.DataFrame(cols)


def test_trc_in_millimetres_is_scaled_to_metres(tmp_path=None):
    import tempfile
    o = RefinedHittingOptimizer(81, 1.83)
    n = 24
    body = '\n'.join('\t'.join(['%d' % (i + 1), '%.4f' % (i / 120)] + ['1000.0', '950.0', '0.0']) for i in range(n))
    for unit, expect in (('mm', 1.0), ('m', 1000.0)):
        txt = ('PathFileType\t4\t(X/Y/Z)\tswing_time.trc\n'
               'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n'
               f'120.0\t120.0\t{n}\t1\t{unit}\t120.0\t1\t{n}\n'
               'Frame#\tTime\tmidHip\t\t\t\n\t\tX1\tY1\tZ1\n' + body + '\n')
        with tempfile.NamedTemporaryFile('w', suffix='.trc', delete=False) as f:
            f.write(txt); path = f.name
        df = o.load_trc_file(path)
        os.unlink(path)
        assert df is not None, 'header line containing "time" in the file name must not break parsing'
        assert abs(float(df['midHip_X'].iloc[0]) - expect) < 1e-9, (unit, df['midHip_X'].iloc[0])


def test_foot_plant_maps_between_trc_and_mot_rates():
    # .trc at 120 Hz, .mot at 60 Hz: the lead ankle lifts over trc rows 120-180
    # (1.0-1.5 s) and lands at 1.5 s, which is .mot frame 90, not 180.
    trc = _trc_df(120.0, 360, ankle_lift_frames=(120, 180))
    o = RefinedHittingOptimizer(81, 1.83)
    o.handedness = 'right'
    mot_time = np.arange(180) / 60.0
    plant = o._lead_foot_touchdown(trc, peak_frame=120, fs=60.0, mot_time=mot_time)
    assert plant is not None and abs(mot_time[plant] - 1.5) < 0.06, plant


def test_stride_from_markers_uses_time_not_frame_numbers():
    trc = _trc_df(120.0, 360)
    o = RefinedHittingOptimizer(81, 1.83)
    pelvis, lumbar = _swing(n=180)        # 60 Hz .mot: 3 s of motion
    mot = _mot(pelvis, lumbar); mot['time'] = np.arange(180) / 60.0
    rot = o.calculate_rotational_torques_refined(mot)
    out = o._stride_from_markers(trc, mot, rot)
    assert out and out['stride_detected'] and abs(out['stride_length_m'] - 0.60) < 0.01, out


def test_com_drops_missing_segments_and_refuses_without_trunk():
    from grf_estimation import compute_com
    trc = _trc_df(60.0, 10)
    full = compute_com(trc)
    no_hand = compute_com(trc.drop(columns=['RWrist_X', 'RWrist_Y', 'RWrist_Z']))
    # One hand is 0.6% of body mass: the centre barely moves, and is not pulled
    # toward the origin as it was when a missing marker read (0, 0, 0).
    assert abs(no_hand[0, 1] - full[0, 1]) < 0.01
    try:
        compute_com(trc.drop(columns=['Neck_X', 'Neck_Y', 'Neck_Z']))
        assert False, 'trunk missing must raise'
    except ValueError:
        pass


def _balance_case(neck_shift_m=0.10, com_shift_m=0.0, finish_drift_m=0.0):
    """120 Hz .trc + .mot. Plant at 1.0 s, contact at 1.3 s. The neck moves
    neck_shift_m forward between plant and contact; the whole body is shifted
    com_shift_m toward the lead foot; after 1.7 s the body drifts finish_drift_m."""
    fs, n = 120.0, 360
    trc = _trc_df(fs, n)
    t = trc['Time'].values
    # Smooth (cosine) ramp: real head motion has no instantaneous corners.
    ramp = 0.5 * (1 - np.cos(np.clip((t - 1.0) / 0.3, 0, 1) * np.pi))
    trc['Neck_X'] = trc['Neck_X'] + neck_shift_m * ramp
    upper = [c for c in trc.columns if c.split('_')[0] in ('Neck', 'midHip', 'RShoulder', 'LShoulder', 'RElbow',
             'LElbow', 'RWrist', 'LWrist', 'RHip', 'LHip') and c.endswith('_Z')]
    for c in upper:
        trc[c] = trc[c] - com_shift_m          # lead (left) ankle sits at z = -0.45
    drift = np.clip((t - 1.7) / 0.3, 0, 1) * finish_drift_m
    for c in [c for c in trc.columns if c.endswith('_X') and not c.startswith(('LAnkle', 'RAnkle', 'LBigToe', 'RBigToe'))]:
        trc[c] = trc[c] + drift
    pelvis, lumbar = _swing(n=n)
    mot = _mot(pelvis, lumbar)
    o = RefinedHittingOptimizer(81, 1.83, skill_level='college')
    o.handedness = 'right'
    rot = {'contact_frame': int(1.3 * fs), 'pelvis_omega': np.zeros(n)}
    stride = {'plant_frame': int(1.0 * fs)}
    return o, o.calculate_balance(trc, mot, rot, stride)


def test_head_movement_measures_neck_travel_plant_to_contact():
    _, b = _balance_case(neck_shift_m=0.10)
    assert abs(b['head_movement_cm'] - 10.0) < 0.6, b['head_movement_cm']
    _, b = _balance_case(neck_shift_m=0.0)
    assert b['head_movement_cm'] < 0.5


def test_com_over_base_sign_and_scale():
    # Base: rear ankle z=+0.15, lead ankle z=-0.45 (60 cm). The centred body
    # sits slightly toward the rear foot; shifting the upper body 12 cm toward
    # the lead foot moves the centre of mass forward along the base.
    _, centred = _balance_case()
    _, forward = _balance_case(com_shift_m=0.12)
    assert forward['com_offset_pct'] > centred['com_offset_pct'] + 10
    assert abs(forward['base_width_m'] - 0.60) < 0.01


def test_finish_balance_reads_drift_after_contact():
    _, still = _balance_case()
    _, drifting = _balance_case(finish_drift_m=0.20)
    assert still['finish_travel_cm'] < 1.0
    assert drifting['finish_travel_cm'] > 8.0


def test_balance_phase_holds_the_three_measures_and_coil_moved_to_stride():
    from analyzer import SWING_PHASES
    assert SWING_PHASES['balance']['label'] == 'Balance & Head Movement'
    assert SWING_PHASES['balance']['dimensions'] == ['head_movement', 'com_over_base', 'finish_balance']
    assert 'torso_coil_at_plant' in SWING_PHASES['stride']['dimensions']
    o, b = _balance_case(neck_shift_m=0.10)
    rep = o.build_phase_report(None, None, {}, {}, balance=b)
    d = rep['dimensions']
    assert d['head_movement']['value'] == round(b['head_movement_cm'], 1) and d['head_movement']['stars'] == 3
    # Not in the Swing Score.
    sc = {c['key'] for c in rep['swing_score_detail']['components']}
    assert not sc & {'head_movement', 'com_over_base', 'finish_balance'}
    # Without markers all three say unavailable.
    rep = o.build_phase_report(None, None, {}, {})
    assert all(rep['dimensions'][k]['available'] is False for k in ('head_movement', 'com_over_base', 'finish_balance'))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('✓', name)
