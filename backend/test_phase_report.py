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


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('✓', name)
