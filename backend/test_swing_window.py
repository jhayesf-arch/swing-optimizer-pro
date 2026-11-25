"""Swing-window, sequencing and signal-hygiene regressions (analyzer.py).

Run: python -c "import test_swing_window as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]"

Synthetic captures with a known answer, so the checks need no capture file.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import RefinedHittingOptimizer, assess_capture_quality  # noqa: E402

FS = 120.0


def _ramp(t, t0, w):
    return 0.5 * (1 - np.cos(np.clip((t - t0) / w, 0, 1) * np.pi))


def _mot(pelvis_deg, lumbar_deg, fs=FS, extra=None):
    n = len(pelvis_deg)
    df = pd.DataFrame({'time': np.arange(n) / fs, 'pelvis_rotation': pelvis_deg,
                       'lumbar_rotation': lumbar_deg,
                       'pelvis_tilt': np.zeros(n), 'pelvis_list': np.zeros(n),
                       'pelvis_tx': np.zeros(n), 'pelvis_ty': np.full(n, 0.95), 'pelvis_tz': np.zeros(n),
                       'lumbar_bending': np.zeros(n), 'lumbar_extension': np.zeros(n)})
    for c in ('arm_flex_r', 'arm_flex_l', 'elbow_flex_r', 'elbow_flex_l', 'arm_add_r', 'arm_add_l',
              'arm_rot_r', 'arm_rot_l', 'hip_flexion_r', 'hip_flexion_l', 'knee_angle_r', 'knee_angle_l',
              'ankle_angle_r', 'ankle_angle_l'):
        df[c] = 0.0
    for k, v in (extra or {}).items():
        df[k] = v
    return df


def _opt():
    return RefinedHittingOptimizer(81, 1.83, skill_level='college', handedness='right')


def test_pelvis_peak_is_the_swing_near_the_hands_not_a_later_unwind():
    # Swing: pelvis opens 70 deg over 0.25 s from t=1.0 s; hands at contact 1.25 s.
    # Then, 0.7 s later, a faster reverse unwind (-120 deg in 0.15 s) while walking off.
    n = int(3.0 * FS)
    t = np.arange(n) / FS
    pelvis = 70 * _ramp(t, 1.0, 0.25) - 120 * _ramp(t, 1.95, 0.15)
    lumbar = 20 * _ramp(t, 1.05, 0.22)
    r = _opt().calculate_rotational_torques_refined(_mot(pelvis, lumbar), hand_speed_peak_time_s=1.25)
    pk_t = r['peak_pelvis_frame'] / FS
    assert 1.05 < pk_t < 1.25, pk_t                                  # true peak ~1.125 s
    # 70 deg over 0.25 s (cosine ramp) peaks at 70*pi/(2*0.25) = 440 deg/s
    assert 380 < r['peak_pelvis_omega_3d_deg_s'] < 470, r['peak_pelvis_omega_3d_deg_s']
    assert r['swing_start_frame'] < r['contact_frame'], (r['swing_start_frame'], r['contact_frame'])
    assert abs(r['contact_frame'] / FS - 1.25) < 0.01


def test_contact_before_swing_start_marks_the_capture_unusable():
    # Hands peak at 1.25 s, but the pelvis only starts turning after that: the
    # segmentation cannot be trusted, so the capture is unusable (not clamped).
    n = int(3.0 * FS)
    t = np.arange(n) / FS
    pelvis = 60 * _ramp(t, 1.28, 0.12)
    lumbar = 15 * _ramp(t, 1.30, 0.12)
    r = _opt().calculate_rotational_torques_refined(_mot(pelvis, lumbar), hand_speed_peak_time_s=1.25)
    assert r['contact_frame'] != r['swing_start_frame']
    assert r['contact_before_swing_start'] is True
    cq = assess_capture_quality(r, {}, has_markers=True, dt=1 / FS)
    assert cq['usable'] is False, cq
    assert any('contact' in s for s in cq['reasons']), cq['reasons']


if __name__ == '__main__':
    for _n in sorted(dir()):
        if _n.startswith('test_'):
            globals()[_n]()
            print('PASS', _n)
