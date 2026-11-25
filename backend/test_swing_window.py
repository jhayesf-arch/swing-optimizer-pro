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


def _wrists_trc(fs, n, bumps):
    """Two-wrist .trc whose speeds are Gaussian bumps: bumps = {name: [(t, v, w)]}."""
    t = np.arange(n) / fs
    cols = {'Frame': np.arange(1, n + 1), 'Time': t}
    for name in ('RWrist', 'LWrist'):
        sp = sum(v * np.exp(-0.5 * ((t - tc) / w) ** 2) for tc, v, w in bumps.get(name, []))
        cols[f'{name}_X'] = np.cumsum(sp + 0 * t) / fs
        cols[f'{name}_Y'] = np.full(n, 1.0)
        cols[f'{name}_Z'] = np.zeros(n)
    return pd.DataFrame(cols)


def test_one_wrist_burst_before_the_swing_is_not_hand_speed():
    # Talen 1007: the right wrist alone reads 26.6 mph ~0.5 s before the swing
    # while the left reads 0.6 mph. Both wrists then swing together at ~10 m/s.
    trc = _wrists_trc(FS, 300, {'RWrist': [(0.5, 12.0, 0.04), (1.0, 10.0, 0.08)],
                                'LWrist': [(1.0, 9.0, 0.08)]})
    m = _opt().calculate_trc_metrics(trc)
    assert abs(m['hand_speed_peak_time_s'] - 1.0) < 0.05, m['hand_speed_peak_time_s']
    assert 9.5 < m['max_hand_speed_mps'] < 10.5, m['max_hand_speed_mps']


def test_transition_is_the_last_pelvis_reversal_before_the_peak():
    # Load: the pelvis turns back 15 deg (0.6-0.95 s), then fires 85 deg.
    # Top of the backswing = where pelvis omega changes sign, ~0.95 s, which is
    # BEFORE swing start (a speed threshold, by definition past the top).
    n = int(2.5 * FS)
    t = np.arange(n) / FS
    pelvis = -15 * _ramp(t, 0.6, 0.35) + 85 * _ramp(t, 0.95, 0.30)
    lumbar = -25 * _ramp(t, 0.97, 0.12) + 25 * _ramp(t, 1.15, 0.20)   # trunk lags: separation grows
    r = _opt().calculate_rotational_torques_refined(_mot(pelvis, lumbar), hand_speed_peak_time_s=1.30)
    tr, ss = r['transition_frame'], r['swing_start_frame']
    assert abs(tr / FS - 0.95) < 0.03, tr / FS
    assert tr < ss < r['peak_pelvis_frame'], (tr, ss, r['peak_pelvis_frame'])
    assert r['x_factor_stretch_deg'] > 15.0, r['x_factor_stretch_deg']


def test_a_trunk_burst_after_contact_does_not_void_the_sequence():
    # Aiden 1009_6: one clean torso peak in the swing, then a tracking burst
    # ~180 ms after contact. The stability check looks at the swing (to contact
    # + 150 ms), so the burst in the follow-through must not flag the torso.
    n = int(2.5 * FS)
    t = np.arange(n) / FS
    pelvis = 85 * _ramp(t, 1.0, 0.25)
    lumbar = 30 * _ramp(t, 1.03, 0.20) + 28 * _ramp(t, 1.26, 0.06)
    r = _opt().calculate_rotational_torques_refined(_mot(pelvis, lumbar), hand_speed_peak_time_s=1.10)
    tor = r['torso_omega_abs_deg_s']
    assert tor[int(1.27 * FS):int(1.31 * FS)].max() > 0.6 * tor[int(1.0 * FS):int(1.25 * FS)].max()   # the burst is there
    assert r['torso_signal_unstable'] is False


def _upsample(df, tcol, k=2):
    """The same capture at k x the frame rate (cubic spline through every sample)."""
    from scipy.interpolate import CubicSpline
    t = df[tcol].values.astype(float)
    tn = np.linspace(t[0], t[-1], (len(t) - 1) * k + 1)
    cols = {c: tn if c == tcol else (np.arange(len(tn)) + 1) if c == 'Frame'
            else CubicSpline(t, df[c].values.astype(float))(tn) for c in df.columns}
    out = pd.DataFrame(cols)[list(df.columns)]
    out.attrs.update(df.attrs)
    return out


def test_sequence_verdict_does_not_depend_on_the_frame_rate():
    # Torso peaks 6 ms before the pelvis: inside the ~8 ms resolution limit. With
    # a 1-FRAME tolerance this was "proper / indeterminate" at 120 Hz and
    # "reversed / determinate" at 240 Hz for the same motion.
    t = np.arange(int(2.5 * FS)) / FS
    mot120 = _mot(85 * _ramp(t, 1.0, 0.25), 40 * _ramp(t, 0.981, 0.25))
    out = []
    for mot in (mot120, _upsample(mot120, 'time')):
        r = _opt().calculate_rotational_torques_refined(mot, hand_speed_peak_time_s=1.2)
        assert -7.0 < r['sequence_timing_ms'] < -5.0, r['sequence_timing_ms']
        out.append((r['proper_sequence'], r['sequence_indeterminate'], r['torso_signal_unstable']))
    assert out[0] == out[1], out


def test_real_120hz_trials_and_their_240hz_resamples_agree():
    # Ethan 1003_7 (OpenCap, 120 Hz): proper_sequence flipped True -> False and
    # the swing score 57.9 -> 52.7 when the same trial was resampled to 240 Hz.
    # Over all of Ethan's trials the sequence timing must agree to ~1 ms, and the
    # verdict must agree wherever the timing is not within 1 ms of the tolerance
    # itself (a knife-edge trial can fall either side of any threshold).
    from analyzer import SEQUENCE_TOL_S
    base = os.environ.get('SWING_DATA', '/tmp/claude-0/-home-user-swing-optimizer-pro/'
                          '715982cf-832c-5b43-aaf0-1a0a354478dd/scratchpad/cuesta/raw')
    import glob
    mots = sorted(glob.glob(os.path.join(base, 'Ethan', '*.mot')))
    if not mots:
        print('skip: real trials not available')
        return
    import main
    tol_ms = SEQUENCE_TOL_S * 1000.0
    for mot in mots:
        o = _opt()
        k0, t0 = o.load_mot_file(mot), o.load_trc_file(mot[:-4] + '.trc')
        res = []
        for k, tr in ((k0, t0), (_upsample(k0, 'time'), _upsample(t0, 'Time'))):
            k, tr = main._crop_to_swing(k.copy(), tr.copy())
            d = RefinedHittingOptimizer(94, 1.84, skill_level='college', handedness='right') \
                .comprehensive_diagnosis(k, 'x.mot', trc_data=tr)
            res.append((bool(d['metrics']['proper_sequence']), d['metrics']['sequence_timing_ms'],
                        d['swing_score']))
        (ok0, seq0, sc0), (ok1, seq1, sc1) = res
        name = os.path.basename(mot)
        assert abs(seq0 - seq1) < 1.5, (name, res)
        if min(abs(abs(seq0) - tol_ms), abs(abs(seq1) - tol_ms)) > 1.0:
            assert ok0 == ok1, (name, res)
            assert abs(sc0 - sc1) < 3.0, (name, res)


if __name__ == '__main__':
    for _n in sorted(dir()):
        if _n.startswith('test_'):
            globals()[_n]()
            print('PASS', _n)
