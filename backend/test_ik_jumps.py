"""Arm IK branch-jump detection and its effect on metric reliability.
Run: python test_ik_jumps.py  (or pytest)."""
import numpy as np
import pandas as pd

from metric_evidence import build_metric_evidence, detect_ik_branch_jumps

FS = 60.0
T = np.arange(0, 2, 1 / FS)


def _kin(**cols):
    return pd.DataFrame({'time': T, **cols})


def test_fast_but_human_elbow_is_not_flagged():
    # 0 -> 150 deg in 100 ms = 1,500 deg/s mean: a fast real extension.
    elbow = np.clip((T - 1.0) * 1500.0, 0, 150)
    assert detect_ik_branch_jumps(_kin(elbow_flex_l=elbow)) == {}


def test_one_frame_snap_is_flagged_at_its_time():
    # The Quinn_1_60fps pattern: pinned at 0, then 1 -> 101 deg in one frame.
    elbow = np.where(T < 1.0, 0.0, 100.0)
    jumps = detect_ik_branch_jumps(_kin(elbow_flex_l=elbow, elbow_flex_r=0 * T + 60))
    assert list(jumps) == ['elbow_flex_l']
    assert jumps['elbow_flex_l'] == [round(T[np.argmax(T >= 1.0)], 3)]


def test_start_frame_ignores_jumps_before_the_swing():
    elbow = np.where(T < 0.5, 0.0, 100.0)
    assert detect_ik_branch_jumps(_kin(elbow_flex_l=elbow), start_frame=int(0.6 * FS)) == {}


def test_arm_rot_uses_its_higher_limit():
    # 100 deg in one frame = 6,000 deg/s: impossible at the elbow, possible at
    # the shoulder (pitching internal rotation reaches ~7,000 deg/s).
    step = np.where(T < 1.0, 0.0, 100.0)
    assert list(detect_ik_branch_jumps(_kin(arm_rot_l=step, elbow_flex_l=step))) == ['elbow_flex_l']


def _evidence(jumps, lead='l'):
    return build_metric_evidence({'metrics': {'max_hand_speed_mph': 70.0},
                                  '_rotation_context': {'ik_branch_jumps': jumps,
                                                        'lead_arm_side': lead}})


def test_lead_arm_jump_marks_lead_arm_metrics_unreliable():
    ev = _evidence({'elbow_flex_l': [4.35]})
    for m in ('peak_elbow_flex_l_deg', 'forearm_ke_J', 'torso_arm_sequence_gap_ms',
              'energy_transfer_proxy_pct'):
        assert not ev[m]['reliable'], m
        assert 'full extension' in ev[m]['caveat']
    assert ev['peak_prosup_r_deg']['reliable']
    assert ev['peak_pelvis_omega_3d_deg_s']['reliable']


def test_trail_arm_jump_leaves_lead_arm_chain_alone():
    ev = _evidence({'pro_sup_r': [4.2]}, lead='l')
    assert not ev['peak_prosup_r_deg']['reliable']
    assert ev['forearm_ke_J']['reliable']


def test_clean_capture_changes_nothing():
    assert _evidence({}) == _evidence(None)
    assert _evidence({})['forearm_ke_J']['reliable']


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in tests:
        fn(); print(f'✓ {fn.__name__}')
    print(f'{len(tests)} passed')
