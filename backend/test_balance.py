"""Balance & Head Movement: head travel, centre of mass between the feet, finish.
Run: python -m pytest test_balance.py"""
import numpy as np
import pandas as pd

from analyzer import RefinedHittingOptimizer, SWING_PHASES
from swing_score import INFORMATIONAL

FS = 100.0
N = 200            # 2 s; contact at 1.0 s
CONTACT = 100
H = 1.80


def body(shift=lambda t: 0.0, head=lambda t: 0.0, lead_step=lambda t: 0.0, drop=lambda t: 0.0):
    """Synthetic right-handed hitter, Y up, stride along +X (lead = left foot at +X).

    shift(t): whole upper body moved along +X (m) - moves the centre of mass.
    head(t):  extra forward travel of the upper trunk only (m).
    lead_step(t): lead foot moved along +X (m).
    """
    t = np.arange(N) / FS
    cols = {'Frame': np.arange(N), 'Time': t}

    def put(name, x, y, z):
        cols[f'{name}_X'] = np.broadcast_to(x, (N,)).astype(float)
        cols[f'{name}_Y'] = np.broadcast_to(y, (N,)).astype(float)
        cols[f'{name}_Z'] = np.broadcast_to(z, (N,)).astype(float)

    sx = np.array([shift(v) for v in t])
    hx = np.array([head(v) for v in t])
    dy = np.array([drop(v) for v in t])
    lx = np.array([lead_step(v) for v in t])
    # feet: back (right) at x=0, lead (left) at x=0.8
    put('RAnkle', 0.0, 0.08, 0.0); put('RBigToe', 0.0, 0.02, 0.15)
    put('LAnkle', 0.8 + lx, 0.08, 0.0); put('LBigToe', 0.8 + lx, 0.02, 0.15)
    put('RKnee', 0.1, 0.5, 0.0); put('LKnee', 0.7 + lx, 0.5, 0.0)
    put('RHip', 0.3 + sx, 0.95, -0.1); put('LHip', 0.5 + sx, 0.95, 0.1)
    put('midHip', 0.4 + sx, 0.95, 0.0)
    up_x = 0.4 + sx + hx
    put('Neck', up_x, 1.45 - dy, 0.0); put('C7_study', up_x - 0.03, 1.47 - dy, 0.0)
    put('RShoulder', up_x, 1.42 - dy, -0.18); put('LShoulder', up_x, 1.42 - dy, 0.18)
    put('RElbow', up_x + 0.1, 1.2 - dy, -0.2); put('LElbow', up_x + 0.1, 1.2 - dy, 0.2)
    put('RWrist', up_x + 0.3, 1.1 - dy, -0.1); put('LWrist', up_x + 0.3, 1.1 - dy, 0.1)
    return pd.DataFrame(cols)


PLANT = 80         # foot plant at 0.8 s, 0.2 s before contact


def run(trc, plant=PLANT):
    opt = RefinedHittingOptimizer(80, H, 'college', handedness='right')
    kin = pd.DataFrame({'time': np.arange(N) / FS})
    stride = {} if plant is None else {'plant_frame': plant}
    return opt, opt.calculate_balance(trc, kin, {'contact_frame': CONTACT}, stride)


def test_a_still_centred_hitter_scores_well():
    opt, b = run(body())
    assert b['head_movement_cm'] < 1 and abs(b['com_position_pct'] - 50) < 15
    assert b['finish_drift_cm'] < 1 and b['finish_stepped'] is False
    dims = {}
    opt._balance_dims(dims, b)
    assert all(dims[k]['stars'] == 5 for k in ('head_movement_load', 'head_movement_swing',
                                                'com_balance', 'finish_balance')), dims


def test_head_drifting_forward_is_measured_along_the_stride():
    _, b = run(body(shift=lambda t: 0.15 * min(t, 1.0)))    # whole upper body 15 cm forward, upright
    assert 13 < b['head_forward_cm'] < 17 and b['head_movement_cm'] > 13


def test_a_forward_lean_moves_the_head_more_than_the_neck():
    _, b = run(body(head=lambda t: 0.15 * min(t, 1.0)))     # neck 15 cm forward, hips still
    assert b['head_forward_cm'] > 17


def test_head_drop_is_positive_down():
    _, b = run(body(drop=lambda t: 0.08 * min(t, 1.0)))
    assert 6 < b['head_drop_cm'] < 10


def test_a_lunge_puts_the_centre_of_mass_over_the_front_foot():
    opt, b = run(body(shift=lambda t: 0.45 * min(t, 1.0)))
    assert b['com_position_pct'] > 75
    dims = {}
    opt._balance_dims(dims, b)
    assert dims['com_balance']['stars'] <= 3 and 'toward the front foot' in dims['com_balance']['description']


def test_a_catch_step_caps_the_finish():
    opt, b = run(body(lead_step=lambda t: 0.0 if t < 1.2 else 0.3))
    assert b['finish_stepped'] is True
    dims = {}
    opt._balance_dims(dims, b)
    assert dims['finish_balance']['stars'] <= 2


def test_drifting_after_contact_lowers_finish_balance():
    _, still = run(body())
    _, drift = run(body(shift=lambda t: 0.0 if t < 1.0 else 0.4 * (t - 1.0)))
    assert drift['finish_drift_cm'] > still['finish_drift_cm'] + 15


def test_short_capture_reports_finish_as_not_measured():
    trc = body().iloc[:CONTACT + 20].reset_index(drop=True)
    opt = RefinedHittingOptimizer(80, H, 'college', handedness='right')
    kin = pd.DataFrame({'time': np.arange(CONTACT + 20) / FS})
    b = opt.calculate_balance(trc, kin, {'contact_frame': CONTACT}, {})
    assert 'finish_drift_cm' not in b and b['finish_reason']
    dims = {}
    opt._balance_dims(dims, b)
    assert dims['finish_balance']['available'] is False
    assert dims['head_movement_swing'].get('available') is False   # no foot plant passed


def test_without_markers_the_phase_is_not_measured_and_not_failed():
    opt = RefinedHittingOptimizer(80, H, 'college', handedness='right')
    assert opt.calculate_balance(None, pd.DataFrame({'time': [0.0]}), {'contact_frame': 0}, {}) == {}
    dims = {}
    opt._balance_dims(dims, {})
    assert all(dims[k]['available'] is False and dims[k]['stars'] is None
               for k in ('head_movement_load', 'head_movement_swing', 'com_balance', 'finish_balance'))


def test_phase_layout_and_score_weight():
    assert SWING_PHASES['balance_load']['label'] == 'Balance & Head Movement'
    assert SWING_PHASES['balance_load']['dimensions'] == ['head_movement_load', 'head_movement_swing',
                                                          'com_balance', 'finish_balance']
    assert 'torso_coil_at_plant' in SWING_PHASES['stride']['dimensions']
    assert {'head_movement_load', 'head_movement_swing', 'com_balance', 'finish_balance'} <= set(INFORMATIONAL)


def _ramp(t0, t1, d):
    """0 before t0, d after t1, smooth (cosine) in between."""
    return lambda t: d * 0.5 * (1 - np.cos(np.clip((t - t0) / (t1 - t0), 0, 1) * np.pi))


def test_head_travel_during_the_stride_lands_in_the_load_part():
    # Upper body (and head) drifts 12 cm forward, upright, between 0.2 and 0.7 s (plant at 0.8 s).
    opt, b = run(body(shift=_ramp(0.2, 0.7, 0.12)))
    assert 11 < b['head_load_movement_cm'] < 15, b['head_load_movement_cm']
    assert b['head_swing_movement_cm'] < 1.0, b['head_swing_movement_cm']
    dims = {}
    opt._balance_dims(dims, b)
    assert dims['head_movement_swing']['stars'] == 5 and dims['head_movement_load']['stars'] < 5


def test_head_travel_after_plant_lands_in_the_swing_part():
    # Upper body (and head) lunges 10 cm forward between plant (0.8 s) and contact (1.0 s).
    opt, b = run(body(shift=_ramp(0.82, 0.98, 0.10)))
    assert b['head_load_movement_cm'] < 1.0, b['head_load_movement_cm']
    assert 9 < b['head_swing_movement_cm'] < 13, b['head_swing_movement_cm']
    dims = {}
    opt._balance_dims(dims, b)
    assert dims['head_movement_load']['stars'] == 5 and dims['head_movement_swing']['stars'] <= 3


def test_without_foot_plant_both_head_parts_are_not_measured():
    opt, b = run(body(head=_ramp(0.2, 0.7, 0.12)), plant=None)
    assert 'head_movement_cm' in b and b['head_split_reason']
    dims = {}
    opt._balance_dims(dims, b)
    for k in ('head_movement_load', 'head_movement_swing'):
        assert dims[k]['available'] is False and 'Foot plant' in dims[k]['unavailable_reason']


def test_balance_measures_carry_no_percentile():
    opt, b = run(body())
    rep = opt.build_phase_report(None, None, {}, {}, balance=b)
    for k in ('head_movement_load', 'head_movement_swing', 'com_balance', 'finish_balance'):
        assert 'percentile' not in rep['dimensions'][k], k


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('✓', name)
