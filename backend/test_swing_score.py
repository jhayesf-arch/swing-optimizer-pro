"""Swing Score v2. Run: python test_swing_score.py"""
from analyzer import DIMENSION_THRESHOLDS
from swing_score import INFORMATIONAL, WEIGHTS, compute_swing_score

GOOD = {
    'hand_speed': {'value': 25.0, 'stars': 5},
    'sequence_quality': {'value': 40, 'stars': 5},
    'torso_speed': {'value': 857, 'stars': 5},
    'max_hip_shoulder_separation': {'value': 45, 'stars': 4},
    'lead_leg_block': {'value': 59, 'stars': 5},
    'pelvis_speed': {'value': 678, 'stars': 5},
    'torso_coil_at_plant': {'value': 14, 'stars': 5},
    'stride_length': {'value': 0.48, 'stars': 5},
    'pelvis_direction_at_contact': {'value': 12, 'stars': 5},
    'upper_torso_direction_at_contact': {'value': 16, 'stars': 5},
    'pelvis_rotation_range': {'value': 72, 'stars': 5},
    'upper_torso_rotation_range': {'value': 98, 'stars': 5},
    'follow_through_quality': {'value': 0, 'stars': 1},
    'kinetic_chain_efficiency': {'value': 0, 'stars': 1},
}


def score(dims=GOOD, evidence=None, mechanics=None, level='college'):
    return compute_swing_score(dims, evidence or {}, mechanics, level, DIMENSION_THRESHOLDS)


def test_a_sound_college_swing_scores_high_and_uses_every_part():
    r = score()
    assert r['score'] >= 90 and r['coverage'] == 1.0 and r['parts_scored'] == len(WEIGHTS), r


def test_informational_dimensions_do_not_move_the_score():
    worse = {**GOOD, 'follow_through_quality': {'value': 90, 'stars': 5},
             'kinetic_chain_efficiency': {'value': 90, 'stars': 5}}
    assert score(worse)['score'] == score()['score']
    assert set(score()['informational']) <= set(INFORMATIONAL)


def test_no_star_cliffs():
    a = score({**GOOD, 'max_hip_shoulder_separation': {'value': 37.9, 'stars': 3}})['score']
    b = score({**GOOD, 'max_hip_shoulder_separation': {'value': 38.1, 'stars': 4}})['score']
    assert abs(a - b) < 0.5, (a, b)


def test_trunk_flip_skips_the_parts_it_corrupts():
    flip = {'reliable': False, 'caveat': 'The trunk angle flips faster than a spine can turn'}
    ev = {k: flip for k in ('peak_pelvis_omega_3d_deg_s', 'max_separation_deg', 'sequence_timing_ms',
                            'torso_to_pelvis_rot_ratio')}
    r = score(evidence=ev)
    skipped = {p['key'] for p in r['components'] if p['status'] == 'skipped'}
    assert {'pelvis_speed', 'max_hip_shoulder_separation', 'sequence_quality', 'torso_speed',
            'torso_coil_at_plant'} <= skipped
    assert 'lead_leg_block' not in skipped


def test_torso_far_faster_than_pelvis_is_an_artifact():
    r = score({**GOOD, 'torso_speed': {'value': 1147, 'stars': 5}, 'pelvis_speed': {'value': 448, 'stars': 1}})
    torso = next(p for p in r['components'] if p['key'] == 'torso_speed')
    assert torso['status'] == 'skipped' and '2x' in torso['reason']


def test_too_little_data_gives_no_score():
    r = score({'stride_length': {'value': 0.48, 'stars': 5}})
    assert r['score'] is None and r['coverage'] < 0.5 and r['note']


def test_sequence_uses_the_mechanics_ramp_when_available():
    mech = {'components': [{'key': 'sequencing', 'status': 'scored', 'score': 62.5}]}
    seq = next(p for p in score(mechanics=mech)['components'] if p['key'] == 'sequence_quality')
    assert seq['score'] == 62.5


def test_lower_is_better_dimensions_rate_by_value():
    from analyzer import RefinedHittingOptimizer
    o = RefinedHittingOptimizer(84, 1.8, 'college')
    stars = [o._rate_dimension('pelvis_direction_at_contact', v, invert=True) for v in (5, 18, 30, 60, 95)]
    assert stars == [5, 4, 3, 1, 1], stars


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('✓', name)
