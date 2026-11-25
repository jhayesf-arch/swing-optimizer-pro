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


# ── Reliability gate: parts the score withholds stay out of advice and rankings ──

def test_sequencing_is_skipped_when_the_mechanics_score_skipped_it():
    mech = {'components': [{'key': 'sequencing', 'status': 'skipped',
                            'reason': '400 ms lag is beyond any published value'}]}
    r = score({**GOOD, 'sequence_quality': {'value': 400, 'stars': 3}}, mechanics=mech)
    seq = next(p for p in r['components'] if p['key'] == 'sequence_quality')
    assert seq['status'] == 'skipped' and '400 ms' in seq['reason'], seq


def test_implausible_lead_leg_block_and_stride_are_not_scored():
    r = score({**GOOD, 'lead_leg_block': {'value': 150, 'stars': 5},
               'stride_length': {'value': 2.0, 'stars': 5}})
    parts = {p['key']: p for p in r['components']}
    assert parts['lead_leg_block']['status'] == 'skipped', parts['lead_leg_block']
    assert parts['stride_length']['status'] == 'skipped', parts['stride_length']


def test_a_nan_value_is_not_measured_rather_than_scored_100():
    r = score({**GOOD, 'pelvis_speed': {'value': float('nan'), 'stars': 1}})
    pelvis = next(p for p in r['components'] if p['key'] == 'pelvis_speed')
    assert pelvis['status'] == 'skipped' and pelvis['score'] is None, pelvis


def test_exactly_half_the_weight_still_gets_a_score():
    # hand speed + sequencing + torso speed + pelvis speed = 0.15 + 0.15 + 0.12 + 0.08 = 0.50
    keep = {'hand_speed', 'sequence_quality', 'torso_speed', 'pelvis_speed'}
    dims = {k: (v if k in keep else {**v, 'available': False}) for k, v in GOOD.items()}
    r = score(dims)
    assert r['score'] is not None and abs(r['coverage'] - 0.5) < 1e-9, r


def test_gate_removes_withheld_parts_from_prescriptions_and_percentile():
    from swing_score import gate_phase_report
    dims = {'sequence_quality': {'value': -77, 'stars': 1, 'percentile': 8},
            'hand_speed': {'value': 17.7, 'stars': 3, 'percentile': 70},
            'torso_speed': {'value': 1151, 'stars': 5, 'percentile': 99}}
    report = {
        'dimensions': dims,
        'phases': {'p': {'dimensions': [{**dims['sequence_quality'], 'key': 'sequence_quality'},
                                        {**dims['hand_speed'], 'key': 'hand_speed'}]}},
        'prescriptions': [{'key': 'sequence_quality', 'priority': 1}, {'key': 'hand_speed', 'priority': 2}],
        'overall_percentile': 60,
    }
    detail = {'components': [
        {'key': 'sequence_quality', 'status': 'skipped', 'reason': 'torso signal unstable'},
        {'key': 'hand_speed', 'status': 'scored', 'reason': ''},
        {'key': 'torso_speed', 'status': 'skipped', 'reason': 'far above the youth range'}]}
    gate_phase_report(report, detail, {'sequence_quality': 0.15, 'hand_speed': 0.15, 'torso_speed': 0.12})
    seq = report['dimensions']['sequence_quality']
    assert seq['available'] is False and seq['unavailable_reason'] == 'torso signal unstable'
    assert 'percentile' not in seq and seq['value'] == -77           # the value stays visible
    assert report['phases']['p']['dimensions'][0]['available'] is False
    assert [p['key'] for p in report['prescriptions']] == ['hand_speed']
    assert report['prescriptions'][0]['priority'] == 1
    assert report['overall_percentile'] == 70                          # only the scored part ranks


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('✓', name)


def test_youth_rotation_speed_far_above_published_range_is_not_scored():
    # 1160 deg/s pelvis for a youth hitter is ~4.5 SD above the published youth mean
    # (632 +/- 117): flagged as a capture artifact, not rewarded.
    fast = {**GOOD, 'pelvis_speed': {'value': 1160, 'stars': 5}}
    r = score(fast, level='youth')
    p = next(c for c in r['components'] if c['key'] == 'pelvis_speed')
    assert p['status'] == 'skipped' and 'youth' in p['reason'], p


def test_same_speed_within_range_for_a_faster_level_is_still_capped_absolutely():
    # The level cap never loosens the absolute cap: 1250 deg/s is out for everyone.
    fast = {**GOOD, 'pelvis_speed': {'value': 1250, 'stars': 5}}
    for level in ('youth', 'college', 'professional'):
        p = next(c for c in score(fast, level=level)['components'] if c['key'] == 'pelvis_speed')
        assert p['status'] == 'skipped', (level, p)


def test_normal_youth_speeds_are_scored():
    ok = {**GOOD, 'pelvis_speed': {'value': 700, 'stars': 5}, 'torso_speed': {'value': 800, 'stars': 5}}
    r = score(ok, level='youth')
    assert all(c['status'] == 'scored' for c in r['components']
               if c['key'] in ('pelvis_speed', 'torso_speed')), r
