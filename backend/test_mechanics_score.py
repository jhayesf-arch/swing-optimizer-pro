"""Mechanics score v2. Run: python test_mechanics_score.py"""
from analyzer import DIMENSION_THRESHOLDS, SKILL_LEVEL_BENCHMARKS
from mechanics_score import compute_mechanics_score, _ramp

GOOD = {'sequence_timing_ms': 45, 'proper_sequence': True, 'max_separation_deg': 40,
        'peak_pelvis_omega_3d_deg_s': 640, 'time_to_peak_pelvis_from_plant_ms': 70,
        'stride_ratio': 0.7, 'lateral_sway_range_m': 0.05}
DIMS = {'lead_leg_block': {'value': 26, 'available': True}}


def score(metrics, dims=DIMS, evidence=None, level='college'):
    return compute_mechanics_score(metrics, dims, evidence or {}, level,
                                   SKILL_LEVEL_BENCHMARKS, DIMENSION_THRESHOLDS)


def test_a_sound_college_swing_scores_high_and_uses_every_part():
    r = score(GOOD)
    assert r['score'] >= 85 and r['coverage'] == 1.0 and r['parts_scored'] == 5, r


def test_no_cliff_edges():
    # X-factor 29.9 vs 30.1 deg: v1 jumped 20 points here; v2 moves a fraction.
    a = score({**GOOD, 'max_separation_deg': 29.9})['score']
    b = score({**GOOD, 'max_separation_deg': 30.1})['score']
    assert abs(a - b) <= 1, (a, b)


def test_out_of_order_sequence_caps_sequencing():
    r = score({**GOOD, 'proper_sequence': False})
    seq = next(p for p in r['components'] if p['key'] == 'sequencing')
    assert seq['score'] <= 25


def test_unreliable_and_implausible_parts_are_skipped_not_scored():
    ev = {'max_separation_deg': {'reliable': False, 'caveat': 'clipped at the joint limit'}}
    r = score({**GOOD, 'stride_ratio': 2.1}, evidence=ev)
    by = {p['key']: p for p in r['components']}
    assert by['separation']['status'] == 'skipped' and 'clipped' in by['separation']['reason']
    assert by['stride']['status'] == 'skipped' and 'plausible' in by['stride']['reason']
    assert r['parts_scored'] == 3 and r['score'] is not None


def test_too_little_data_gives_no_score():
    r = score({'stride_ratio': 0.7}, dims={})
    assert r['score'] is None and r['coverage'] < 0.5 and r['note']


def test_levels_raise_the_bar():
    assert score(GOOD, level='professional')['score'] < score(GOOD, level='high_school')['score']


def test_every_part_cites_its_source():
    assert all(p['source'] and p['tier'] in ('A', 'B') for p in score(GOOD)['components'])


def test_ramp_is_flat_beyond_the_ends():
    pts = [(0, 0), (10, 100)]
    assert _ramp(-5, pts) == 0 and _ramp(5, pts) == 50 and _ramp(50, pts) == 100

def test_stride_the_report_could_not_measure_is_skipped():
    dims = {**DIMS, 'stride_length': {'available': False, 'unavailable_reason': 'no foot markers'}}
    r = score({**GOOD, 'stride_ratio': 0.01}, dims=dims)
    stride = next(p for p in r['components'] if p['key'] == 'stride')
    assert stride['status'] == 'skipped' and 'foot markers' in stride['reason']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('✓', name)
