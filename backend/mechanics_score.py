"""Mechanics score v2: how well the swing's mechanics match what research says
drives bat speed, on a 0-100 scale.

It replaces the v1 "physics efficiency" checklist (start at 100, subtract fixed
penalties) with these rules:

1. Only mechanisms with published evidence drive the score, one vote each:
   sequencing, hip-shoulder separation, pelvis drive, lead-leg block and
   stride. Our own composite measures (energy-transfer proxy, stride
   efficiency, pelvis force proxy) do not count, and neither does lateral
   sway: no published batting study measures it.
2. Every part is scored on a smooth ramp between research reference values for
   the athlete's level; there are no cliff edges where one degree costs 20 points.
3. Weights follow the strength of the evidence and of the published link to
   bat speed.
4. A part the capture cannot support (missing, or flagged unreliable in the
   report's evidence) is skipped and the weights renormalise; the score says
   how much of it is based on measured data. Below half, it is not given.

Each part carries its reference so a coach can see where the numbers come from.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

VERSION = 'mechanics_v2'
MIN_COVERAGE = 0.5

LEVELS = ('youth', 'high_school', 'college', 'professional')

# Peak pelvis angular velocity (deg/s): Escamilla et al. 2009 report 632 +/- 117
# for skilled youth (12-17 y) and 678 +/- 87 for college/pro adults; Welch et
# al. 1995 report 714. Perrett 2024 finds it associated with bat speed. The
# professional ramp above ~720 is ours.
PELVIS_SPEED = {
    'youth':        [(250, 0), (400, 60), (550, 100)],
    'high_school':  [(350, 0), (500, 60), (650, 100)],
    'college':      [(400, 0), (550, 60), (700, 100)],
    'professional': [(500, 0), (700, 70), (850, 100)],
}
# Pelvis peak after front-foot plant (ms): Welch et al. 1995, elite ~50-100 ms.
# Negative means the pelvis peaked before the foot landed (spinning open early).
PELVIS_TIMING = [(-60, 0), (0, 50), (50, 100), (100, 100), (180, 55), (260, 20)]

# Values beyond these are not one swing measured well: a whole clip's movement,
# a tracking jump, or an event detected in the wrong place. Such a part is
# skipped with the reason shown rather than scored at an extreme.
PLAUSIBLE = {
    'sequence_timing_ms': 250,          # |pelvis-to-torso gap|
    'peak_pelvis_omega_3d_deg_s': 1500,
    'lead_leg_block': 100,              # deg of knee extension into contact (skilled adults ~59)
    'stride_ratio': 1.3,
}


def _implausible(key: str, value: Optional[float]) -> str:
    lim = PLAUSIBLE.get(key)
    if value is None or lim is None or abs(value) <= lim:
        return ''
    return f'{value:g} is outside the plausible range for one swing (limit {lim:g}); likely a capture artifact'



def _ramp(x: float, pts: Iterable[Tuple[float, float]]) -> float:
    """Piecewise-linear through (value, score) points, flat beyond the ends."""
    pts = sorted(pts)
    if x <= pts[0][0]:
        return float(pts[0][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    return float(pts[-1][1])


def _stars_to_points(thresholds: List[Tuple[float, int]]) -> List[Tuple[float, float]]:
    """A star table [(value, stars 1-5)] as a 0-100 ramp: 1 star = 0, 5 = 100."""
    return [(v, (s - 1) * 25.0) for v, s in thresholds]


def _reliable(evidence: Dict, *keys: str) -> Tuple[bool, str]:
    for k in keys:
        e = (evidence or {}).get(k)
        if e and e.get('reliable') is False:
            return False, e.get('caveat') or 'flagged unreliable for this capture'
    return True, ''


def _num(d: Dict, key: str) -> Optional[float]:
    v = (d or {}).get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if v == v else None          # NaN guard


def compute_mechanics_score(metrics: Dict, dims: Dict, evidence: Dict, skill_level: str,
                            benchmarks: Dict, thresholds: Dict) -> Dict:
    level = skill_level if skill_level in LEVELS else 'high_school'
    bench = benchmarks.get(level, {})
    parts = []

    def part(key, label, weight, tier, source, value, unit, score, reference, skip=''):
        parts.append({'key': key, 'label': label, 'weight': weight, 'tier': tier, 'source': source,
                      'value': None if value is None else round(value, 2), 'unit': unit,
                      'reference': reference,
                      'score': None if skip else round(max(0.0, min(100.0, score)), 1),
                      'status': 'skipped' if skip else 'scored', 'reason': skip})

    # 1. Sequencing: pelvis peaks before torso, with a real gap.
    lag = _num(metrics, 'sequence_timing_ms')
    lo, hi = bench.get('sequence_timing_ms', (20, 55))
    ok, why = _reliable(evidence, 'sequence_timing_ms', 'proper_sequence')
    if lag is None or (lag == 0 and not metrics.get('proper_sequence')):
        part('sequencing', 'Sequencing', 0.25, 'A', 'Welch et al. 1995; Escamilla et al. 2009; Taguchi et al. 2023',
             lag, 'ms', 0, f'pelvis peaks {lo}-{hi} ms before torso', 'sequence timing not measured')
    elif not ok or _implausible('sequence_timing_ms', lag):
        why = why or _implausible('sequence_timing_ms', lag)
        part('sequencing', 'Sequencing', 0.25, 'A', 'Welch et al. 1995; Escamilla et al. 2009; Taguchi et al. 2023',
             lag, 'ms', 0, f'pelvis peaks {lo}-{hi} ms before torso', why)
    else:
        s = _ramp(lag, [(-40, 0), (0, 30), (lo, 100), (hi, 100), (hi + 60, 55), (hi + 140, 25)])
        if metrics.get('proper_sequence') is False:
            s = min(s, 25.0)          # out of order: the chain is not building from the ground
        part('sequencing', 'Sequencing', 0.25, 'A', 'Welch et al. 1995; Escamilla et al. 2009; Taguchi et al. 2023',
             lag, 'ms', s, f'pelvis peaks {lo}-{hi} ms before torso')

    # 2. Hip-shoulder separation (X-factor).
    sep = _num(metrics, 'max_separation_deg')
    xlo, xhi = bench.get('x_factor_optimal', (20, 42))
    ok, why = _reliable(evidence, 'max_separation_deg')
    ref = f'{xlo}-{xhi}° at this level'
    if not sep:
        part('separation', 'Hip-shoulder separation', 0.20, 'A', 'Fleisig et al. 2013; Hoechstetter et al. 2026',
             sep, '°', 0, ref, 'separation not measured')
    elif not ok:
        part('separation', 'Hip-shoulder separation', 0.20, 'A', 'Fleisig et al. 2013; Hoechstetter et al. 2026',
             sep, '°', 0, ref, why)
    else:
        s = _ramp(sep, [(xlo * 0.4, 0), (xlo, 85), ((xlo + xhi) / 2, 100), (xhi, 95), (xhi + 20, 50)])
        part('separation', 'Hip-shoulder separation', 0.20, 'A', 'Fleisig et al. 2013; Hoechstetter et al. 2026',
             sep, '°', s, ref)

    # 3. Pelvis drive: how fast, and when relative to foot plant.
    w = _num(metrics, 'peak_pelvis_omega_3d_deg_s')
    t = _num(metrics, 'time_to_peak_pelvis_from_plant_ms')
    ok_w, why_w = _reliable(evidence, 'peak_pelvis_omega_3d_deg_s')
    pts = PELVIS_SPEED[level]
    ref = f'{pts[1][0]}-{pts[-1][0]}°/s peak; peak 50-100 ms after plant'
    sub = []
    if w and ok_w and _implausible('peak_pelvis_omega_3d_deg_s', w):
        ok_w, why_w = False, _implausible('peak_pelvis_omega_3d_deg_s', w)
    if w and ok_w:
        sub.append((0.7, _ramp(w, pts)))
    if t not in (None, 0.0):
        ok_t, _ = _reliable(evidence, 'time_to_peak_pelvis_from_plant_ms')
        if ok_t:
            sub.append((0.3, _ramp(t, PELVIS_TIMING)))
    src = 'Welch et al. 1995; Escamilla et al. 2009; Perrett 2024'
    if sub:
        s = sum(a * b for a, b in sub) / sum(a for a, _ in sub)
        part('pelvis_drive', 'Pelvis drive', 0.25, 'A', src, w, '°/s', s, ref)
    else:
        part('pelvis_drive', 'Pelvis drive', 0.25, 'A', src, w, '°/s', 0, ref,
             why_w or 'pelvis speed not measured')

    # 4. Lead-leg block: lead-knee extension from plant to contact.
    d = (dims or {}).get('lead_leg_block') or {}
    llb = _num(d, 'value')
    table = (thresholds.get('lead_leg_block') or {}).get(level)
    src = 'Escamilla et al. 2009; Perrett 2024'
    if llb is None or d.get('available') is False or not table or _implausible('lead_leg_block', llb):
        part('lead_leg_block', 'Lead-leg block', 0.15, 'B', src, llb, '°', 0,
             'knee extension into contact', _implausible('lead_leg_block', llb) or 'lead-leg block not measured')
    else:
        part('lead_leg_block', 'Lead-leg block', 0.15, 'B', src, llb, '°',
             _ramp(llb, _stars_to_points(table)), f'{table[2][0]}-{table[4][0]}° of extension at this level')

    # 5. Stride, height-normalised.
    sr = _num(metrics, 'stride_ratio')
    table = (thresholds.get('stride_length') or {}).get(level)
    src = 'Escamilla et al. 2009'
    if not sr or not table or _implausible('stride_ratio', sr):
        part('stride', 'Stride', 0.10, 'A', src, sr, '× height', 0, 'stride length / height',
             _implausible('stride_ratio', sr) or 'stride not measured')
    else:
        part('stride', 'Stride', 0.10, 'A', src, sr, '× height',
             _ramp(sr, _stars_to_points(table)), f'{table[2][0]}-{table[4][0]} × height at this level')

    scored = [p for p in parts if p['status'] == 'scored']
    total_w = sum(p['weight'] for p in parts)
    used_w = sum(p['weight'] for p in scored)
    coverage = used_w / total_w if total_w else 0.0
    score = (round(sum(p['score'] * p['weight'] for p in scored) / used_w)
             if scored and coverage >= MIN_COVERAGE else None)
    return {
        'version': VERSION,
        'score': score,
        'coverage': round(coverage, 2),
        'parts_scored': len(scored),
        'parts_total': len(parts),
        'skill_level': level,
        'components': parts,
        'note': (None if score is not None else
                 'Not enough reliable measurements on this capture to give a mechanics score.'),
    }
