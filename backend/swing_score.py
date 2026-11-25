"""Swing Score v2: the whole swing, mechanics plus result, on a 0-100 scale.

It follows the same rules as the mechanics score (mechanics_score.py):

1. Only dimensions with published batting evidence carry weight. The rest
   (negative move, forward move, the kinetic-chain-efficiency proxy and
   follow-through) stay in the report as information, at weight 0.
2. Each dimension is scored on a smooth ramp through its level's threshold
   table instead of being rounded to stars first, so one degree never moves
   the score by a whole star.
3. Weight follows the strength of the evidence and of the published link to
   bat speed or to playing level.
4. A dimension the capture cannot support (unavailable, or flagged unreliable
   in the report's evidence, e.g. by a trunk IK flip) is skipped and the weights
   renormalise. Below half the weight measured, no score is given.

Where the mechanics score asks "how sound is the movement?", this adds the
result (hand speed) to ask "how good is the swing?".
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from mechanics_score import PLAUSIBLE, PLAUSIBLE_SD, _ramp, _stars_to_points, level_plausible_max
from mechanics_score import _implausible as _mech_implausible

VERSION = 'swing_v2'
MIN_COVERAGE = 0.5

# Weight, and the metric_evidence keys whose reliability the dimension inherits.
WEIGHTS = {
    'hand_speed':                       (0.15, ('max_hand_speed_mph',)),
    'sequence_quality':                 (0.15, ('sequence_timing_ms', 'proper_sequence')),
    'torso_speed':                      (0.12, ('torso_to_pelvis_rot_ratio',)),
    'max_hip_shoulder_separation':      (0.12, ('max_separation_deg',)),
    'lead_leg_block':                   (0.10, ()),
    'pelvis_speed':                     (0.08, ('peak_pelvis_omega_3d_deg_s',)),
    'torso_coil_at_plant':              (0.06, ('max_separation_deg',)),
    'stride_length':                    (0.06, ()),
    'pelvis_direction_at_contact':      (0.06, ('pelvis_rotation_at_contact_deg',)),
    'upper_torso_direction_at_contact': (0.05, ('torso_to_pelvis_rot_ratio',)),
    'pelvis_rotation_range':            (0.025, ()),
    'upper_torso_rotation_range':       (0.025, ('torso_to_pelvis_rot_ratio',)),
}

# Beyond these a value is a capture artifact, not a swing. Published peaks:
# torso <= ~1,000 deg/s (Nakata et al. 2014: 984 +/- 57), pelvis <= ~900
# (Inkster 2010: 867 +/- 65); torso/pelvis peak ratio 1.1-1.6 in batters, and
# the analyzer already treats > 2x as an IK artifact.
PLAUSIBLE_MAX = {'torso_speed': 1300.0, 'pelvis_speed': 1200.0}
MAX_TORSO_PELVIS_RATIO = 2.0
# Dimensions whose believable range the mechanics score already defines (same limits, so
# the two scores never disagree about what is a capture artifact).
MECHANICS_LIMIT_KEY = {'lead_leg_block': 'lead_leg_block', 'stride_length': 'stride_ratio'}


def plausible_max(key: str, skill_level: Optional[str] = None) -> Optional[float]:
    """Largest believable value of key: the absolute cap, tightened per level."""
    cap = PLAUSIBLE_MAX.get(key)
    level_cap = level_plausible_max(key, skill_level)
    if level_cap is not None:
        cap = level_cap if cap is None else min(cap, level_cap)
    return cap


# In the report, not in the score: no published batting values to score against.
INFORMATIONAL = ('negative_move', 'forward_move', 'kinetic_chain_efficiency',
                 'follow_through_quality',
                 # balance as coaches use the word: coaching consensus, needs the .trc
                 'head_movement_load', 'head_movement_swing', 'com_balance',
                 'finish_balance')


def _unreliable(evidence: Dict, keys) -> str:
    for k in keys:
        e = (evidence or {}).get(k)
        if e and e.get('reliable') is False:
            return e.get('caveat') or 'flagged unreliable for this capture'
    return ''


def _implausible(key: str, dims: Dict, skill_level: Optional[str] = None) -> str:
    def val(k):
        try:
            return float(((dims or {}).get(k) or {}).get('value'))
        except (TypeError, ValueError):
            return None
    v = val(key)
    if v is not None and not math.isfinite(v):
        return 'not measured (no finite value on this capture)'
    if key in MECHANICS_LIMIT_KEY and v is not None:
        why = _mech_implausible(MECHANICS_LIMIT_KEY[key], v)
        if why:
            return why
    # The torso/pelvis ratio names the cause, so it is reported ahead of the caps.
    if key == 'torso_speed':
        p = val('pelvis_speed')
        if v and p and v > MAX_TORSO_PELVIS_RATIO * p:
            return (f'torso {v:.0f} vs pelvis {p:.0f} deg/s is over {MAX_TORSO_PELVIS_RATIO:g}x; '
                    'batters show 1.1-1.6x, so the trunk signal is likely corrupted')
    lim = plausible_max(key, skill_level)
    if v is not None and lim is not None and v > lim:
        if lim < PLAUSIBLE_MAX.get(key, float('inf')):
            level = (skill_level or '').replace('_', ' ')
            return (f'{v:.0f} is far above the published range for {level} hitters '
                    f'(over {PLAUSIBLE_SD:g} SD above the mean, cap {lim:.0f}); likely a '
                    'capture artifact, so it is not scored. Check this swing\'s video.')
        return f'{v:.0f} is beyond any published value ({lim:.0f}); likely a capture artifact'
    return ''


def _mechanics_skipped(key: str, mechanics: Optional[Dict]) -> str:
    """Sequencing is graded by the mechanics score's timing ramp; when that part was
    skipped (implausible lag, unreliable trunk), it is skipped here too, for the same reason."""
    if key != 'sequence_quality' or not mechanics:
        return ''
    for c in mechanics.get('components', []):
        if c.get('key') == 'sequencing' and c.get('status') == 'skipped':
            return c.get('reason') or 'sequencing not measured reliably on this capture'
    return ''


def _dimension_score(key: str, dim: Dict, table: Optional[List], mechanics: Optional[Dict]) -> Optional[float]:
    if key == 'sequence_quality':
        # Scored like the mechanics score's sequencing part, which ramps through
        # the published timing window instead of bucketing it into stars.
        for c in (mechanics or {}).get('components', []):
            if c.get('key') == 'sequencing' and c.get('status') == 'scored':
                return float(c['score'])
        stars = dim.get('stars')
        if mechanics is not None:
            return None   # the mechanics score judged the timing and did not score it
        return None if stars is None else (stars - 1) * 25.0
    value = dim.get('value')
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    if not table:
        return (dim['stars'] - 1) * 25.0 if dim.get('stars') else None
    return _ramp(value, _stars_to_points(table))


def compute_swing_score(dims: Dict, evidence: Dict, mechanics: Optional[Dict],
                        skill_level: str, thresholds: Dict) -> Dict:
    parts = []
    for key, (weight, ev_keys) in WEIGHTS.items():
        dim = (dims or {}).get(key)
        label = (dim or {}).get('label', key)
        skip = ''
        if not dim:
            skip = 'not measured'
        elif dim.get('available') is False:
            skip = dim.get('unavailable_reason') or 'not measured on this capture'
        else:
            skip = (_unreliable(evidence, ev_keys) or _implausible(key, dims, skill_level)
                    or _mechanics_skipped(key, mechanics))
        score = None
        if not skip:
            table = (thresholds.get(key) or {}).get(skill_level)
            score = _dimension_score(key, dim, table, mechanics)
            if score is None:
                skip = 'not measured'
        parts.append({
            'key': key, 'label': label, 'weight': weight,
            'value': (dim or {}).get('value'), 'unit': (dim or {}).get('unit'),
            'score': None if skip else round(max(0.0, min(100.0, score)), 1),
            'status': 'skipped' if skip else 'scored', 'reason': skip,
        })

    scored = [p for p in parts if p['status'] == 'scored']
    total_w = sum(w for w, _ in WEIGHTS.values())
    used_w = sum(p['weight'] for p in scored)
    # Rounded so floating-point sums (weights add to 1.0000000000000002) can't fail
    # a swing measured on exactly half the weight.
    coverage = round(used_w / total_w, 9) if total_w else 0.0
    score = (round(sum(p['score'] * p['weight'] for p in scored) / used_w, 1)
             if scored and coverage >= MIN_COVERAGE else None)
    return {
        'version': VERSION,
        'score': score,
        'coverage': round(coverage, 2),
        'parts_scored': len(scored),
        'parts_total': len(parts),
        'skill_level': skill_level,
        'components': parts,
        'informational': [k for k in INFORMATIONAL if k in (dims or {})],
        'note': (None if score is not None else
                 'Not enough reliable measurements on this capture to give a swing score.'),
    }


def gate_phase_report(report: Dict, detail: Dict, weights: Dict[str, float]) -> None:
    """Keep what the Swing Score withheld out of everything else the report says.

    A part skipped as unreliable or implausible (a trunk IK flip, a torso far above its
    level's range) must not still earn stars, a percentile, a drill or a share of the
    overall percentile. Each such dimension is marked unavailable with the score's reason
    (its value stays, for the record), the prescriptions are rebuilt without it, and the
    overall percentile is re-weighted over the parts that remain. Mutates report in place.
    The multi-swing profile and the cohort builder already ignore unavailable dimensions,
    so they inherit the gate.
    """
    dims = report.get('dimensions') or {}
    withheld = {}
    for c in (detail or {}).get('components', []):
        d = dims.get(c.get('key'))
        if c.get('status') == 'skipped' and d is not None and d.get('available') is not False:
            withheld[c['key']] = c.get('reason') or 'not measured reliably on this capture'
    for key, why in withheld.items():
        d = dims[key]
        d['available'] = False
        d['unavailable_reason'] = why
        d['withheld_by_score'] = True
        for k in ('percentile', 'percentile_benchmark', 'percentile_library', 'percentile_n',
                  'percentile_weight'):
            d.pop(k, None)
    for phase in (report.get('phases') or {}).values():
        phase['dimensions'] = [({**pd, **dims[pd['key']]} if pd.get('key') in withheld else pd)
                               for pd in phase.get('dimensions', [])]
        rated = [pd for pd in phase['dimensions'] if pd.get('available') is not False]
        if not rated:
            phase['avg_stars'] = None
            phase['badge'] = 'unavailable'
    if withheld:
        rx = [p for p in report.get('prescriptions', []) if p.get('key') not in withheld]
        for i, p in enumerate(rx):
            p['priority'] = i + 1
        report['prescriptions'] = rx
        num = den = 0.0
        for key, w in weights.items():
            d = dims.get(key) or {}
            if d.get('available') is False or not isinstance(d.get('percentile'), (int, float)) or w <= 0:
                continue
            num += d['percentile'] * w
            den += w
        if den > 0:
            report['overall_percentile'] = int(round(num / den))
        else:
            report['overall_percentile'] = None
    report['withheld_dimensions'] = sorted(withheld)
