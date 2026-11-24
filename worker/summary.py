"""A one-page summary per athlete: a score table built by code, plus a coach's note written by Claude.

Same rule as the in-app coach (backend/coach.py): the model never computes
biomechanics. Every number in the summary comes from the analysis; the table is
built here, and the note may only cite values it is handed. Parts the app skipped
as unreliable are passed with their reason, so the note can say "don't trust
this swing" instead of coaching on a bad capture.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .session import Athlete

MODEL = os.environ.get('SUMMARY_MODEL', 'claude-opus-5-5')
EFFORT = os.environ.get('SUMMARY_EFFORT', 'medium')

SYSTEM_PROMPT = """You write the summary a hitting coach sends after a motion-capture session. You are given every number from the session's analysis, computed by a physics engine from OpenCap motion capture.

Rules for numbers:
- Cite only values that appear in the SESSION block. Do not calculate, estimate, average or infer any value, and do not convert units.
- A part marked "skipped" was not measured reliably on that swing. Never coach on a skipped value; at most say it could not be measured and give the plain-language reason.
- A swing with no score was withheld because the capture was unreliable. Say so, and suggest re-filming it rather than reading anything into it.

What to write (Markdown, under 350 words, no title, no score table; the table is added separately):
1. **Summary**: two or three sentences on how this hitter's swings looked overall, judged against their level.
2. **What's working**: two or three bullets, each tied to a scored value.
3. **Top priority**: one fix, the one with the most room to improve among parts that were actually scored on most swings. Give the cue and the drill from the prescriptions as written, and say why in one sentence.
4. **Capture notes**: which swings to ignore or re-film and why, in one or two bullets. Leave this out if every swing was scored.

Write for the hitter and their coach in plain words. If the level is youth, write so a parent can follow it, and keep the tone encouraging. No jargon without a short definition in the same sentence."""


def _component_rows(detail: Optional[Dict]) -> List[Dict]:
    rows = []
    for c in (detail or {}).get('components', []):
        row = {'part': c.get('label'), 'value': c.get('value'), 'unit': c.get('unit'),
               'status': c.get('status')}
        if c.get('status') == 'scored':
            row['score_0_100'] = c.get('score')
        elif c.get('reason'):
            row['why_skipped'] = c.get('reason')
        rows.append(row)
    return rows


def swing_grounding(result: Dict) -> Dict:
    """The facts about one swing that the note may use."""
    if 'error' in result:
        return {'swing': result['trial'], 'analyzed': False, 'why': result['error']}
    d = result.get('analysis') or {}
    rep = d.get('phase_report') or {}
    detail = rep.get('swing_score_detail') or {}
    scored = {c.get('key') for c in detail.get('components', []) if c.get('status') == 'scored'}
    mech = d.get('mechanics_score') or {}
    cq = d.get('capture_quality') or {}
    return {
        'swing': result['trial'],
        'analyzed': True,
        'swing_score': detail.get('score'),
        'swing_score_note': detail.get('note'),
        'mechanics_score': mech.get('score'),
        'hand_speed_mph': (d.get('metrics') or {}).get('max_hand_speed_mph'),
        'capture_usable': cq.get('usable'),
        'capture_problems': cq.get('reasons') or [],
        'parts': _component_rows(detail),
        # Only advice for parts that were measured on this swing.
        'prescriptions': [{k: p.get(k) for k in ('label', 'cue', 'drill', 'why')}
                          for p in rep.get('prescriptions', []) if p.get('key') in scored][:3],
    }


def grounding(athlete: Athlete, results: List[Dict], skipped: List[str]) -> Dict:
    return {
        'athlete': {'name': athlete.name, 'level': athlete.level.replace('_', ' '),
                    'handedness': athlete.handedness or 'inferred from the capture',
                    'bat_length_in': athlete.bat_length_in, 'bat_weight_oz': athlete.bat_weight_oz},
        'swings': [swing_grounding(r) for r in results],
        'trials_not_processed': skipped,
    }


def _fmt(v, nd=0):
    try:
        return f'{float(v):.{nd}f}'
    except (TypeError, ValueError):
        return '—'


def score_table(results: List[Dict]) -> str:
    lines = ['| Swing | Swing Score | Mechanics | Hand speed (mph) | Notes |',
             '|---|---|---|---|---|']
    for r in results:
        if 'error' in r:
            lines.append(f"| {r['trial']} | — | — | — | not analyzed: {r['error']} |")
            continue
        g = swing_grounding(r)
        note = '' if g['swing_score'] is not None else 'score withheld: capture unreliable'
        lines.append(f"| {r['trial']} | {_fmt(g['swing_score'])} | {_fmt(g['mechanics_score'])} | "
                     f"{_fmt(g['hand_speed_mph'], 1)} | {note} |")
    return '\n'.join(lines)


def coach_note(facts: Dict) -> Optional[str]:
    """Claude's narrative, or None when no API key is configured or the call fails."""
    if not (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN')):
        return None
    import anthropic

    client = anthropic.Anthropic()
    try:
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            output_config={'effort': EFFORT},
            # If a safety classifier declines, the API retries on a suitable model itself.
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
            messages=[{'role': 'user', 'content':
                       'SESSION\n```json\n' + json.dumps(facts, indent=1, default=str) + '\n```'}],
        )
    except anthropic.APIStatusError as e:
        return f'_The written summary could not be generated (API error {e.status_code})._'
    except anthropic.APIConnectionError:
        return '_The written summary could not be generated (could not reach the API)._'
    if resp.stop_reason == 'refusal':
        return '_The written summary could not be generated for this session._'
    return ''.join(b.text for b in resp.content if b.type == 'text').strip()


def build_summary(athlete: Athlete, results: List[Dict], skipped: List[str]) -> str:
    facts = grounding(athlete, results, skipped)
    parts = [f'# {athlete.name}: session summary',
             f"Level: {facts['athlete']['level']} · Bats: {facts['athlete']['handedness']}"
             + (f" · Bat: {athlete.bat_length_in:g} in" if athlete.bat_length_in else '')
             + (f" / {athlete.bat_weight_oz:g} oz" if athlete.bat_weight_oz else ''),
             '']
    note = coach_note(facts)
    if note:
        parts += [note, '']
    parts += ['## Every swing', '', score_table(results)]
    if skipped:
        parts += ['', 'Not processed: ' + '; '.join(skipped)]
    parts += ['', '_Scores are relative to published values for the selected level. '
                  'A withheld score means the motion capture for that swing was not reliable enough to grade._']
    return '\n'.join(parts) + '\n'
