"""Read an OpenCap session folder: which swings it holds, and who the athlete is.

An OpenCap session export looks like this (as downloaded from OpenCap or copied to Drive):

    MarkerData/<trial>.trc
    OpenSimData/Kinematics/<trial>.mot
    OpenSimData/Model/<model>.osim
    Videos/Cam0/InputMedia/<trial>/<trial>_sync.mp4   (and Cam1)
    sessionMetadata.yaml                              (height_m, mass_kg)

What OpenCap does not know comes from the folder name, e.g.
"Mason youth right 28in 18oz 10yo" (name first; level, side, bat and age in any order), or from
an athlete.yaml / athlete.txt / athlete.json (or a Google Doc named "athlete") at the
top of the folder, which wins over the folder name:

    name: Mason
    level: youth            # youth | high_school | college | professional
    handedness: right       # right | left | (blank to infer)
    age: 10                 # printed on the report only
    bat_length_in: 28
    bat_weight_oz: 18
    height_m: 1.51          # overrides sessionMetadata.yaml
    mass_kg: 39
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

LEVELS = ('youth', 'high_school', 'college', 'professional')
LEVEL_ALIASES = {'hs': 'high_school', 'high school': 'high_school', 'highschool': 'high_school',
                 'pro': 'professional', 'mlb': 'professional', 'milb': 'professional',
                 'little league': 'youth', 'kid': 'youth', 'kids': 'youth'}
# Trials that are not swings: OpenCap's calibration pose and static captures.
NOT_SWINGS = re.compile(r'^(neutral|static|calibration)', re.I)


@dataclass
class Athlete:
    name: str
    level: str = 'high_school'
    handedness: Optional[str] = None          # None: let the app infer it
    age: Optional[int] = None                 # printed on the report only
    height_m: Optional[float] = None
    mass_kg: Optional[float] = None
    bat_length_in: Optional[float] = None
    bat_weight_oz: Optional[float] = None


@dataclass
class Trial:
    name: str
    mot: str
    trc: Optional[str] = None
    cam0: Optional[str] = None
    cam1: Optional[str] = None


@dataclass
class Session:
    root: str
    athlete: Athlete
    model: Optional[str]
    trials: List[Trial] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)   # trials found but not analyzable, with why


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def normalise_level(level: Optional[str], default: str) -> str:
    if not level:
        return default
    v = str(level).strip().lower().replace('-', '_')
    v = LEVEL_ALIASES.get(v.replace('_', ' '), v)
    if v not in LEVELS:
        raise ValueError(f'unknown level {level!r}; use one of {", ".join(LEVELS)}')
    return v


def normalise_handedness(h) -> Optional[str]:
    if h is None or str(h).strip() == '':
        return None
    v = str(h).strip().lower()
    if v in ('r', 'right', 'rhh', 'righty'):
        return 'right'
    if v in ('l', 'left', 'lhh', 'lefty'):
        return 'left'
    raise ValueError(f'unknown handedness {h!r}; use right or left')


def _walk(root: str) -> List[str]:
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            out.append(os.path.join(d, f))
    return out


def parse_folder_name(folder: str) -> Dict:
    """Athlete details from a folder name like "Mason youth right 28in 18oz"."""
    out, name = {}, []
    words = re.split(r'[\s_,]+', folder.strip())
    i = 0
    while i < len(words):
        w, lw = words[i], words[i].lower()
        two = f'{lw} {words[i + 1].lower()}' if i + 1 < len(words) else ''
        if two in LEVEL_ALIASES or two.replace(' ', '_') in LEVELS:
            out['level'] = two; i += 2; continue
        if lw in LEVELS or lw in LEVEL_ALIASES:
            out['level'] = lw
        elif lw in ('r', 'right', 'rhh', 'righty', 'l', 'left', 'lhh', 'lefty'):
            out['handedness'] = lw
        elif re.fullmatch(r'\d+(\.\d+)?(in|\"|inch|inches)', lw):
            out['bat_length_in'] = float(re.match(r'[\d.]+', lw).group())
        elif re.fullmatch(r'\d{1,2}(yo|y/o|yrs?|years?)', lw):
            out['age'] = int(re.match(r'\d+', lw).group())
        elif re.fullmatch(r'\d+(\.\d+)?(oz|ounce|ounces)', lw):
            out['bat_weight_oz'] = float(re.match(r'[\d.]+', lw).group())
        elif not out and not re.fullmatch(r'[-–]', w):
            name.append(w)
        i += 1
    if name:
        out['name'] = ' '.join(name)
    return out


def _read_config(root: str) -> Dict:
    for name in ('athlete.yaml', 'athlete.yml', 'athlete.txt', 'athlete.json'):
        p = os.path.join(root, name)
        if os.path.exists(p):
            with open(p) as fh:
                data = json.load(fh) if name.endswith('.json') else yaml.safe_load(fh)
            return data or {}
    return {}


def _read_metadata(files: List[str]) -> Dict:
    meta = [f for f in files if os.path.basename(f) == 'sessionMetadata.yaml']
    if not meta:
        return {}
    with open(meta[0]) as fh:
        return yaml.safe_load(fh) or {}


def load_session(root: str, default_name: Optional[str] = None, default_level: str = 'high_school') -> Session:
    files = _walk(root)
    folder = default_name or os.path.basename(os.path.normpath(root))
    cfg = {**parse_folder_name(folder), **_read_config(root)}
    meta = _read_metadata(files)

    def num(*vals):
        for v in vals:
            if v not in (None, ''):
                return float(v)
        return None

    athlete = Athlete(
        name=str(cfg.get('name') or folder),
        level=normalise_level(cfg.get('level'), default_level),
        handedness=normalise_handedness(cfg.get('handedness')),
        height_m=num(cfg.get('height_m'), meta.get('height_m')),
        mass_kg=num(cfg.get('mass_kg'), meta.get('mass_kg')),
        bat_length_in=num(cfg.get('bat_length_in')),
        bat_weight_oz=num(cfg.get('bat_weight_oz')),
        age=int(cfg['age']) if cfg.get('age') not in (None, '') else None,
    )

    mots = {os.path.splitext(os.path.basename(f))[0]: f for f in files if f.endswith('.mot')}
    trcs = {os.path.splitext(os.path.basename(f))[0]: f for f in files if f.endswith('.trc')}
    osims = sorted(f for f in files if f.endswith('.osim'))
    # Prefer OpenCap's scaled model over any generic one that came along.
    model = next((f for f in osims if 'scaled' in os.path.basename(f).lower()), osims[0] if osims else None)

    def video(cam: str, trial: str) -> Optional[str]:
        want = f'{trial}_sync.mp4'
        for f in files:
            if os.path.basename(f) == want and f'{os.sep}{cam}{os.sep}' in f:
                return f
        return None

    session = Session(root=root, athlete=athlete, model=model)
    for name in sorted(set(mots) | set(trcs), key=natural_key):
        if NOT_SWINGS.match(name):
            continue
        if name not in mots:
            session.skipped.append(f'{name}: no .mot (OpenCap did not finish this trial)')
            continue
        session.trials.append(Trial(name=name, mot=mots[name], trc=trcs.get(name),
                                    cam0=video('Cam0', name), cam1=video('Cam1', name)))
    return session
