"""Side-by-side comparison of lower-body dynamics from independent sources.

Three sources can appear, each optional:

  ours       bottom-up ID from the CoM-derived GRF (bottom_up_id.py) — needs .trc
  opensim    OpenSim Inverse Dynamics with that same GRF applied as external
             loads (opensim_id.py) — needs opensim installed + a scaled .osim
  reference  anything uploaded alongside the trial: Cortex force-plate values,
             an OpenCap / OpenSimAD simulation, a lab's ID output

The point is attribution. ours and opensim share the GRF but compute moments
differently, so disagreement between them isolates the moment chain. A
force-plate reference measures GRF independently, so disagreement there
isolates our GRF estimate and per-foot split.

Reference formats (see DYNAMICS_REFERENCE.md):
  .json  schema "biostrike.dynamics.v1" — peaks, time series, or both
  .sto   an OpenSim Inverse Dynamics output file
"""

from __future__ import annotations
import json
import os
from typing import Dict, Optional

import numpy as np

SCHEMA = 'biostrike.dynamics.v1'

# (key template, label, unit). {s} is the side: 'l' or 'r'.
QUANTITIES = [
    ('grf_vert_{s}_N',     'peak vertical GRF', 'N'),
    ('hip_moment_{s}_Nm',  'hip moment',        'N·m'),
    ('knee_moment_{s}_Nm', 'knee moment',       'N·m'),
    ('ankle_moment_{s}_Nm', 'ankle moment',     'N·m'),
]
KNOWN_KEYS = {q[0].format(s=s) for q in QUANTITIES for s in ('l', 'r')}


# ── Loading a reference ────────────────────────────────────────────────────

def _peak_of(series) -> Optional[float]:
    """Peak |value| of a scalar series, or peak norm of a vector series."""
    a = np.asarray(series, dtype=float)
    if a.size == 0:
        return None
    if a.ndim == 2:
        return float(np.nanmax(np.linalg.norm(a, axis=1)))
    return float(np.nanmax(np.abs(a)))


def _peaks_from_json(doc: dict) -> Dict[str, float]:
    peaks = {k: float(v) for k, v in (doc.get('peaks') or {}).items()
             if k in KNOWN_KEYS and v is not None}
    ts = doc.get('timeseries') or {}
    for k, series in ts.items():
        if k in KNOWN_KEYS and k not in peaks:
            p = _peak_of(series)
            if p is not None:
                peaks[k] = p
    # Full 3-D foot forces (Y up): the vertical peak is the max of column 1.
    for s in ('l', 'r'):
        vec = ts.get(f'grf_{s}_N')
        key = f'grf_vert_{s}_N'
        if vec is not None and key not in peaks:
            a = np.asarray(vec, dtype=float)
            if a.ndim == 2 and a.shape[1] == 3:
                peaks[key] = float(np.nanmax(a[:, 1]))
    return peaks


def load_reference(path: str, body_mass_kg: Optional[float] = None) -> Dict:
    """Load a reference dynamics file. Raises ValueError with a readable reason."""
    name = os.path.basename(path)
    ext = os.path.splitext(name)[1].lower()

    if ext == '.json':
        with open(path) as f:
            try:
                doc = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f'{name}: not valid JSON ({e.msg}, line {e.lineno})')
        if doc.get('schema') not in (None, SCHEMA):
            raise ValueError(f'{name}: schema "{doc.get("schema")}" is not {SCHEMA}')
        peaks = _peaks_from_json(doc)
        if not peaks:
            raise ValueError(f'{name}: no recognised quantities. Expected keys like '
                             f'hip_moment_l_Nm or grf_vert_r_N — see DYNAMICS_REFERENCE.md')
        return {
            'source': str(doc.get('source') or 'reference'),
            'label': str(doc.get('label') or name),
            'peaks': peaks,
            'file': name,
        }

    if ext == '.sto':
        from opensim_id import _load_sto, summarize_id_magnitudes
        try:
            df = _load_sto(path)
        except Exception as e:
            raise ValueError(f'{name}: could not read as an OpenSim .sto ({e})')
        mag = summarize_id_magnitudes(df, body_mass_kg)
        peaks = {k: v for k, v in mag.items() if k in KNOWN_KEYS and v}
        if not peaks:
            raise ValueError(f'{name}: no hip/knee/ankle moment columns found — is this '
                             f'an Inverse Dynamics output?')
        return {'source': 'opensim_sto', 'label': f'{name} (OpenSim ID)',
                'peaks': peaks, 'file': name,
                'residuals': {k: mag[k] for k in mag if k.startswith('pelvis_residual')}}

    raise ValueError(f'{name}: unsupported reference type {ext} (use .json or .sto)')


def reference_stem(filename: str) -> str:
    """Pairing key for a reference file in a batch upload.

    Strips the suffixes people naturally add — Emilio_1_10_cortex.json,
    Emilio_1_10_ID.sto — so the reference pairs with Emilio_1_10.mot.
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    for suf in ('_dynamics', '_reference', '_ref', '_cortex', '_forceplate',
                '_forces', '_grf', '_id', '_opensim', '_sim'):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


# ── Our side ───────────────────────────────────────────────────────────────

def ours_from_metrics(metrics: dict) -> Dict[str, float]:
    m = metrics or {}
    out = {}
    for s in ('l', 'r'):
        out[f'grf_vert_{s}_N'] = m.get(f'peak_grf_vert_{s}_N', 0.0)
        out[f'hip_moment_{s}_Nm'] = m.get(f'peak_hip_moment_id_{s}_Nm', 0.0)
        out[f'knee_moment_{s}_Nm'] = m.get(f'peak_knee_moment_id_{s}_Nm', 0.0)
        out[f'ankle_moment_{s}_Nm'] = m.get(f'peak_ankle_moment_id_{s}_Nm', 0.0)
    return {k: float(v) for k, v in out.items() if v}


# ── Comparison ─────────────────────────────────────────────────────────────

def build_comparison(metrics: dict,
                     opensim_mag: Optional[dict] = None,
                     reference: Optional[dict] = None,
                     handedness: Optional[str] = None) -> Optional[Dict]:
    """Rows of per-joint peaks from every source present.

    Returns None unless at least two sources have data — a one-column table is
    not a comparison.
    """
    sources = []
    values = {}
    ours = ours_from_metrics(metrics)
    if ours:
        sources.append({'key': 'ours', 'label': 'Bottom-up ID (ours)'})
        values['ours'] = ours
    if opensim_mag:
        os_peaks = {k: v for k, v in opensim_mag.items() if k in KNOWN_KEYS and v}
        if os_peaks:
            sources.append({'key': 'opensim', 'label': 'OpenSim ID (same GRF)'})
            values['opensim'] = os_peaks
    if reference and reference.get('peaks'):
        sources.append({'key': 'reference', 'label': reference.get('label', 'Reference')})
        values['reference'] = reference['peaks']
    if len(sources) < 2:
        return None

    hd = (handedness or '').lower()
    lead = {'right': 'l', 'left': 'r'}.get(hd)

    def side_label(s):
        if lead is None:
            return 'Left' if s == 'l' else 'Right'
        return 'Lead' if s == lead else 'Trail'

    # Percent difference is taken against the most independent source present:
    # the uploaded reference first, OpenSim second.
    baseline = 'reference' if 'reference' in values else 'opensim'

    rows = []
    sides = ('l', 'r') if lead != 'r' else ('r', 'l')    # lead side first
    for tmpl, label, unit in QUANTITIES:
        for s in sides:
            key = tmpl.format(s=s)
            vals = {src['key']: values[src['key']].get(key) for src in sources}
            if all(v is None for v in vals.values()):
                continue
            o, b = vals.get('ours'), vals.get(baseline)
            delta = ((o - b) / abs(b) * 100.0) if (o and b) else None
            rows.append({'key': key, 'quantity': f'{side_label(s)} {label}',
                         'unit': unit, 'values': vals,
                         'delta_pct': None if delta is None else round(delta, 1)})

    notes = []
    if 'opensim' in values:
        notes.append('OpenSim reports the knee on its flexion axis only; the bottom-up '
                     'value is the full 3-D moment, so expect ours ≥ OpenSim at the knee.')
    return {
        'sources': sources,
        'baseline': baseline,
        'rows': rows,
        'residuals': (opensim_mag or {}) and {
            k: v for k, v in opensim_mag.items() if k.startswith('pelvis_residual')},
        'notes': notes,
    }
