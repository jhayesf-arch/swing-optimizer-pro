"""
static_optimization.py
----------------------
OpenSim Static Optimization for one swing: splits the joint moments from
inverse dynamics among the leg muscles, frame by frame, by minimising the sum
of squared activations (Anderson & Pandy 2001).

What the model can and cannot say. LaiUhlrich2022 (OpenCap) has 80 muscles,
all in the legs (40 per side). The trunk and arms are driven by torque
actuators, so there are no obliques, erector spinae, lats or rotator cuff here:
anything about those muscles would be invented. Outputs are therefore leg
muscle activations and forces (glutes, hamstrings, quads, adductors, calves),
which is where the lead-leg block and hip drive live.

Set-up, consistent with opensim_id.py:
  * Ground force: the model-consistent loads from run_inverse_dynamics (total
    from the model's own dynamics, split and placed by the marker estimate).
    The pelvis is free, not locked.
  * Pelvis residual actuators absorb what the loads cannot (the moment part).
  * Leg joints get reserve actuators with optimal force 1 N·m, so they are
    expensive and used only where the muscles cannot produce the moment; their
    share of the joint moment is reported, because a muscle solution leaning on
    reserves is not a muscle solution.
  * Trunk, arm and scapular coordinates are torque-driven (strong, unbounded).

Validated for gait-like tasks, not batting; and minimising activation
underestimates co-contraction, which bracing uses. Treat the numbers as an
experiment until checked against EMG.
"""
import os
import shutil
import tempfile
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

try:
    import opensim as osim
    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

# Muscle name prefix -> group, for the LaiUhlrich2022 / Rajagopal leg set.
MUSCLE_GROUPS = {
    'glmax': 'Glute max', 'glmed': 'Glute med', 'glmin': 'Glute min',
    'bflh': 'Hamstrings', 'bfsh': 'Hamstrings', 'semimem': 'Hamstrings', 'semiten': 'Hamstrings',
    'recfem': 'Quadriceps', 'vaslat': 'Quadriceps', 'vasmed': 'Quadriceps', 'vasint': 'Quadriceps',
    'addbrev': 'Adductors', 'addlong': 'Adductors', 'addmagDist': 'Adductors',
    'addmagIsch': 'Adductors', 'addmagMid': 'Adductors', 'addmagProx': 'Adductors', 'grac': 'Adductors',
    'iliacus': 'Hip flexors', 'psoas': 'Hip flexors', 'sart': 'Hip flexors', 'tfl': 'Hip flexors',
    'piri': 'Deep hip rotators',
    'gaslat': 'Calf', 'gasmed': 'Calf', 'soleus': 'Calf',
    'tibant': 'Shin/foot', 'tibpost': 'Shin/foot', 'perbrev': 'Shin/foot', 'perlong': 'Shin/foot',
    'edl': 'Shin/foot', 'ehl': 'Shin/foot', 'fdl': 'Shin/foot', 'fhl': 'Shin/foot',
}
PELVIS = ('pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz')
LEG_JOINTS = ('hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle',
              'ankle_angle', 'subtalar_angle', 'mtp_angle')
RESERVE_OPTIMAL_FORCE = 1.0      # N·m per unit control: expensive on purpose
RESIDUAL_OPTIMAL_FORCE = 1.0
TORQUE_DRIVEN_OPTIMAL_FORCE = 300.0
# A leg joint where reserves supply more than this share of the ID moment is
# reported as not explained by the muscles.
MAX_RESERVE_SHARE = 0.25
# Hicks et al. (2015) J Biomech Eng 137:020905, "Is my model good enough?":
# peak residual moments under ~1% of (centre-of-mass height x body weight).
# Larger means the applied ground force (split, centre of pressure) and the
# motion disagree, and the muscles are being asked to make up the difference.
RESIDUAL_MOMENT_LIMIT_FRAC = 0.01


def _group(muscle: str) -> Optional[str]:
    base = muscle.rsplit('_', 1)[0]
    for prefix in sorted(MUSCLE_GROUPS, key=len, reverse=True):
        if base.startswith(prefix):
            return MUSCLE_GROUPS[prefix]
    return None


def _build_so_model(src_model: str, dst_path: str) -> None:
    """Add pelvis residuals and leg reserves; make torque-driven DOFs unbounded."""
    model = osim.Model(src_model)
    fs = model.updForceSet()
    actuated = set()
    for i in range(fs.getSize()):
        ca = osim.CoordinateActuator.safeDownCast(fs.get(i))
        if ca is not None:
            ca.setMinControl(-np.inf)
            ca.setMaxControl(np.inf)
            ca.setOptimalForce(TORQUE_DRIVEN_OPTIMAL_FORCE)
            actuated.add(ca.get_coordinate())
    cs = model.getCoordinateSet()
    for i in range(cs.getSize()):
        c = cs.get(i)
        name = c.getName()
        if name in actuated or c.isConstrained(model.initSystem()) or name.endswith('_beta'):
            continue
        if name in PELVIS:
            label, opt = 'residual_' + name, RESIDUAL_OPTIMAL_FORCE
        elif name.rsplit('_', 1)[0] in LEG_JOINTS:
            label, opt = 'reserve_' + name, RESERVE_OPTIMAL_FORCE
        else:   # e.g. scapular translations of the shoulder model
            label, opt = 'torque_' + name, TORQUE_DRIVEN_OPTIMAL_FORCE
        act = osim.CoordinateActuator(name)
        act.setName(label)
        act.setOptimalForce(opt)
        act.setMinControl(-np.inf)
        act.setMaxControl(np.inf)
        model.addForce(act)
    model.finalizeConnections()
    model.printToXML(dst_path)


def _write_setup(path, model_path, mot_path, results_dir, t0, t1, lowpass_hz, loads_xml):
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
  <AnalyzeTool name="SO">
    <model_file>{model_path}</model_file>
    <replace_force_set>false</replace_force_set>
    <results_directory>{results_dir}</results_directory>
    <initial_time>{t0}</initial_time>
    <final_time>{t1}</final_time>
    <solve_for_equilibrium_for_auxiliary_states>true</solve_for_equilibrium_for_auxiliary_states>
    <AnalysisSet><objects>
      <StaticOptimization name="StaticOptimization">
        <on>true</on>
        <start_time>{t0}</start_time>
        <end_time>{t1}</end_time>
        <step_interval>1</step_interval>
        <in_degrees>true</in_degrees>
        <use_model_force_set>true</use_model_force_set>
        <activation_exponent>2</activation_exponent>
        <use_muscle_physiology>true</use_muscle_physiology>
        <optimizer_convergence_criterion>0.0001</optimizer_convergence_criterion>
        <optimizer_max_iterations>200</optimizer_max_iterations>
      </StaticOptimization>
    </objects></AnalysisSet>
    <external_loads_file>{loads_xml or ''}</external_loads_file>
    <coordinates_file>{mot_path}</coordinates_file>
    <lowpass_cutoff_frequency_for_coordinates>{lowpass_hz}</lowpass_cutoff_frequency_for_coordinates>
  </AnalyzeTool>
</OpenSimDocument>"""
    with open(path, 'w') as f:
        f.write(xml)


def _load_sto(path: str) -> pd.DataFrame:
    with open(path) as f:
        lines = f.readlines()
    he = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(path, sep='\t', skiprows=he, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def summarize(act: pd.DataFrame, force: pd.DataFrame, id_df: Optional[pd.DataFrame],
              handedness: Optional[str] = None) -> Dict:
    """Per-group peaks per leg, reserve reliance per joint, pelvis residuals."""
    lead = {'right': 'l', 'left': 'r'}.get((handedness or '').lower())
    side_name = (lambda s: ('Lead' if s == lead else 'Trail') if lead else ('Left' if s == 'l' else 'Right'))
    groups: Dict[str, Dict] = {}
    for col in act.columns:
        g = _group(col)
        if g is None or col[-2:] not in ('_l', '_r'):
            continue
        key = f'{side_name(col[-1])} {g.lower()}'
        e = groups.setdefault(key, {'peak_activation': 0.0, 'peak_force_N': 0.0, 'muscles': []})
        e['muscles'].append(col)
        e['peak_activation'] = round(max(e['peak_activation'], float(act[col].max())), 2)
        if force is not None and col in force:
            # Group force: per-frame sum across the group's muscles.
            e['_sum'] = e.get('_sum', 0) + force[col].values
    for e in groups.values():
        if '_sum' in e:
            e['peak_force_N'] = round(float(np.max(e.pop('_sum'))), 0)

    reserves = {}
    if force is not None:
        for col in force.columns:
            if not col.startswith('reserve_'):
                continue
            coord = col[len('reserve_'):]
            peak = float(np.max(np.abs(force[col].values)))
            share = None
            if id_df is not None:
                idc = coord + '_moment'
                if idc in id_df:
                    tot = float(np.max(np.abs(id_df[idc].values)))
                    share = round(peak / tot, 2) if tot > 1 else None
            reserves[coord] = {'peak_Nm': round(peak, 1), 'share_of_id_moment': share}
    residuals = {}
    if force is not None:
        for col in force.columns:
            if col.startswith('residual_'):
                residuals[col[len('residual_'):]] = round(float(np.max(np.abs(force[col].values))), 1)
    leaning = [c for c, r in reserves.items()
               if r['share_of_id_moment'] is not None and r['share_of_id_moment'] > MAX_RESERVE_SHARE]
    return {'groups': dict(sorted(groups.items())), 'reserves': reserves,
            'pelvis_residuals': residuals, 'joints_not_explained_by_muscles': sorted(leaning)}


def run_static_optimization(mot_path: str, model_path: str,
                            external_loads: Optional[dict] = None,
                            time_range: Optional[tuple] = None,
                            bat_mass_kg: float = 0.88, bat_length_m: float = 0.864,
                            handedness: Optional[str] = None,
                            lowpass_hz: Optional[float] = None) -> Dict:
    """Static optimization on the same window, model and ground force as ID."""
    if not HAS_OPENSIM:
        raise RuntimeError('opensim not available')
    from opensim_id import (run_inverse_dynamics, _trim_to_swing_window,
                            _build_modified_model, write_external_loads)
    t_begin = time.time()
    idr = run_inverse_dynamics(mot_path, model_path=model_path, bat_mass_kg=bat_mass_kg,
                               bat_length_m=bat_length_m, external_loads=external_loads,
                               time_range=time_range,
                               **({'lowpass_hz': lowpass_hz} if lowpass_hz else {}))
    with tempfile.TemporaryDirectory() as tmp:
        trimmed = os.path.join(tmp, 'swing.mot')
        t0, t1 = _trim_to_swing_window(mot_path, trimmed, time_range=time_range)
        bat_model = os.path.join(tmp, 'model_bat.osim')
        _build_modified_model(model_path, bat_model, bat_mass_kg, bat_length_m)
        so_model = os.path.join(tmp, 'model_so.osim')
        _build_so_model(bat_model, so_model)
        loads_xml = None
        if idr.get('external_loads') is not None:
            loads_xml = write_external_loads(idr['external_loads'], tmp)
        setup = os.path.join(tmp, 'so_setup.xml')
        _write_setup(setup, so_model, trimmed, tmp, t0, t1, idr['lowpass_hz'], loads_xml)
        t_so = time.time()
        osim.AnalyzeTool(setup).run()
        so_seconds = time.time() - t_so
        act_f = os.path.join(tmp, 'SO_StaticOptimization_activation.sto')
        frc_f = os.path.join(tmp, 'SO_StaticOptimization_force.sto')
        if not os.path.exists(act_f):
            raise RuntimeError('Static optimization produced no output.')
        act, force = _load_sto(act_f), (_load_sto(frc_f) if os.path.exists(frc_f) else None)
        out_act = mot_path.replace('.mot', '_SO_activation.sto')
        out_frc = mot_path.replace('.mot', '_SO_force.sto')
        shutil.copy(act_f, out_act)
        if force is not None:
            shutil.copy(frc_f, out_frc)
    summary = summarize(act, force, idr['dataframe'], handedness)
    chk = idr.get('grf_check')
    m = osim.Model(model_path)
    st = m.initSystem()
    com_h = float(m.calcMassCenterPosition(st).get(1))
    limit = RESIDUAL_MOMENT_LIMIT_FRAC * idr['body_weight_N'] * com_h
    res_m = max([v for k, v in summary['pelvis_residuals'].items()
                 if k in ('pelvis_tilt', 'pelvis_list', 'pelvis_rotation')] or [0.0])
    summary['residual_moment_check'] = {'peak_Nm': res_m, 'limit_Nm': round(limit, 1),
                                        'ok': res_m <= limit}
    reliable = (bool(chk is None or chk['plausible'])
                and not summary['joints_not_explained_by_muscles']
                and res_m <= limit)
    return {'summary': summary, 'grf_check': chk, 'grf_source': idr.get('grf_source'),
            'reliable': reliable, 'window_s': (t0, t1), 'frames': len(act),
            'seconds_total': round(time.time() - t_begin, 1), 'seconds_so': round(so_seconds, 1),
            'activation_sto': out_act, 'force_sto': out_frc if force is not None else None}


def _cli():
    import argparse
    p = argparse.ArgumentParser(description='Static optimization (leg muscles) for one swing.')
    p.add_argument('mot')
    p.add_argument('--trc', help='markers, for the ground force (strongly recommended)')
    p.add_argument('--model', required=True, help='athlete-scaled .osim')
    p.add_argument('--handedness', choices=['right', 'left'])
    p.add_argument('--height', type=float, default=1.83)
    a = p.parse_args()
    loads = tr = None
    if a.trc:
        from analyzer import RefinedHittingOptimizer
        from main import _crop_to_swing, _time_range
        m = osim.Model(a.model)
        opt = RefinedHittingOptimizer(body_mass_kg=m.getTotalMass(m.initSystem()), body_height_m=a.height)
        kin, trc = _crop_to_swing(opt.load_mot_file(a.mot), opt.load_trc_file(a.trc))
        loads = opt.comprehensive_diagnosis(kin, os.path.basename(a.mot), trc_data=trc,
                                            model_path=a.model).get('_foot_loads')
        tr = _time_range(kin)
    r = run_static_optimization(a.mot, a.model, external_loads=loads, time_range=tr,
                                handedness=a.handedness)
    s = r['summary']
    print(f"\nWindow {r['window_s'][0]:.2f}-{r['window_s'][1]:.2f} s, {r['frames']} frames; "
          f"static optimization {r['seconds_so']} s ({r['seconds_total']} s with ID)")
    if r['grf_check']:
        c = r['grf_check']
        print(f"Ground force check: {'plausible' if c['plausible'] else 'NOT plausible: ' + '; '.join(c['reasons'])}")
    print('\nPeak activation / peak group force:')
    for g, e in s['groups'].items():
        print(f"  {g:28s} {e['peak_activation']:5.2f}  {e['peak_force_N']:7.0f} N")
    print('\nReserve torque needed (share of the ID moment):')
    for c, e in s['reserves'].items():
        sh = '' if e['share_of_id_moment'] is None else f"{100 * e['share_of_id_moment']:.0f}%"
        print(f"  {c:20s} {e['peak_Nm']:8.1f} N·m  {sh}")
    print('\nPelvis residuals:', s['pelvis_residuals'])
    rc = s['residual_moment_check']
    print(f"Residual moment {rc['peak_Nm']:.0f} N·m vs limit {rc['limit_Nm']:.0f} N·m "
          f"(Hicks et al. 2015): {'ok' if rc['ok'] else 'too large'}")
    why = []
    if r['grf_check'] and not r['grf_check']['plausible']:
        why.append('implausible ground force')
    if s['joints_not_explained_by_muscles']:
        why.append('reserves carry ' + ', '.join(s['joints_not_explained_by_muscles']))
    if not rc['ok']:
        why.append('pelvis residual moment too large')
    print('\nReliable:', r['reliable'], f"({'; '.join(why)})" if why else '')


if __name__ == '__main__':
    _cli()
