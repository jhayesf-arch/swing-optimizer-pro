"""
static_optimization.py
----------------------
Runs OpenSim Static Optimization on a swing .mot file to decompose net joint
torques into individual muscle forces and activations.

Static Optimization minimizes sum of squared muscle activations subject to:
  - Muscle force = activation × max_isometric_force × force-length-velocity factor
  - Net joint torque (from ID) = sum of muscle moment arms × muscle forces

Outputs per-muscle activation (0-1) and force (N) at each time frame.
Key muscles for baseball swing injury risk and performance:
  - Obliques (external/internal): lumbar rotation torque
  - Gluteus maximus/medius: hip extension and rotation power
  - Erector spinae: lumbar extension stability
  - Latissimus dorsi: shoulder internal rotation
  - Subscapularis/infraspinatus: shoulder rotator cuff loading

Reference:
  Anderson & Pandy (2001) Static and dynamic optimization solutions for gait
  are practically equivalent. J Biomech 34(2):153-161.
"""
import os
import shutil
import tempfile
import numpy as np
import pandas as pd

try:
    import opensim as osim
    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False

# Muscles most relevant to baseball swing biomechanics and injury risk
SWING_RELEVANT_MUSCLES = {
    # Lumbar rotation — oblique strain risk
    'extobl_r': 'External Oblique (R)',
    'extobl_l': 'External Oblique (L)',
    'intobl_r': 'Internal Oblique (R)',
    'intobl_l': 'Internal Oblique (L)',
    # Hip power generation
    'glmax1_r': 'Glut Max (R)',
    'glmax2_r': 'Glut Max (R)',
    'glmax3_r': 'Glut Max (R)',
    'glmax1_l': 'Glut Max (L)',
    'glmax2_l': 'Glut Max (L)',
    'glmax3_l': 'Glut Max (L)',
    'glmed1_r': 'Glut Med (R)',
    'glmed1_l': 'Glut Med (L)',
    # Lumbar stability
    'ercspn_r': 'Erector Spinae (R)',
    'ercspn_l': 'Erector Spinae (L)',
    # Hamstrings — trail leg bracing
    'bflh_r': 'Biceps Femoris LH (R)',
    'bfsh_r': 'Biceps Femoris SH (R)',
    # Quadriceps — lead leg bracing
    'recfem_l': 'Rectus Femoris (L)',
    'vasmed_l': 'Vastus Medialis (L)',
    'vaslat_l': 'Vastus Lateralis (L)',
}


def run_static_optimization(mot_path: str,
                             model_path: str,
                             id_sto_path: str = None,
                             lowpass_hz: float = 15.0) -> dict:
    """
    Run OpenSim Static Optimization on a swing .mot file.
    Trims to swing window before running to avoid pre-swing optimizer failures.
    """
    if not HAS_OPENSIM:
        raise RuntimeError("opensim not available")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Trim .mot to swing window (reuse opensim_id's trimmer)
        trimmed_mot = os.path.join(tmpdir, 'swing_trimmed.mot')
        try:
            from opensim_id import _trim_to_swing_window
            t_start, t_end = _trim_to_swing_window(mot_path, trimmed_mot)
        except Exception:
            shutil.copy(mot_path, trimmed_mot)
            df_mot = _load_mot(mot_path)
            t_start = float(df_mot['time'].iloc[0])
            t_end   = float(df_mot['time'].iloc[-1])

        mod_model_path = os.path.join(tmpdir, 'model_so.osim')
        _build_so_model(model_path, mod_model_path)

        setup_path = os.path.join(tmpdir, 'so_setup.xml')
        _write_so_setup(setup_path, mod_model_path, trimmed_mot,
                        tmpdir, t_start, t_end, lowpass_hz)

        analyze_tool = osim.AnalyzeTool(setup_path)
        analyze_tool.run()

        act_file   = os.path.join(tmpdir, 'StaticOptimization_activation.sto')
        force_file = os.path.join(tmpdir, 'StaticOptimization_force.sto')

        if not os.path.exists(act_file):
            raise RuntimeError("Static Optimization produced no output. "
                               "The optimizer likely failed — check reserve actuator strength.")

        act_df   = _load_sto(act_file)
        force_df = _load_sto(force_file) if os.path.exists(force_file) else None

        out_act   = mot_path.replace('.mot', '_SO_activation.sto')
        out_force = mot_path.replace('.mot', '_SO_force.sto')
        shutil.copy(act_file, out_act)
        if force_df is not None:
            shutil.copy(force_file, out_force)

    summary      = _summarize_activations(act_df, force_df)
    injury_flags = _flag_high_activation(act_df)

    return {
        'activations':   act_df,
        'forces':        force_df,
        'summary':       summary,
        'injury_flags':  injury_flags,
        'sto_path':      out_act,
    }


def _build_so_model(src_path: str, dst_path: str):
    """
    Copy model, add reserve actuators, and lock pelvis translations/rotations.
    Locking the pelvis allows SO to converge without force plates by treating
    the pelvis as a fixed base. This is physically approximate but gives valid
    upper-body (lumbar, shoulder, arm) muscle activations.
    Note: lower-extremity activations will be less accurate with a locked pelvis.
    """
    with open(src_path) as f:
        content = f.read()

    import re
    coords = re.findall(r'<Coordinate name="([^"]+)"', content)

    # Lock pelvis DOFs — these require GRFs to solve correctly
    pelvis_coords = {'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
                     'pelvis_tx', 'pelvis_ty', 'pelvis_tz'}

    # Add reserve actuators only for non-pelvis coordinates
    reserves_xml = ''
    for coord in coords:
        if coord in pelvis_coords:
            continue
        reserves_xml += f'''
        <CoordinateActuator name="reserve_{coord}">
            <coordinate>{coord}</coordinate>
            <optimal_force>500</optimal_force>
            <min_control>-Inf</min_control>
            <max_control>Inf</max_control>
        </CoordinateActuator>'''

    content = content.replace('</ForceSet>', reserves_xml + '\n</ForceSet>', 1)

    # Lock pelvis coordinates by setting <locked>true</locked>
    for coord in pelvis_coords:
        # Find the coordinate block and set locked=true
        pattern = f'<Coordinate name="{coord}">'
        if pattern in content:
            idx = content.find(pattern)
            block_end = content.find('</Coordinate>', idx)
            block = content[idx:block_end]
            if '<locked>' in block:
                block = re.sub(r'<locked>.*?</locked>', '<locked>true</locked>', block)
            else:
                block = block.rstrip() + '\n\t\t\t\t\t\t\t<locked>true</locked>\n'
            content = content[:idx] + block + content[block_end:]

    with open(dst_path, 'w') as f:
        f.write(content)


def _write_so_setup(setup_path: str, model_path: str, mot_path: str,
                    results_dir: str, t_start: float, t_end: float,
                    lowpass_hz: float):
    """Write the AnalyzeTool XML setup file for Static Optimization."""
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <AnalyzeTool name="StaticOptimization">
        <model_file>{model_path}</model_file>
        <coordinates_file>{mot_path}</coordinates_file>
        <lowpass_cutoff_frequency_for_coordinates>{lowpass_hz}</lowpass_cutoff_frequency_for_coordinates>
        <initial_time>{t_start}</initial_time>
        <final_time>{t_end}</final_time>
        <results_directory>{results_dir}</results_directory>
        <output_precision>8</output_precision>
        <AnalysisSet>
            <objects>
                <StaticOptimization name="StaticOptimization">
                    <start_time>{t_start}</start_time>
                    <end_time>{t_end}</end_time>
                    <step_interval>1</step_interval>
                    <in_degrees>true</in_degrees>
                    <use_model_force_set>true</use_model_force_set>
                    <activation_exponent>2</activation_exponent>
                    <optimizer_convergence_criterion>1e-004</optimizer_convergence_criterion>
                    <optimizer_max_iterations>100</optimizer_max_iterations>
                </StaticOptimization>
            </objects>
        </AnalysisSet>
    </AnalyzeTool>
</OpenSimDocument>"""
    with open(setup_path, 'w') as f:
        f.write(xml)


def _load_mot(path: str) -> pd.DataFrame:
    with open(path) as f:
        lines = f.readlines()
    he = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(path, sep='\t', skiprows=he, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def _load_sto(path: str) -> pd.DataFrame:
    with open(path) as f:
        lines = f.readlines()
    he = next(i for i, l in enumerate(lines) if 'endheader' in l.lower()) + 1
    df = pd.read_csv(path, sep='\t', skiprows=he, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return df


def _summarize_activations(act_df: pd.DataFrame,
                            force_df: pd.DataFrame = None) -> dict:
    """Peak activation and force for swing-relevant muscles."""
    summary = {}
    muscle_cols = [c for c in act_df.columns if c != 'time']

    for muscle_id, label in SWING_RELEVANT_MUSCLES.items():
        # Try exact match and prefix match (model may append _r/_l)
        matches = [c for c in muscle_cols if c == muscle_id or c.startswith(muscle_id)]
        if not matches:
            continue
        col = matches[0]
        peak_act = float(act_df[col].max())
        peak_force = float(force_df[col].max()) if force_df is not None and col in force_df.columns else None
        summary[col] = {
            'label':       label,
            'peak_activation': round(peak_act, 3),
            'peak_force_N':    round(peak_force, 1) if peak_force is not None else None,
        }
    return summary


def _flag_high_activation(act_df: pd.DataFrame, threshold: float = 0.8) -> list:
    """Flag muscles with peak activation above threshold — potential injury risk."""
    flags = []
    for col in [c for c in act_df.columns if c != 'time']:
        if act_df[col].max() > threshold:
            label = SWING_RELEVANT_MUSCLES.get(col, col)
            flags.append({
                'muscle': col,
                'label':  label,
                'peak_activation': round(float(act_df[col].max()), 3),
            })
    return flags


if __name__ == '__main__':
    import sys

    mot  = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        '~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae'
        '/OpenSimData/Kinematics/swing_lower_first.mot')
    model = os.path.expanduser(
        '~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae'
        '/OpenSimData/Model/LaiUhlrich2022_scaled.osim')

    print(f'Running Static Optimization: {os.path.basename(mot)}')
    result = run_static_optimization(mot, model)

    print(f'\nSaved: {result["sto_path"]}')
    print('\n=== PEAK MUSCLE ACTIVATIONS (swing-relevant) ===')
    for muscle, data in result['summary'].items():
        bar = '█' * int(data['peak_activation'] * 20)
        force_str = f"  {data['peak_force_N']:.0f} N" if data['peak_force_N'] else ''
        print(f"  {data['label']:<30} {data['peak_activation']:.2f}  {bar}{force_str}")

    if result['injury_flags']:
        print('\n⚠️  HIGH ACTIVATION FLAGS (>80%):')
        for flag in result['injury_flags']:
            print(f"  {flag['label']}: {flag['peak_activation']:.2f}")
    else:
        print('\n✅ No muscles flagged above 80% activation')
