"""Pose the athlete's OpenSim model through the swing, for the 3-D bone view.

For each skeleton frame the report already carries, set the model's joint
coordinates from the .mot, realise positions, and record every body's
position and body-fixed XYZ rotation in ground. That is the data OpenCap's web
viewer animates (opencap-core generateVisualizerJson); the frontend places the
bone meshes from docs/bones/geometry.json at these transforms.

Model choice: the athlete's scaled .osim when one was uploaded; otherwise the
generic LaiUhlrich2022 model (OpenCap's model, opencap-core, Apache 2.0)
scaled uniformly to the athlete's height, which gets the proportions roughly
right but not the athlete's own segment lengths.

Optional: without the opensim package the report simply has no bones and the
viewer shows the stick figure.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    import opensim as osim
    HAS_OPENSIM = True
except Exception:          # pragma: no cover - depends on the deployment
    osim = None
    HAS_OPENSIM = False

HERE = os.path.dirname(os.path.abspath(__file__))
GENERIC_MODEL = os.path.join(HERE, 'models', 'LaiUhlrich2022.osim')
GENERIC_HEIGHT_M = 1.70       # the generic model's standing height
GEOMETRY_URL = 'bones/geometry.json'


def _mesh_info(body) -> tuple[list, list]:
    names, scale = [], [1.0, 1.0, 1.0]
    frame = osim.Frame.safeDownCast(body)
    for i in range(frame.getPropertyByName('attached_geometry').size()):
        g = body.get_attached_geometry(i)
        if g.getConcreteClassName() == 'Mesh':
            names.append(osim.Mesh.safeDownCast(g).getGeometryFilename())
            scale = list(g.get_scale_factors().to_numpy())
    return names, scale


def _load_model(model_path: Optional[str], height_m: Optional[float]):
    if model_path and os.path.exists(model_path):
        return osim.Model(model_path), 'uploaded'
    model = osim.Model(GENERIC_MODEL)
    state = model.initSystem()
    if height_m and abs(height_m / GENERIC_HEIGHT_M - 1) > 0.01:
        factor = height_m / GENERIC_HEIGHT_M
        scales = osim.ScaleSet()
        for body in model.getBodySet():
            s = osim.Scale()
            s.setSegmentName(body.getName())
            s.setScaleFactors(osim.Vec3(factor, factor, factor))
            s.setApply(True)
            scales.cloneAndAppend(s)
        model.scale(state, scales, True)
    return model, 'generic'


def bones_motion(mot_df, frame_times, contact_frame: int,
                 model_path: Optional[str] = None, height_m: Optional[float] = None) -> Optional[dict]:
    """biostrike.bones.v1 for the given frame times, or None when unavailable."""
    if not HAS_OPENSIM or mot_df is None or not len(frame_times):
        return None
    model, source = _load_model(model_path, height_m)
    model.finalizeConnections()
    state = model.initSystem()
    coords = model.getCoordinateSet()
    cols = [c for c in mot_df.columns if c != 'time' and coords.contains(c)]
    if not cols:
        return None
    times = mot_df['time'].to_numpy(dtype=float)
    values = mot_df[cols].to_numpy(dtype=float)
    rotational = [coords.get(c).getMotionType() == 1 for c in cols]   # 1 = rotational

    bodies, order = {}, []
    for body in model.getBodySet():
        meshes, scale = _mesh_info(body)
        if meshes:
            bodies[body.getName()] = {'geometries': meshes, 'scale': [round(x, 5) for x in scale],
                                      'rotation': [], 'translation': []}
            order.append(body)

    for t in frame_times:
        row = values[int(np.argmin(np.abs(times - t)))]
        for c, v, rot in zip(cols, row, rotational):
            coords.get(c).setValue(state, float(np.radians(v) if rot else v), False)
        model.realizePosition(state)
        for body in order:
            tf = body.getTransformInGround(state)
            r = tf.R().convertRotationToBodyFixedXYZ().to_numpy()
            p = tf.T().to_numpy()
            entry = bodies[body.getName()]
            entry['rotation'].append([round(float(x), 4) for x in r])
            entry['translation'].append([round(float(x), 4) for x in p])

    ft = np.asarray(frame_times, dtype=float)
    return {
        'format': 'biostrike.bones.v1',
        'model': source,
        'time': [round(float(x), 4) for x in ft],
        'fps': round(float(1 / np.median(np.diff(ft))), 2) if len(ft) > 1 else None,
        'contact_frame': int(contact_frame),
        'bodies': bodies,
    }


def attach_bones(diagnosis: dict, mot_df, model_path: Optional[str], height_m: Optional[float]) -> None:
    """Add diagnosis['bones'] for the report's skeleton frames; never raises."""
    sk = diagnosis.get('skeleton_frames') or {}
    times = sk.get('frame_times')
    if not times:
        return
    try:
        motion = bones_motion(mot_df, times, sk.get('contact_frame', 0), model_path, height_m)
    except Exception as e:  # a bad model must never break the analysis
        print(f'[bones] skipped: {e}')
        return
    if motion:
        diagnosis['bones'] = {'motion': motion, 'geometry_url': GEOMETRY_URL}
