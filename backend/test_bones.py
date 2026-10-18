"""Server-side bones for the 3-D view. Run: python test_bones.py (skips without opensim)."""
import numpy as np
import pandas as pd

import bones


def _mot(n=30):
    t = np.arange(n) / 60.0
    return pd.DataFrame({'time': t, 'pelvis_ty': 0.9 + 0 * t, 'knee_angle_r': np.linspace(0, 60, n),
                         'hip_flexion_r': np.linspace(0, 30, n), 'unknown_coordinate': 0 * t})


def test_bones_follow_the_skeleton_frames():
    if not bones.HAS_OPENSIM:
        print('opensim not installed: skipped'); return
    df = _mot()
    times = list(df['time'].values[::3])
    m = bones.bones_motion(df, times, contact_frame=4, height_m=1.85)
    assert m['format'] == 'biostrike.bones.v1' and m['model'] == 'generic'
    assert len(m['time']) == len(times) and m['contact_frame'] == 4
    for name in ('pelvis', 'femur_r', 'tibia_r', 'torso'):
        b = m['bodies'][name]
        assert len(b['rotation']) == len(times) and b['geometries']
    # generic model scaled to the athlete's height
    assert abs(m['bodies']['femur_r']['scale'][0] - 1.85 / 1.70) < 1e-3
    # the knee actually bends: the tibia turns relative to the femur over the swing
    tib = np.array(m['bodies']['tibia_r']['rotation']); fem = np.array(m['bodies']['femur_r']['rotation'])
    assert np.abs((tib - fem)[-1] - (tib - fem)[0]).max() > 0.5
    # pelvis height comes from the .mot
    assert abs(m['bodies']['pelvis']['translation'][0][1] - 0.9) < 1e-3


def test_attach_bones_never_raises():
    d = {'skeleton_frames': {'frame_times': [0.0, 0.1], 'contact_frame': 1}}
    bones.attach_bones(d, None, '/no/such/model.osim', 1.8)          # no data: no bones, no error
    assert 'bones' not in d
    bones.attach_bones({}, _mot(), None, 1.8)                         # no skeleton: nothing to do


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('✓', name)
