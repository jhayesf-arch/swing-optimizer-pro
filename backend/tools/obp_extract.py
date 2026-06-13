#!/usr/bin/env python3
"""
Extract OpenSim-style .trc (markers) and .mot (joint kinematics) files from the
Driveline OpenBiomechanics (OBP) hitting dataset for ingestion by Swing Optimizer Pro.

Source data (OBP repo, baseball_hitting/data, unzip the full_sig archives):
  - full_sig/landmarks.csv     joint-centre XYZ, metres   -> .trc markers + pelvis translation
  - full_sig/joint_angles.csv  segment/joint Euler angles, degrees -> .mot rotational DOFs
  - metadata.csv               per-swing strata (level, side, age, height, weight, exit velo)

Coordinate handling
  Driveline global: +x -> pitcher, +y -> RH batter's box, +z -> up.
  App frame:        +x -> right,   +y -> up,                +z -> toward pitcher.
  => marker app(x,y,z) = (drv_y, drv_z, drv_x).
  Rotational DOFs come straight from Driveline's validated joint_angles (cleaner than
  deriving from short joint-centre segments). Lead/rear are mapped *functionally*
  (rear -> _r/top-hand side, lead -> _l) so the mapping is handedness-agnostic; the
  analyzer keys off rotation magnitudes/ranges, so per-swing sign is irrelevant.

Usage:
  python3 obp_extract.py --landmarks landmarks.csv --joint-angles joint_angles.csv \
      --metadata metadata.csv --out ./obp_out [--per-level 3] [--limit N] \
      [--session-swings 6_1,103_1]

Outputs:  <out>/trc/<id>.trc, <out>/mot/<id>.mot, <out>/manifest.csv
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# Driveline landmark prefix -> app marker name
LANDMARK_TO_MARKER = {
    'rsjc': 'RShoulder', 'lsjc': 'LShoulder', 'rejc': 'RElbow', 'lejc': 'LElbow',
    'rwjc': 'RWrist', 'lwjc': 'LWrist', 'rhjc': 'RHip', 'lhjc': 'LHip',
    'rkjc': 'RKnee', 'lkjc': 'LKnee', 'rajc': 'RAnkle', 'lajc': 'LAnkle',
    'thorax_prox': 'Neck',
}
MARKER_ORDER = ['Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist',
                'midHip', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'Bat']

# app .mot column  <-  Driveline joint_angles column   (rear=top-hand side -> _r)
MOT_FROM_JA = {
    'pelvis_tilt': 'pelvis_angle_x', 'pelvis_list': 'pelvis_angle_y', 'pelvis_rotation': 'pelvis_angle_z',
    'lumbar_extension': 'torso_pelvis_angle_x', 'lumbar_bending': 'torso_pelvis_angle_y',
    'lumbar_rotation': 'torso_pelvis_angle_z',
    'hip_flexion_r': 'rear_hip_angle_x', 'hip_flexion_l': 'lead_hip_angle_x',
    'knee_angle_r': 'rear_knee_angle_x', 'knee_angle_l': 'lead_knee_angle_x',
    'arm_flex_r': 'rear_shoulder_angle_x', 'arm_flex_l': 'lead_shoulder_angle_x',
    'elbow_flex_r': 'rear_elbow_angle_x', 'elbow_flex_l': 'lead_elbow_angle_x',
}
MOT_COLS = (['time', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz'] + list(MOT_FROM_JA.keys()))


def to_app(x, y, z):
    return np.array([y, z, x], dtype=float)  # (drv_y, drv_z, drv_x)


def build_markers_for_frame(row):
    m = {}
    for prefix, name in LANDMARK_TO_MARKER.items():
        m[name] = to_app(row[f'{prefix}_x'], row[f'{prefix}_y'], row[f'{prefix}_z'])
    m['midHip'] = (m['RHip'] + m['LHip']) / 2.0
    if 'sweet_spot_x' in row and not pd.isna(row.get('sweet_spot_x', np.nan)):
        m['Bat'] = to_app(row['sweet_spot_x'], row['sweet_spot_y'], row['sweet_spot_z'])
    return m


def write_trc(path, name, frames_markers, times, markers):
    n = len(times)
    rate = 1.0 / np.median(np.diff(times)) if n > 1 else 240.0
    with open(path, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{name}.trc\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate:.4f}\t{rate:.4f}\t{n}\t{len(markers)}\tm\t{rate:.4f}\t1\t{n}\n")
        hdr = ['Frame#', 'Time']
        for mname in markers:
            hdr += [mname, '', '']
        f.write('\t'.join(hdr) + '\n')
        sub = ['', '']
        for i in range(len(markers)):
            sub += [f'X{i+1}', f'Y{i+1}', f'Z{i+1}']
        f.write('\t'.join(sub) + '\n')
        for i in range(n):
            r = [str(i + 1), f'{times[i]:.5f}']
            mk = frames_markers[i]
            for mname in markers:
                p = mk.get(mname)
                r += (['', '', ''] if p is None else [f'{p[0]:.6f}', f'{p[1]:.6f}', f'{p[2]:.6f}'])
            f.write('\t'.join(r) + '\n')


def write_mot(path, name, rows, events):
    n = len(rows)
    with open(path, 'w') as f:
        f.write(f"{name}\nversion=1\nnRows={n}\nnColumns={len(MOT_COLS)}\ninDegrees=yes\n")
        for k, v in events.items():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                f.write(f"{k}={v}\n")
        f.write("endheader\n")
        f.write('\t'.join(MOT_COLS) + '\n')
        for row in rows:
            f.write('\t'.join(f'{row[c]:.6f}' for c in MOT_COLS) + '\n')


def filter_csv(path, ids):
    keep = []
    for chunk in pd.read_csv(path, chunksize=200_000):
        keep.append(chunk[chunk['session_swing'].isin(ids)])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--landmarks', required=True)
    ap.add_argument('--joint-angles', required=True)
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--out', default='./obp_out')
    ap.add_argument('--per-level', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--session-swings', default='')
    args = ap.parse_args()

    meta = {r['session_swing']: r for r in csv.DictReader(open(args.metadata))}

    if args.session_swings:
        selected = [s for s in args.session_swings.split(',') if s in meta]
    else:
        selected, per_level = [], defaultdict(int)
        for sid, r in meta.items():
            lvl = r['highest_playing_level']
            if args.per_level and per_level[lvl] >= args.per_level:
                continue
            selected.append(sid); per_level[lvl] += 1
        if args.limit:
            selected = selected[:args.limit]
    sel = set(selected)
    print(f"Selected {len(sel)} swings.")

    os.makedirs(os.path.join(args.out, 'trc'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'mot'), exist_ok=True)

    lm = filter_csv(args.landmarks, sel)
    ja = filter_csv(args.joint_angles, sel)
    print(f"Loaded {len(lm)} landmark rows, {len(ja)} joint-angle rows.")

    manifest = []
    for sid in sorted(sel):
        lg = lm[lm['session_swing'] == sid].sort_values('time').reset_index(drop=True)
        ag = ja[ja['session_swing'] == sid].sort_values('time').reset_index(drop=True)
        if len(lg) == 0 or len(ag) == 0:
            print(f"  {sid}: SKIP (missing landmarks or joint angles)"); continue
        r = meta[sid]

        # ---- .trc from landmark joint centres ----
        times = lg['time'].to_numpy(dtype=float)
        frames = [build_markers_for_frame(lg.iloc[i]) for i in range(len(lg))]
        present = [m for m in MARKER_ORDER if all(m in fm for fm in frames)]
        trc_path = os.path.join(args.out, 'trc', f'{sid}.trc')
        write_trc(trc_path, sid, frames, times, present)

        # ---- .mot: rotations from joint_angles, translation from landmark midHip,
        #      aligned to the joint-angle time base via nearest-time merge ----
        midhip = np.array([(build_markers_for_frame(lg.iloc[i])['midHip']) for i in range(len(lg))])
        lm_t = times
        rows = []
        for i in range(len(ag)):
            t = float(ag['time'].iloc[i])
            j = int(np.argmin(np.abs(lm_t - t)))
            row = {'time': t, 'pelvis_tx': midhip[j, 0], 'pelvis_ty': midhip[j, 1], 'pelvis_tz': midhip[j, 2]}
            for mot_col, ja_col in MOT_FROM_JA.items():
                row[mot_col] = float(ag[ja_col].iloc[i]) if ja_col in ag else 0.0
            rows.append(row)
        mot_path = os.path.join(args.out, 'mot', f'{sid}.mot')
        events = {k: (ag[k].iloc[0] if k in ag else None) for k in ('fp_10_time', 'fp_100_time', 'contact_time')}
        write_mot(mot_path, sid, rows, events)

        manifest.append({
            'session_swing': sid, 'level': r.get('highest_playing_level'), 'side': r.get('hitter_side'),
            'age': r.get('athlete_age'), 'height_in': r.get('session_height_in'),
            'weight_lbs': r.get('session_mass_lbs'), 'bat_weight_oz': r.get('bat_weight_oz'),
            'bat_length_in': r.get('bat_length_in'), 'bat_speed_mph': r.get('bat_speed_mph_max_x'),
            'exit_velo_mph': r.get('exit_velo_mph_x'), 'n_frames': len(lg),
            'trc': os.path.relpath(trc_path, args.out), 'mot': os.path.relpath(mot_path, args.out),
        })
        print(f"  {sid}: {len(lg)} fr, {len(present)} mk ({r.get('highest_playing_level')}, {r.get('hitter_side')})")

    if manifest:
        mf = os.path.join(args.out, 'manifest.csv')
        with open(mf, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader(); w.writerows(manifest)
        print(f"Wrote {len(manifest)} swings + manifest -> {mf}")


if __name__ == '__main__':
    main()
