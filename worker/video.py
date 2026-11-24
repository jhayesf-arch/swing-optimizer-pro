"""Swing video: Camera 0 | Camera 1 | 3-D skeleton, frame-aligned, quarter speed.

usage: python -m worker.video <trial.trc> <cam0_sync.mp4> <cam1_sync.mp4> <out.mp4> <title>

OpenCap's *_sync.mp4 videos have one frame per .trc frame, so frame i of each
camera and skeleton frame i show the same instant. Every frame is shown, at 30 fps:
a 120 fps capture plays at 1/4 speed, a 240 fps capture at 1/8.
"""
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BONES = [('Neck', 'midHip'), ('Neck', 'RShoulder'), ('Neck', 'LShoulder'),
         ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
         ('midHip', 'RHip'), ('midHip', 'LHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
         ('LHip', 'LKnee'), ('LKnee', 'LAnkle'), ('RAnkle', 'RHeel'), ('RAnkle', 'RBigToe'),
         ('RHeel', 'RBigToe'), ('LAnkle', 'LHeel'), ('LAnkle', 'LBigToe'), ('LHeel', 'LBigToe')]
COLOR = {'R': '#d95926', 'L': '#2456d6', 'C': '#1d1c17'}


def read_trc(path):
    lines = open(path).read().splitlines()
    names = [n for n in lines[3].split('\t')[2:] if n]
    rows = [l.split('\t') for l in lines[5:] if l.strip()]
    data = np.array([[float(x) if x else np.nan for x in r[:2 + 3 * len(names)]] for r in rows])
    t = data[:, 1]
    P = {n: data[:, 2 + 3 * i: 5 + 3 * i] for i, n in enumerate(names)}
    return t, P


def side(a, b):
    s = a[0] if a[0] in 'RL' else (b[0] if b[0] in 'RL' else 'C')
    return COLOR[s] if not (a in ('Neck', 'midHip') and b in ('Neck', 'midHip')) else COLOR['C']


def capture_fps(trc):
    t, _ = read_trc(trc)
    return float(round(1.0 / np.median(np.diff(t))))


def render_skeleton(trc, frames_dir, title, n_frames):
    t, P = read_trc(trc)
    slow = capture_fps(trc) / 30.0
    n = min(len(t), n_frames)
    # OpenCap: Y up. Plot (X, -Z) horizontal, Y vertical: a rotation, not a mirror.
    pts = np.concatenate([P[m][:n] for a, b in BONES for m in (a, b)])
    ctr = np.nanmean(pts, axis=0)
    # Frame the body, not the whole trial's travel: half-height of a standing frame.
    body_h = float(np.nanmedian(P['Neck'][:n, 1] - np.minimum(P['LAnkle'][:n, 1], P['RAnkle'][:n, 1])))
    half = 0.5 * body_h + 0.25
    # Look at the hitter's front: the direction the feet point (heel -> toe).
    f = np.nanmean(np.concatenate([P['RBigToe'][:n] - P['RHeel'][:n], P['LBigToe'][:n] - P['LHeel'][:n]]), 0)
    azim = float(np.degrees(np.arctan2(-f[2], f[0])))   # plot y = -Z keeps the frame right-handed
    fig = plt.figure(figsize=(3.6, 6.4), dpi=100)
    ax = fig.add_axes([-0.18, -0.06, 1.36, 1.0], projection='3d')
    for i in range(n):
        ax.cla()
        for a, b in BONES:
            if a in P and b in P:
                pa, pb = P[a][i], P[b][i]
                ax.plot([pa[0], pb[0]], [-pa[2], -pb[2]], [pa[1], pb[1]], color=side(a, b), lw=4, solid_capstyle='round')
        for m in ('Neck', 'midHip', 'RWrist', 'LWrist'):
            if m in P:
                p = P[m][i]
                ax.scatter([p[0]], [-p[2]], [p[1]], s=30, color='#1d1c17')
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(-ctr[2] - half, -ctr[2] + half)
        ax.set_zlim(ctr[1] - half * 1.25, ctr[1] + half * 1.25)
        ax.set_box_aspect((1, 1, 1.25))
        ax.view_init(elev=10, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_axis_off()
        fig.texts.clear()
        fig.text(0.5, 0.965, title, ha='center', fontsize=10, weight='bold')
        fig.text(0.5, 0.935, f'3-D skeleton   t = {t[i] - t[0]:.3f} s   (1/{slow:g} speed)', ha='center', fontsize=8, color='#5f5d55')
        fig.text(0.5, 0.02, 'orange = right side   blue = left side', ha='center', fontsize=7, color='#5f5d55')
        fig.savefig(os.path.join(frames_dir, f'{i:05d}.png'), facecolor='white')
    plt.close(fig)
    return n


def main(trc, cam0, cam1, out, title):
    def nframes(v):
        r = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                            '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', v],
                           capture_output=True, text=True)
        return int(r.stdout.strip())
    n = min(nframes(cam0), nframes(cam1))
    with tempfile.TemporaryDirectory() as tmp:
        n = render_skeleton(trc, tmp, title, n)
        slow = capture_fps(trc) / 30.0      # stretch the camera clips to one frame per 1/30 s
        fc = (f'[0:v]setpts={slow:g}*PTS,scale=360:640,setsar=1[a];'
              f'[1:v]setpts={slow:g}*PTS,scale=360:640,setsar=1[b];'
              '[2:v]scale=360:640,setsar=1[c];[a][b][c]hstack=inputs=3[v]')
        cmd = ['ffmpeg', '-v', 'error', '-y', '-i', cam0, '-i', cam1,
               '-framerate', '30', '-i', os.path.join(tmp, '%05d.png'),
               '-filter_complex', fc, '-map', '[v]', '-r', '30', '-frames:v', str(n),
               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-movflags', '+faststart', out]
        subprocess.run(cmd, check=True)
    print(out, n, 'frames')


if __name__ == '__main__':
    main(*sys.argv[1:6])
