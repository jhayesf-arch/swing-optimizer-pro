"""usage: overlay.py <out_prefix> <calA> <calB> <sheet.png> [step]
Draw the reconstructed 3-D bat (projected) on both camera views: a contact sheet for checking.
calA / calB are the Cam0 / Cam1 calibrations; the .mov paths come from the tracker's output."""
import json
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from batcap import calib, project, video_rgb  # noqa: E402

pre, calA, calB, png = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
step = int(sys.argv[5]) if len(sys.argv) > 5 else 4
Z = dict(np.load(pre + '_bat.npz'))
import os
if os.path.exists(pre + '_clean.npz'):
    C = np.load(pre + '_clean.npz')
    for k_ in ('knob', 'tip', 'sweet', 'speed'): Z[k_] = C[k_]
J = json.load(open(pre + '_bat.json'))
tiles = []
for name, cal in (('A', calA), ('B', calB)):
    mov = J['cameras'][name]['mov']
    K, dist = calib(cal)
    rv, tv = (np.array(v).reshape(3, 1) for v in J['cams'][name])
    cam = (rv, tv, K, dist)
    off = J['cameras'][name]['offset']
    rgb = video_rgb(mov)
    row = []
    for k in range(0, len(Z['idx']), step):
        img = np.ascontiguousarray(rgb[Z['idx'][k] + off][..., ::-1])
        kn, tp, sw = project(np.array([Z['knob'][k], Z['tip'][k], Z['sweet'][k]]), cam)
        cv2.line(img, tuple(int(v) for v in kn), tuple(int(v) for v in tp), (0, 0, 255), 4)
        cv2.circle(img, tuple(int(v) for v in sw), 9, (0, 255, 255), -1)
        cv2.putText(img, f"{Z['speed'][k]:.0f} mph", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)
        hx, hy = Z['h2' + name][k]
        x0, y0 = int(np.clip(hx - 360, 0, 720 - 720)), int(np.clip(hy - 450, 0, 1280 - 900))
        row.append(cv2.resize(img[y0:y0 + 900, :], (240, 300)))
    tiles.append(np.hstack(row))
    del rgb
cv2.imwrite(png, np.vstack(tiles))
print(png, len(tiles[0][0]) // 240, 'frames per row')
