"""Clean a bat track saved by batcap.py and recompute bat speed.

- hands: Hampel-despiked, then smoothed (wrist keypoints jump in the fast part of the swing)
- bat direction per frame from the saved 2-D angles in both cameras (plane intersection)
- frames dropped when the camera planes nearly coincide (sep < 0.3) or when the direction jumps
  more than MAX_STEP_DEG from the local trend (faster than a real bat can rotate)
- dropped frames filled from neighbours; peak only counted where real tracked frames support it

usage: postbat.py <out_prefix> <cam0_calib.pickle> <cam1_calib.pickle> <trial.trc>
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from batcap import (MPH, SWEET_FROM_HANDS, TIP_FROM_HANDS, KNOB_FROM_HANDS,  # noqa: E402
                    calib, plane_normal, project, read_trc)

MAX_STEP_DEG = 35.0
MIN_SEP = 0.30


def despike(x, tol=0.08):
    """Replace single-frame jumps: samples far from a local cubic fit (Savitzky-Golay residual)."""
    from scipy.signal import savgol_filter
    trend = savgol_filter(x, 9, 3)
    r = x - trend
    bad = np.abs(r) > tol          # metres: a wrist keypoint >4 cm off a smooth local curve is a glitch
    if bad.any() and (~bad).sum() > 3:
        i = np.arange(len(x))
        x = x.copy(); x[bad] = np.interp(i[bad], i[~bad], x[~bad])
    return x


def smooth(x, n=5):
    k = np.ones(n) / n
    if x.ndim == 1:
        return np.convolve(np.pad(x, n // 2, mode='edge'), k, 'valid')
    return np.stack([smooth(x[:, j], n) for j in range(x.shape[1])], 1)


def cam_center_ray(cam, px):
    rvec, tvec, K, dist = cam
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).ravel()
    und = cv2.undistortPoints(np.asarray(px, np.float64).reshape(1, 1, 2), K, dist)[0, 0]
    d = R.T @ np.array([und[0], und[1], 1.0])
    return C, d / np.linalg.norm(d)


def tip_candidates(cam, H, h2, ang, end_px, L=TIP_FROM_HANDS):
    """3-D tips consistent with this camera's barrel end and a rigid bat of length L from the hands."""
    if not np.isfinite(end_px):
        return []
    tip2 = h2 + end_px * np.array([np.cos(ang), np.sin(ang)])
    C, d = cam_center_ray(cam, tip2)
    w = C - H
    b = np.dot(d, w); c = np.dot(w, w) - L * L
    disc = b * b - c
    if disc < 0:                                  # streak looks longer than the bat can be: closest point
        return [C - b * d]
    r = np.sqrt(disc)
    return [C + (-b - r) * d, C + (-b + r) * d]


def view_err(cam, H, tip, ang, end_px=None):
    """Image-angle disagreement (degrees) between a candidate 3-D bat and one camera's 2-D bat.
    (A length term from the streak end was tried; blur makes that measurement too noisy.)"""
    p0 = project(H, cam)[0]; p1 = project(tip, cam)[0]
    v = p1 - p0
    if np.linalg.norm(v) < 3:
        return 0.0
    a = np.arctan2(v[1], v[0])
    return float(np.degrees(abs((a - ang + np.pi) % (2 * np.pi) - np.pi)))


def clean(prefix, calA, calB, trc):
    Z = np.load(prefix + '_bat.npz')
    J = json.load(open(prefix + '_bat.json'))
    t, P = read_trc(trc)
    cams = {}
    for name, cal in (('A', calA), ('B', calB)):
        K, dist = calib(cal)
        rv, tv = (np.array(v).reshape(3, 1) for v in J['cams'][name])
        cams[name] = (rv, tv, K, dist)
    idx = Z['idx']
    H = (P['LWrist'] + P['RWrist']) / 2
    H = np.stack([despike(H[:, j]) for j in range(3)], 1)
    H = smooth(H, 3)[idx]
    tt = t[idx]
    n = len(idx)
    ang = {'A': Z['angA'], 'B': Z['angB']}
    endp = {'A': Z['endA'], 'B': Z['endB']}
    u = np.full((n, 3), np.nan); err = np.full(n, np.nan)
    for k in range(n):
        h2 = {c: project(H[k], cams[c])[0] for c in 'AB'}
        cands = []
        for c, o in (('A', 'B'), ('B', 'A')):
            for tip in tip_candidates(cams[c], H[k], h2[c], ang[c][k], endp[c][k]):
                cands.append((view_err(cams[o], H[k], tip, ang[o][k], endp[o][k])
                              + view_err(cams[c], H[k], tip, ang[c][k], endp[c][k]), tip))
        # plane intersection too, when the two views are well separated
        nA = plane_normal(cams['A'], H[k], h2['A'], ang['A'][k]); nB = plane_normal(cams['B'], H[k], h2['B'], ang['B'][k])
        d = np.cross(nA, nB)
        if np.linalg.norm(d) > 0.5:
            d /= np.linalg.norm(d)
            for sgn in (1, -1):
                tip = H[k] + sgn * TIP_FROM_HANDS * d
                cands.append((view_err(cams['A'], H[k], tip, ang['A'][k], endp['A'][k])
                              + view_err(cams['B'], H[k], tip, ang['B'][k], endp['B'][k]), tip))
        if not cands:
            continue
        e, tip = min(cands, key=lambda x: x[0])
        if e > 30:
            continue
        v = tip - H[k]
        u[k] = v / np.linalg.norm(v); err[k] = e
    good = np.all(np.isfinite(u), 1)
    for _ in range(2):
        gi = np.where(good)[0]
        for k in gi:
            nb = [j for j in gi if j != k and abs(j - k) <= 3]
            if len(nb) < 2:
                continue
            ref = np.median(u[nb], 0); ref /= np.linalg.norm(ref)
            a = np.degrees(np.arccos(np.clip(np.dot(u[k], ref), -1, 1)))
            if a > MAX_STEP_DEG * max(1, min(abs(j - k) for j in nb)):
                good[k] = False
    if good.sum() < 5:
        return None
    for j in range(3):
        u[:, j] = np.interp(np.arange(n), np.where(good)[0], u[good, j])
    u = smooth(u, 3)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    sweet = H + SWEET_FROM_HANDS * u
    v = smooth(np.linalg.norm(np.gradient(sweet, tt, axis=0), axis=1) * MPH, 3)
    support = np.convolve(good.astype(float), np.ones(5) / 5, 'same')
    vv = np.where(support >= 0.6, v, np.nan)
    pk = int(np.nanargmax(vv)) if np.isfinite(vv).any() else None
    hv = np.linalg.norm(np.gradient(H, tt, axis=0), axis=1) * MPH
    # how much of the fast part of the swing was genuinely tracked (not filled): judged around
    # the frame of most motion in the raw track and around the reported peak
    raw_pk = int(np.argmax(Z['speed']))
    def frac(c):
        return float(good[max(0, c - 8):c + 9].mean())
    fast_good = min(frac(raw_pk), frac(pk)) if pk is not None else 0.0
    res = {'fast_window_good': fast_good,'peak_bat_speed_mph': float(vv[pk]) if pk is not None else None,
           'peak_frame': int(idx[pk]) if pk is not None else None,
           'good_fraction': float(good.mean()),
           'median_view_err_deg': float(np.nanmedian(err)),
           'peak_hand_speed_mph': float(hv.max())}
    np.savez(prefix + '_clean.npz', idx=idx, t=tt, u=u, good=good, sweet=sweet, speed=v,
             knob=H + KNOB_FROM_HANDS * u, tip=H + TIP_FROM_HANDS * u, hands=H, err=err)
    return res


if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.exit('usage: postbat.py <out_prefix> <cam0_calib.pickle> <cam1_calib.pickle> <trial.trc>')
    pre, calA, calB, trc = sys.argv[1:5]
    res = clean(pre, calA, calB, trc)
    print(json.dumps(res, indent=1))
