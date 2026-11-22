"""Bat tracking from the two OpenCap phone videos -> 3-D bat -> bat speed.

usage: batcap.py <trial.trc> <camA.mov> <camA_sync.mp4> <camA_keypoints.pkl> <camA_calib.pickle>
                             <camB.mov> <camB_sync.mp4> <camB_keypoints.pkl> <camB_calib.pickle>
                             <out_prefix> [lefty]

1. Frame alignment: find which raw .mov frame matches each OpenCap (.trc / _sync.mp4) frame.
2. Camera pose in the .trc frame: solvePnP of the 3-D .trc markers against OpenPose keypoints,
   using each phone's calibrated intrinsics.
3. Bat direction in each view: the bright barrel streak leaving the hands. Every frame, rays are
   cast from the projected hands; each angle is scored by how much the pixels along it differ
   from the clip's median background and how bat-coloured they are. A dynamic-programming pass
   picks the smoothest high-scoring angle track.
4. 3-D bat: each view's 2-D bat line plus its camera centre spans a plane; the planes cross along
   the bat. Anchored at the 3-D hands -> knob, sweet spot, tip. Bat speed = sweet-spot speed.
"""
import json
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from safepkl import load as safe_load  # noqa: E402

FS = 120.0
MPH = 2.23694
SWEET_FROM_HANDS = 0.61     # m: 33-in bat, sweet spot 15 cm from the tip, hands ~8 cm above the knob
TIP_FROM_HANDS = 0.76
KNOB_FROM_HANDS = -0.08
BAT_PX_MIN, BAT_PX_MAX = 40, 420

# .trc marker -> OpenPose BODY_25 index
B25 = {'Neck': 1, 'RShoulder': 2, 'RElbow': 3, 'RWrist': 4, 'LShoulder': 5, 'LElbow': 6, 'LWrist': 7,
       'midHip': 8, 'RHip': 9, 'RKnee': 10, 'RAnkle': 11, 'LHip': 12, 'LKnee': 13, 'LAnkle': 14,
       'LBigToe': 19, 'LHeel': 21, 'RBigToe': 22, 'RHeel': 24}


def read_trc(path):
    L = open(path).read().splitlines()
    names = [n for n in L[3].split('\t')[2:] if n]
    rows = np.array([[float(x) if x else np.nan for x in r.split('\t')[:2 + 3 * len(names)]]
                     for r in L[5:] if r.strip()])
    return rows[:, 1], {n: rows[:, 2 + 3 * i:5 + 3 * i] for i, n in enumerate(names)}


def video_gray(path, w, h):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'scale={w}:{h},format=gray',
                        '-f', 'rawvideo', '-'], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w).astype(np.float32)


def video_rgb(path, frames=None):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       capture_output=True, check=True)
    W, H = 720, 1280
    a = np.frombuffer(r.stdout, np.uint8).reshape(-1, H, W, 3)
    return a


def sync_offset(mov, sync):
    """mov frame index = trc/sync frame index + offset."""
    A = video_gray(mov, 72, 128)
    B = video_gray(sync, 72, 128)
    A = (A - A.mean((1, 2), keepdims=True)) / (A.std((1, 2), keepdims=True) + 1e-6)
    B = (B - B.mean((1, 2), keepdims=True)) / (B.std((1, 2), keepdims=True) + 1e-6)
    best = None
    for off in range(0, len(A) - len(B) + 1):
        e = np.mean((A[off:off + len(B)] - B) ** 2)
        if best is None or e < best[0]:
            best = (e, off)
    return best[1], float(best[0])


def hitter_keypoints(pkl, ref_xy=None):
    """Per frame, the BODY_25 keypoints (25x3) of the hitter: the person nearest the previous pick
    (or the largest person in the first frame)."""
    raw = safe_load(pkl)
    out, prev = [], ref_xy
    for people in raw:
        cands = [np.asarray(p['pose_keypoints_2d'], float).reshape(25, 3) for p in people]
        if not cands:
            out.append(np.full((25, 3), np.nan)); continue
        if prev is None:
            def size(k):
                v = k[k[:, 2] > 0.3, :2]
                return np.ptp(v[:, 1]) if len(v) else 0
            k = max(cands, key=size)
        else:
            def dist(k):
                v = k[:, 2] > 0.3
                return np.nanmean(np.linalg.norm(k[v, :2] - prev[v], axis=1)) if v.any() else 1e9
            k = min(cands, key=dist)
        out.append(k)
        good = k[:, 2] > 0.3
        prev = np.where(good[:, None], k[:, :2], prev if prev is not None else k[:, :2])
    return np.array(out)


def calib(path):
    d = safe_load(path)
    return np.asarray(d['intrinsicMat'], float), np.asarray(d['distortion'], float).reshape(-1)


def fit_pose(P, kp, off, K, dist):
    """solvePnP: trc metres -> this camera. Returns rvec, tvec, median reprojection error (px)."""
    X, x = [], []
    n = len(next(iter(P.values())))
    for i in range(0, n, 2):
        j = i + off
        if j < 0 or j >= len(kp):
            continue
        for m, b in B25.items():
            if m in P and kp[j, b, 2] > 0.6 and np.all(np.isfinite(P[m][i])):
                X.append(P[m][i]); x.append(kp[j, b, :2])
    X = np.array(X, np.float64); x = np.array(x, np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X, x, K, dist, reprojectionError=12.0, iterationsCount=500,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    rvec, tvec = cv2.solvePnPRefineLM(X[inl[:, 0]], x[inl[:, 0]], K, dist, rvec, tvec)
    proj, _ = cv2.projectPoints(X, rvec, tvec, K, dist)
    err = np.linalg.norm(proj[:, 0] - x, axis=1)
    return rvec, tvec, float(np.median(err)), len(X)


def project(pts, cam):
    rvec, tvec, K, dist = cam
    p, _ = cv2.projectPoints(np.asarray(pts, np.float64).reshape(-1, 3), rvec, tvec, K, dist)
    return p[:, 0]


def tan_score(rgb):
    """How bat-barrel-like (light, warm wood) each pixel is, 0..1."""
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    v = f.max(-1)
    warm = np.clip((r - b) / (v + 1e-3), 0, 1)
    return np.clip((v - 0.45) / 0.4, 0, 1) * np.clip(warm / 0.25, 0, 1)


def bat_angles(frames_rgb, bg, hands2d, prev_rgb=None, next_rgb=None):
    """Angle track (radians, image coords) of the bat from the hands, per frame, plus a confidence."""
    n = len(frames_rgb)
    angs = np.radians(np.arange(0, 360, 2.0))
    radii = np.arange(BAT_PX_MIN, BAT_PX_MAX, 6.0)
    S = np.zeros((n, len(angs)))
    E = np.full((n, len(angs)), np.nan)        # image distance from the hands where the barrel ends
    for i in range(n):
        img = frames_rgb[i]
        f = img.astype(np.float32)
        before = (frames_rgb[i - 1] if i > 0 else prev_rgb).astype(np.float32)
        after = (frames_rgb[i + 1] if i < n - 1 else next_rgb).astype(np.float32)
        # moving in both directions: the bat streak, without the trailing ghost
        dmin = np.minimum(np.linalg.norm(f - before, axis=-1), np.linalg.norm(f - after, axis=-1)) / 255.0
        tan = tan_score(img)
        bgd = np.linalg.norm(f - bg, axis=-1) / 255.0
        score = np.maximum(np.clip(dmin / 0.12, 0, 1) * (0.4 + 0.6 * tan),
                           0.6 * np.clip(bgd / 0.25, 0, 1) * tan)
        hx, hy = hands2d[i]
        if not np.isfinite(hx):
            continue
        xs = hx + np.outer(np.cos(angs), radii)
        ys = hy + np.outer(np.sin(angs), radii)
        inside = (xs >= 0) & (xs < img.shape[1] - 1) & (ys >= 0) & (ys < img.shape[0] - 1)
        vals = np.where(inside, score[np.clip(ys, 0, img.shape[0] - 1).astype(int),
                                      np.clip(xs, 0, img.shape[1] - 1).astype(int)], 0)
        # a bat fills the ray continuously from near the hands: reward the longest filled run
        filled = vals > 0.35
        run = np.zeros(len(angs))
        for a in range(len(angs)):
            r, gap, last = 0, 0, -1
            for q, v in enumerate(filled[a]):
                if v:
                    r += 1; gap = 0; last = q
                else:
                    gap += 1
                    if r > 2 and gap > 2:
                        break
            run[a] = r
            if last >= 0:
                E[i, a] = radii[last]
        S[i] = vals.mean(1) + 0.02 * run
    # Viterbi: smooth angle track, penalise jumps
    jump = np.abs(((angs[:, None] - angs[None, :]) + np.pi) % (2 * np.pi) - np.pi)
    trans = -1.0 * (jump / np.radians(45)) ** 2
    acc = S[0].copy(); back = np.zeros((n, len(angs)), int)
    for i in range(1, n):
        m = acc[:, None] + trans
        back[i] = np.argmax(m, 0)
        acc = m[back[i], np.arange(len(angs))] + S[i]
    path = np.zeros(n, int); path[-1] = int(np.argmax(acc))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    conf = S[np.arange(n), path] / (np.median(S, 1) + 1e-6)
    return angs[path], conf, E[np.arange(n), path]


def plane_normal(cam, hand3d, hand2d, ang):
    """World-frame normal of the plane through the camera centre containing the image bat line."""
    rvec, tvec, K, dist = cam
    R, _ = cv2.Rodrigues(rvec)
    p2 = np.array([hand2d, hand2d + 100 * np.array([np.cos(ang), np.sin(ang)])], np.float64)
    und = cv2.undistortPoints(p2.reshape(-1, 1, 2), K, dist)[:, 0]
    r1 = np.array([und[0, 0], und[0, 1], 1.0]); r2 = np.array([und[1, 0], und[1, 1], 1.0])
    n_cam = np.cross(r1, r2)
    n = R.T @ n_cam
    return n / np.linalg.norm(n)


def main(trc, movA, syncA, kpA, calA, movB, syncB, kpB, calB, out, lefty=False):
    t, P = read_trc(trc)
    n = len(t)
    hands3d = (P['LWrist'] + P['RWrist']) / 2
    cams, info = {}, {}
    # OpenCap does not always file each phone's raw video under the same camera slot as its
    # calibration / synced video. Pair them by evidence: each raw .mov is aligned against both
    # synced videos, paired with the keypoint file of the same length, and each calibration is
    # assigned to the .mov that gives the tightest camera fit.
    movs = [movA, movB]
    kps_raw = [hitter_keypoints(kpA), hitter_keypoints(kpB)]
    cand = []
    for mi, mov in enumerate(movs):
        offs = [sync_offset(mov, s) for s in (syncA, syncB)]
        off, e = min(offs, key=lambda x: x[1])
        nfr = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                                  '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', mov],
                                 capture_output=True, text=True).stdout.strip().split(',')[0])
        kp_opts = [k for k in kps_raw if len(k) == nfr] or kps_raw
        cand.append((mov, off, e, kp_opts))
    fits = {}
    for ci, cal in enumerate((calA, calB)):
        K, dist = calib(cal)
        for mi, (mov, off, e, kp_opts) in enumerate(cand):
            best = None
            for kps in kp_opts:
                try:
                    r = fit_pose(P, kps, off, K, dist)
                except Exception:
                    continue
                if best is None or r[2] < best[2]:
                    best = r
            if best:
                fits[(ci, mi)] = (best, K, dist)
    straight = fits.get((0, 0), (None,))[0], fits.get((1, 1), (None,))[0]
    swapped = fits.get((0, 1), (None,))[0], fits.get((1, 0), (None,))[0]
    score = lambda pr: (pr[0][2] + pr[1][2]) if all(pr) else 1e9
    pairing = [(0, 0), (1, 1)] if score(straight) <= score(swapped) else [(0, 1), (1, 0)]
    for name, (ci, mi) in zip(('A', 'B'), pairing):
        (rvec, tvec, err, npts), K, dist = fits[(ci, mi)]
        mov, off, e, _ = cand[mi]
        cams[name] = (rvec, tvec, K, dist)
        info[name] = {'offset': off, 'sync_mse': round(e, 3), 'reproj_px_median': round(err, 1), 'n_pts': npts,
                      'mov': mov, 'swapped': pairing[0] != (0, 0)}
        print(name, info[name], flush=True)
    movA, movB = cand[pairing[0][1]][0], cand[pairing[1][1]][0]

    # swing window: centred on the most motion in camera A's video (the bat coming through),
    # not on peak wrist speed, which a single bad wrist keypoint can hijack
    gA = video_gray(movA, 90, 160)
    energy = np.abs(np.diff(gA, axis=0)).mean((1, 2))
    energy = np.convolve(energy, np.ones(5) / 5, 'same')
    offA = info['A']['offset']
    lo, hi = offA, min(len(energy), offA + n)
    pk = int(np.argmax(energy[lo:hi]))            # trc frame index
    i0, i1 = max(0, pk - int(0.45 * FS)), min(n, pk + int(0.20 * FS))
    idx = np.arange(i0, i1)

    angles, confs, h2ds, ends = {}, {}, {}, {}
    for name, mov in (('A', movA), ('B', movB)):
        off = info[name]['offset']
        allrgb = video_rgb(mov)
        bg = np.median(allrgb[::6].astype(np.float32), 0)
        fr = allrgb[idx + off]
        h2 = project(hands3d[idx], cams[name])
        a, c, e = bat_angles(fr, bg, h2, allrgb[max(0, idx[0] + off - 1)], allrgb[min(len(allrgb) - 1, idx[-1] + off + 1)])
        angles[name], confs[name], h2ds[name], ends[name] = a, c, h2, e
        del allrgb

    u = np.full((len(idx), 3), np.nan); sep = np.zeros(len(idx))
    for k in range(len(idx)):
        nA = plane_normal(cams['A'], hands3d[idx[k]], h2ds['A'][k], angles['A'][k])
        nB = plane_normal(cams['B'], hands3d[idx[k]], h2ds['B'][k], angles['B'][k])
        d = np.cross(nA, nB)
        sep[k] = np.linalg.norm(d)                      # sin(angle between planes): ~0 = degenerate
        if sep[k] < 0.15:
            continue
        d /= sep[k]
        # sign: the 3-D bat must point the way the 2-D bat points in camera A
        p0 = project(hands3d[idx[k]], cams['A'])[0]
        p1 = project(hands3d[idx[k]] + 0.5 * d, cams['A'])[0]
        if np.dot(p1 - p0, [np.cos(angles['A'][k]), np.sin(angles['A'][k])]) < 0:
            d = -d
        u[k] = d

    # fill short gaps, smooth the direction, then derive points and speed
    good = np.all(np.isfinite(u), 1)
    for j in range(3):
        u[:, j] = np.interp(np.arange(len(u)), np.where(good)[0], u[good, j])
    k5 = np.ones(5) / 5
    u = np.stack([np.convolve(np.pad(u[:, j], 2, mode='edge'), k5, 'valid') for j in range(3)], 1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    H = hands3d[idx]
    sweet = H + SWEET_FROM_HANDS * u
    tip = H + TIP_FROM_HANDS * u
    knob = H + KNOB_FROM_HANDS * u
    tt = t[idx]
    vs = np.linalg.norm(np.gradient(sweet, tt, axis=0), axis=1) * MPH
    vs = np.convolve(np.pad(vs, 2, mode='edge'), k5, 'valid')
    vh = np.linalg.norm(np.gradient(H, tt, axis=0), axis=1) * MPH
    res = {
        'cameras': info, 'frames': [int(i0), int(i1)],
        'tracked_fraction': float(good.mean()),
        'peak_bat_speed_mph': float(np.nanmax(vs)),
        'peak_hand_speed_mph': float(np.nanmax(vh)),
        'peak_frame': int(idx[int(np.nanargmax(vs))]),
        'median_conf': {k: float(np.median(v)) for k, v in confs.items()},
    }
    np.savez(out + '_bat.npz', t=tt, idx=idx, knob=knob, sweet=sweet, tip=tip, u=u, good=good,
             speed=vs, angA=angles['A'], angB=angles['B'], h2A=h2ds['A'], h2B=h2ds['B'], sep=sep,
             endA=ends['A'], endB=ends['B'], confA=confs['A'], confB=confs['B'], hands=hands3d[idx])
    json.dump({**res, 'cams': {k: [c[0].ravel().tolist(), c[1].ravel().tolist()] for k, c in cams.items()}},
              open(out + '_bat.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
    return res, cams


if __name__ == '__main__':
    a = sys.argv[1:]
    main(*a[:10], lefty=(len(a) > 10 and a[10] == 'lefty'))
