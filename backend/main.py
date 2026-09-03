import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from analyzer import RefinedHittingOptimizer
from subject_profile import aggregate_swings

try:
    from opensim_id import run_inverse_dynamics, summarize_id_results
    HAS_OPENSIM_ID = True
except Exception:
    HAS_OPENSIM_ID = False

app = FastAPI(title="Hitting Optimizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../docs")
TMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

DEFAULT_MODEL = os.path.expanduser(
    "~/Desktop/OpenCapData_94fba876-8deb-4074-afe5-8d7872fec1ae"
    "/OpenSimData/Model/LaiUhlrich2022_scaled.osim"
)


def _run_id(file_path, model_path, bat_mass_kg, bat_length_m, diagnosis):
    """Run OpenSim ID and attach results to diagnosis dict. Silent on failure."""
    if HAS_OPENSIM_ID and os.path.exists(model_path):
        try:
            id_result = run_inverse_dynamics(
                file_path, model_path=model_path,
                bat_mass_kg=bat_mass_kg, bat_length_m=bat_length_m
            )
            diagnosis['opensim_id'] = summarize_id_results(id_result)
        except Exception:
            pass


# Key markers for stick figure (subset of 63 TRC markers)
_SKEL_MARKERS = [
    'Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow', 'RWrist', 'LWrist',
    'midHip', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle',
    'RHeel', 'LHeel',
]


def _kinematic_sequence(mot_df, rotation=None, max_points: int = 72) -> dict:
    """Proximal-to-distal kinematic sequence for the chart: segment angular
    velocity (deg/s) vs time.

    The series come from the analyzer's own segment velocities so the chart and
    the Sequence Quality metric are drawn from identical signals. An independent
    proxy for the chart drifts out of agreement with the metric — the previous
    one summed raw joint angles across incompatible axes and rendered the
    sequence reversed (hands peaking before the pelvis on every swing).
    """
    import numpy as np

    if rotation is None or not rotation.get('segment_omega'):
        return {}
    if 'time' not in mot_df.columns:
        return {}
    t = mot_df['time'].values.astype(float)
    if len(t) < 8:
        return {}
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return {}

    # rad/s -> deg/s magnitude, trimmed to the shortest available series
    raw = {}
    for name, arr in rotation['segment_omega'].items():
        a = np.abs(np.asarray(arr, dtype=float)) * 180.0 / np.pi
        if a.size and np.nanmax(a) > 1e-6:
            raw[name] = a
    if 'Pelvis' not in raw:
        return {}
    n = min(len(t), min(len(v) for v in raw.values()))
    t = t[:n]
    raw = {k: v[:n] for k, v in raw.items()}

    # Window the chart to the swing itself, anchored on the pelvis peak.
    swing_start = int(rotation.get('swing_start_frame') or 0)
    pelw = raw['Pelvis']
    peak = int(np.argmax(pelw[swing_start:])) + swing_start
    start = max(0, min(swing_start, peak - int(0.05 / dt)))
    end = min(n - 1, peak + int(0.20 / dt))
    if end - start < 4:
        start, end = 0, n - 1
    idx = np.arange(start, end + 1)
    idx = idx[::max(1, len(idx) // max_points)]
    t0 = t[start]

    time_ms = [round(float((t[i] - t0) * 1000.0), 1) for i in idx]

    # Mark the peaks the sequence METRIC found (sub-frame interpolated, bounded
    # around the pelvis peak). Falling back to a local argmax here is what made
    # the chart disagree with the Sequence Quality shown beside it.
    metric_peaks = rotation.get('segment_peak_frames') or {}

    out_series, peaks = {}, {}
    for k, v in raw.items():
        out_series[k] = [int(round(float(v[i]))) for i in idx]
        pf = metric_peaks.get(k)
        if pf is not None and 0 <= float(pf) < n:
            frac = float(pf)
            pj = int(round(frac))
            # interpolate the timestamp so sub-frame precision isn't thrown away
            t_peak = float(np.interp(frac, np.arange(n), t))
        else:
            pj = int(np.argmax(v[start:end + 1])) + start
            t_peak = float(t[pj])
        peaks[k] = {'t_ms': round((t_peak - t0) * 1000.0, 1),
                    'value': int(round(float(v[min(pj, n - 1)])))}

    contact = peaks.get('Hands/Bat') or peaks.get('Lead Arm') or peaks['Pelvis']
    return {'time_ms': time_ms, 'series': out_series, 'peaks': peaks,
            'contact_ms': contact['t_ms'], 'units': 'deg/s'}


def _extract_skeleton_frames(trc_df, mot_df, max_frames: int = 60) -> dict:
    """
    Extract downsampled marker positions during the swing window for visualization.
    Returns a dict with marker names, frames array, and swing event indices.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt, savgol_filter

    # Find swing window from pelvis_rotation in mot
    if 'pelvis_rotation' not in mot_df.columns:
        return {}

    dt = mot_df['time'].diff().mean()
    fs = 1.0 / dt
    pelvis_rad = np.unwrap(np.deg2rad(mot_df['pelvis_rotation'].values))
    nyq = 0.5 * fs
    b, a = butter(4, min(15.0 / nyq, 0.99), btype='low')
    pelvis_f = filtfilt(b, a, pelvis_rad)
    w = max(11, int(0.10 * fs) | 1)
    pelvis_omega = savgol_filter(pelvis_f, w, 3, deriv=1, delta=dt)

    peak = int(np.argmax(np.abs(pelvis_omega)))
    swing_start = 0
    peak_sign = np.sign(pelvis_omega[peak])
    for i in range(peak - 1, -1, -1):
        if np.sign(pelvis_omega[i]) != peak_sign and abs(pelvis_omega[i]) * 180 / np.pi > 20:
            swing_start = i
            break

    # Map mot frame indices to trc frame indices (both at 60Hz, same time base)
    mot_times = mot_df['time'].values
    trc_times = trc_df['Time'].values if 'Time' in trc_df.columns else trc_df['time'].values

    t_start = mot_times[swing_start]
    t_end   = mot_times[min(peak + int(0.3 / dt), len(mot_times) - 1)]  # include 300ms follow-through

    trc_mask = (trc_times >= t_start) & (trc_times <= t_end)
    trc_sub  = trc_df[trc_mask].reset_index(drop=True)

    if len(trc_sub) == 0:
        return {}

    # Downsample to max_frames — but don't over-downsample short windows
    step = max(1, len(trc_sub) // max_frames)
    trc_sub = trc_sub.iloc[::step].reset_index(drop=True)

    # Extract available markers
    frames = []
    available = []
    for m in _SKEL_MARKERS:
        cols = [f'{m}_X', f'{m}_Y', f'{m}_Z']
        if all(c in trc_sub.columns for c in cols):
            available.append(m)

    # OpenCap marker sets carry RHip/LHip but no midHip, and the spine bone is
    # drawn Neck -> midHip. Without it the figure renders severed: head, shoulders
    # and arms floating free of the pelvis and legs. Synthesise the pelvis centre.
    synth_midhip = 'midHip' not in available and {'RHip', 'LHip'} <= set(available)

    for _, row in trc_sub.iterrows():
        frame = {}
        for m in available:
            frame[m] = [round(float(row[f'{m}_X']), 4),
                        round(float(row[f'{m}_Y']), 4),
                        round(float(row[f'{m}_Z']), 4)]
        if synth_midhip:
            r, l = frame['RHip'], frame['LHip']
            frame['midHip'] = [round((r[i] + l[i]) / 2.0, 4) for i in range(3)]
        frames.append(frame)

    if synth_midhip:
        available = available + ['midHip']

    # Contact frame index within the downsampled frames
    t_contact = mot_times[peak]
    contact_idx = int(np.argmin(np.abs(trc_sub['Time'].values - t_contact))) if 'Time' in trc_sub.columns else len(frames) // 2

    return {
        'markers': available,
        'frames':  frames,
        'contact_frame': contact_idx,
        'fps': round(1.0 / (step * dt), 1),
    }


def _skeleton_from_mot(mot_df, body_height_m: float = 1.83, max_frames: int = 60) -> dict:
    """
    Approximate 3D skeleton from .mot joint angles via forward kinematics.
    Used as fallback when no .trc marker file is available.
    Coordinate system: X=right, Y=up, Z=forward (anterior).
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    if 'pelvis_rotation' not in mot_df.columns:
        return {}

    h = body_height_m
    # Segment lengths scaled to body height (Winter 2009 anthropometric ratios)
    L = {
        'trunk':    h * 0.288,   # pelvis origin to neck
        'upper_arm': h * 0.186,
        'forearm':  h * 0.146,
        'thigh':    h * 0.245,
        'shank':    h * 0.246,
        'hip_half': h * 0.095,   # midHip to each hip joint
        'shoulder_half': h * 0.129,
    }

    dt = mot_df['time'].diff().mean()
    fs = 1.0 / dt
    nyq = 0.5 * fs
    b, a = butter(4, min(10.0 / nyq, 0.99), btype='low')

    def filt(col):
        if col in mot_df.columns:
            return filtfilt(b, a, np.deg2rad(mot_df[col].values))
        return np.zeros(len(mot_df))

    pelvis_rot   = filt('pelvis_rotation')   # axial (Y)
    pelvis_tilt  = filt('pelvis_tilt')       # sagittal (X)
    pelvis_list  = filt('pelvis_list')       # frontal (Z)
    lumbar_rot   = filt('lumbar_rotation')
    lumbar_bend  = filt('lumbar_bending') if 'lumbar_bending' in mot_df.columns else np.zeros(len(mot_df))

    hip_flex_r   = filt('hip_flexion_r')
    hip_flex_l   = filt('hip_flexion_l')
    knee_r       = filt('knee_angle_r')
    knee_l       = filt('knee_angle_l')
    arm_flex_r   = filt('arm_flex_r')
    arm_flex_l   = filt('arm_flex_l')
    elbow_r      = filt('elbow_flex_r')
    elbow_l      = filt('elbow_flex_l')

    # Pelvis translation (already in metres in .mot)
    px = mot_df['pelvis_tx'].values if 'pelvis_tx' in mot_df.columns else np.zeros(len(mot_df))
    py = mot_df['pelvis_ty'].values if 'pelvis_ty' in mot_df.columns else np.ones(len(mot_df)) * h * 0.53
    pz = mot_df['pelvis_tz'].values if 'pelvis_tz' in mot_df.columns else np.zeros(len(mot_df))

    # Find swing window (same logic as _extract_skeleton_frames)
    pelvis_omega = np.gradient(pelvis_rot, dt)
    peak = int(np.argmax(np.abs(pelvis_omega)))
    swing_start = 0
    peak_sign = np.sign(pelvis_omega[peak])
    for i in range(peak - 1, -1, -1):
        if np.sign(pelvis_omega[i]) != peak_sign and abs(pelvis_omega[i]) * 180 / np.pi > 20:
            swing_start = i
            break
    swing_end = min(peak + int(0.3 / dt), len(mot_df) - 1)

    indices = np.arange(swing_start, swing_end + 1)
    step = max(1, len(indices) // max_frames)
    indices = indices[::step]

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    frames = []
    for i in indices:
        R_pelvis = Ry(pelvis_rot[i]) @ Rx(pelvis_tilt[i]) @ Rz(pelvis_list[i])
        R_trunk  = Ry(pelvis_rot[i] + lumbar_rot[i]) @ Rx(pelvis_tilt[i] + lumbar_bend[i])

        mid_hip = np.array([px[i], py[i], pz[i]])
        neck    = mid_hip + R_trunk @ np.array([0, L['trunk'], 0])

        r_hip = mid_hip + R_pelvis @ np.array([ L['hip_half'], 0, 0])
        l_hip = mid_hip + R_pelvis @ np.array([-L['hip_half'], 0, 0])

        r_knee = r_hip + Rx(hip_flex_r[i]) @ np.array([0, -L['thigh'], 0])
        l_knee = l_hip + Rx(hip_flex_l[i]) @ np.array([0, -L['thigh'], 0])
        r_ankle = r_knee + Rx(hip_flex_r[i] - knee_r[i]) @ np.array([0, -L['shank'], 0])
        l_ankle = l_knee + Rx(hip_flex_l[i] - knee_l[i]) @ np.array([0, -L['shank'], 0])
        r_heel  = r_ankle + np.array([0, -0.02, -0.04])
        l_heel  = l_ankle + np.array([0, -0.02, -0.04])

        r_shoulder = neck + R_trunk @ np.array([ L['shoulder_half'], 0, 0])
        l_shoulder = neck + R_trunk @ np.array([-L['shoulder_half'], 0, 0])

        r_elbow = r_shoulder + (Ry(pelvis_rot[i]) @ Rx(arm_flex_r[i])) @ np.array([0, -L['upper_arm'], 0])
        l_elbow = l_shoulder + (Ry(pelvis_rot[i]) @ Rx(arm_flex_l[i])) @ np.array([0, -L['upper_arm'], 0])
        r_wrist = r_elbow + (Ry(pelvis_rot[i]) @ Rx(arm_flex_r[i] - elbow_r[i])) @ np.array([0, -L['forearm'], 0])
        l_wrist = l_elbow + (Ry(pelvis_rot[i]) @ Rx(arm_flex_l[i] - elbow_l[i])) @ np.array([0, -L['forearm'], 0])

        frame = {}
        for name, pt in [
            ('midHip', mid_hip), ('Neck', neck),
            ('RHip', r_hip), ('LHip', l_hip),
            ('RKnee', r_knee), ('LKnee', l_knee),
            ('RAnkle', r_ankle), ('LAnkle', l_ankle),
            ('RHeel', r_heel), ('LHeel', l_heel),
            ('RShoulder', r_shoulder), ('LShoulder', l_shoulder),
            ('RElbow', r_elbow), ('LElbow', l_elbow),
            ('RWrist', r_wrist), ('LWrist', l_wrist),
        ]:
            frame[name] = [round(float(pt[0]), 4), round(float(pt[1]), 4), round(float(pt[2]), 4)]
        frames.append(frame)

    t_contact = mot_df['time'].values[peak]
    contact_idx = min(range(len(indices)), key=lambda k: abs(mot_df['time'].values[indices[k]] - t_contact))

    return {
        'markers': list(frames[0].keys()) if frames else [],
        'frames': frames,
        'contact_frame': contact_idx,
        'fps': round(1.0 / (step * dt), 1),
    }


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.get("/api/health")
def health():
    return JSONResponse(content={"status": "ok"})

@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    trc_file: UploadFile = File(None),
    height_m: float = Form(1.83),
    weight_kg: float = Form(82.0),
    skill_level: str = Form('high_school'),
    bat_mass_kg: float = Form(0.88),
    bat_length_m: float = Form(0.864),
    handedness: str = Form(None),
):
    if not file.filename.endswith('.mot'):
        return JSONResponse(status_code=400, content={"success": False, "error": "File must be a .mot file"})

    file_path = os.path.join(TMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    trc_path = None
    if trc_file and trc_file.filename and trc_file.filename.endswith('.trc'):
        trc_path = os.path.join(TMP_DIR, trc_file.filename)
        with open(trc_path, "wb") as f:
            f.write(await trc_file.read())

    try:
        optimizer = RefinedHittingOptimizer(
            body_mass_kg=weight_kg, body_height_m=height_m,
            skill_level=skill_level,
            bat_mass_kg=bat_mass_kg, bat_length_m=bat_length_m,
            handedness=handedness,
        )
        kinematics = optimizer.load_mot_file(file_path)
        if kinematics is None or len(kinematics) == 0:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid or empty .mot file"})

        trc_data = optimizer.load_trc_file(trc_path) if trc_path else None
        diagnosis = optimizer.comprehensive_diagnosis(kinematics, file.filename, trc_data=trc_data)
        if trc_data is not None:
            try:
                diagnosis['skeleton_frames'] = _extract_skeleton_frames(trc_data, kinematics)
            except Exception:
                pass
        if not diagnosis.get('skeleton_frames'):
            try:
                diagnosis['skeleton_frames'] = _skeleton_from_mot(kinematics, body_height_m=height_m)
            except Exception:
                pass
        try:
            diagnosis['kinematic_sequence'] = _kinematic_sequence(kinematics, diagnosis.get('_rotation'))
        except Exception:
            pass
        _run_id(file_path, DEFAULT_MODEL, bat_mass_kg, bat_length_m, diagnosis)
        diagnosis.pop('_rotation', None)   # numpy arrays — not JSON-serialisable
        return JSONResponse(content={"filename": file.filename, "success": True, "data": diagnosis})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if trc_path and os.path.exists(trc_path):
            os.remove(trc_path)

@app.post("/api/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    trc_files: List[UploadFile] = File(None),
    height_m: float = Form(1.83),
    weight_kg: float = Form(82.0),
    skill_level: str = Form('high_school'),
    bat_mass_kg: float = Form(0.88),
    bat_length_m: float = Form(0.864),
    handedness: str = Form(None),
):
    """Analyze several swings for one athlete and return every per-swing report
    plus an averaged view (outliers excluded) with swing-to-swing consistency.

    One swing is a noisy sample; the average is what an athlete should be judged
    on, and the spread is itself a coaching signal.
    """
    mots = [f for f in files if f.filename and f.filename.endswith('.mot')]
    if not mots:
        return JSONResponse(status_code=400, content={"success": False, "error": "No .mot files provided"})

    # Pair markers to kinematics by filename stem (OpenCap splits them across folders).
    trc_by_stem = {}
    written = []
    try:
        for tf in (trc_files or []):
            if not tf or not tf.filename or not tf.filename.endswith('.trc'):
                continue
            path = os.path.join(TMP_DIR, os.path.basename(tf.filename))
            with open(path, "wb") as fh:
                fh.write(await tf.read())
            written.append(path)
            trc_by_stem[os.path.splitext(os.path.basename(tf.filename))[0].lower()] = path

        swings, errors = [], []
        for uf in mots:
            name = os.path.basename(uf.filename)
            stem = os.path.splitext(name)[0]
            mot_path = os.path.join(TMP_DIR, name)
            with open(mot_path, "wb") as fh:
                fh.write(await uf.read())
            written.append(mot_path)
            try:
                opt = RefinedHittingOptimizer(
                    body_mass_kg=weight_kg, body_height_m=height_m,
                    skill_level=skill_level, bat_mass_kg=bat_mass_kg, bat_length_m=bat_length_m,
                    handedness=handedness,
                )
                kin = opt.load_mot_file(mot_path)
                if kin is None or len(kin) == 0:
                    errors.append({"file": name, "error": "empty/invalid .mot"})
                    continue
                trc_path = trc_by_stem.get(stem.lower())
                trc_data = opt.load_trc_file(trc_path) if trc_path else None
                diag = opt.comprehensive_diagnosis(kin, name, trc_data=trc_data)
                rep = diag.get("phase_report", {})

                # Kinematic sequence is small and worth having per swing; skeleton
                # frames are large, so only the first swing carries them.
                extras = {}
                try:
                    extras['kinematic_sequence'] = _kinematic_sequence(kin, diag.get('_rotation'))
                except Exception:
                    pass
                # Every swing carries its own skeleton (~20KB) — otherwise selecting
                # trial 3 in the Viewing menu showed trial 1's body, or nothing.
                try:
                    extras['skeleton_frames'] = (
                        _extract_skeleton_frames(trc_data, kin) if trc_data is not None
                        else _skeleton_from_mot(kin, body_height_m=height_m)
                    )
                except Exception:
                    pass
                diag.pop('_rotation', None)

                swings.append({
                    "index": len(swings) + 1,
                    "name": stem,
                    "has_markers": trc_data is not None,
                    "swing_score": rep.get("swing_score", 0.0),
                    "overall_percentile": rep.get("overall_percentile"),
                    "percentile_basis": rep.get("percentile_basis"),
                    "capture_quality": diag.get("capture_quality", {}),
                    "efficiency_score": diag.get("efficiency_score", 0),
                    "dimensions": rep.get("dimensions", {}),
                    "phases": rep.get("phases", {}),
                    "lead_leg_block": rep.get("lead_leg_block", {}),
                    "prescriptions": rep.get("prescriptions", []),
                    "metrics": diag.get("metrics", {}),
                    "findings": diag.get("findings", []),
                    "recommendations": diag.get("recommendations", []),
                    "grf_estimation": diag.get("grf_estimation", {}),
                    "data_quality": diag.get("data_quality", {}),
                    # Evidence tier + per-capture reliability, one entry per metric.
                    # Was silently dropped by the batch shaper, so uploads of two or
                    # more files rendered every tile without a badge — the caveat
                    # existed in the payload of a single-file upload only.
                    "metric_evidence": diag.get("metric_evidence", {}),
                    **extras,
                })
            except Exception as e:
                errors.append({"file": name, "error": str(e)})

        if not swings:
            return JSONResponse(status_code=400, content={
                "success": False, "error": "No swings could be analyzed", "errors": errors})

        agg = aggregate_swings(swings, skill_level, height_m, weight_kg, bat_mass_kg, bat_length_m)
        return JSONResponse(content={
            "success": True,
            "n_swings": len(swings),
            "skill_level": skill_level,
            "swings": swings,
            "errors": errors,
            **agg,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        for p in written:
            if os.path.exists(p):
                os.remove(p)


@app.post("/api/coach")
async def coach(request: Request):
    """Streaming conversational coach.

    The browser sends the already-computed report back as grounding; the API key
    lives only here, server-side. The model explains those numbers — it never
    computes biomechanics (see coach.py).
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Expected a JSON body"})

    from coach import stream_reply
    return StreamingResponse(
        stream_reply(payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/coach/status")
def coach_status():
    """Lets the UI hide the coach entirely when no key is configured."""
    from coach import COACH_MODEL
    return {
        "enabled": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")),
        "model": COACH_MODEL,
    }


@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))

@app.get("/app.js")
def serve_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"))

@app.get("/body.svg")
def serve_body_svg():
    # The Coaching Focus heatmap fetches this by relative path; without an
    # explicit route it 404s whenever the app is served from the API backend
    # (only the GitHub Pages build has docs/ as the web root).
    return FileResponse(os.path.join(FRONTEND_DIR, "body.svg"), media_type="image/svg+xml")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
