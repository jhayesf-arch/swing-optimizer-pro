import os
import glob
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from analyzer import RefinedHittingOptimizer

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

    for _, row in trc_sub.iterrows():
        frame = {}
        for m in available:
            frame[m] = [round(float(row[f'{m}_X']), 4),
                        round(float(row[f'{m}_Y']), 4),
                        round(float(row[f'{m}_Z']), 4)]
        frames.append(frame)

    # Contact frame index within the downsampled frames
    t_contact = mot_times[peak]
    contact_idx = int(np.argmin(np.abs(trc_sub['Time'].values - t_contact))) if 'Time' in trc_sub.columns else len(frames) // 2

    return {
        'markers': available,
        'frames':  frames,
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
    height_m: float = Form(1.83),
    weight_kg: float = Form(82.0),
    skill_level: str = Form('high_school'),
    bat_mass_kg: float = Form(0.88),
    bat_length_m: float = Form(0.864),
):
    if not file.filename.endswith('.mot'):
        return JSONResponse(status_code=400, content={"success": False, "error": "File must be a .mot file"})

    file_path = os.path.join(TMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        optimizer = RefinedHittingOptimizer(
            body_mass_kg=weight_kg, body_height_m=height_m,
            skill_level=skill_level,
            bat_mass_kg=bat_mass_kg, bat_length_m=bat_length_m
        )
        kinematics = optimizer.load_mot_file(file_path)
        if kinematics is None or len(kinematics) == 0:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid or empty .mot file"})

        diagnosis = optimizer.comprehensive_diagnosis(kinematics, file.filename)
        if trc_data is not None:
            try:
                diagnosis['skeleton_frames'] = _extract_skeleton_frames(trc_data, kinematics)
            except Exception:
                pass
        _run_id(file_path, DEFAULT_MODEL, bat_mass_kg, bat_length_m, diagnosis)
        return JSONResponse(content={"filename": file.filename, "success": True, "data": diagnosis})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/api/scan-downloads")
def scan_downloads():
    try:
        downloads_path = os.path.expanduser("~/Downloads")
        mot_files = glob.glob(os.path.join(downloads_path, "**/*.mot"), recursive=True)
        return JSONResponse(content={"success": True, "files": [
            {"filename": os.path.basename(f), "filepath": f} for f in mot_files
        ]})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/api/analyze/local")
def analyze_local(payload: dict):
    if "filepath" not in payload:
        return JSONResponse(status_code=400, content={"success": False, "error": "Filepath required"})

    file_path = payload["filepath"]
    filename = payload.get("filename", os.path.basename(file_path))

    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"success": False, "error": "File not found"})

    try:
        height_m    = float(payload.get('height_m', 1.83))
        weight_kg   = float(payload.get('weight_kg', 82.0))
        skill_level = str(payload.get('skill_level', 'high_school'))
        bat_mass_kg = float(payload.get('bat_mass_kg', 0.88))
        bat_length_m = float(payload.get('bat_length_m', 0.864))

        optimizer = RefinedHittingOptimizer(
            body_mass_kg=weight_kg, body_height_m=height_m,
            skill_level=skill_level,
            bat_mass_kg=bat_mass_kg, bat_length_m=bat_length_m
        )
        kinematics = optimizer.load_mot_file(file_path)

        trc_name = os.path.basename(file_path).replace('.mot', '.trc')
        # 1. Standard OpenCap folder structure: .../OpenSimData/Kinematics/ -> .../MarkerData/
        trc_path = file_path.replace('Kinematics', 'MarkerData').replace('.mot', '.trc')
        if not os.path.exists(trc_path):
            # 2. Walk up from the .mot file's directory looking for a sibling MarkerData/ folder
            search_dir = os.path.dirname(file_path)
            for _ in range(4):  # up to 4 levels up
                candidate = os.path.join(search_dir, 'MarkerData', trc_name)
                if os.path.exists(candidate):
                    trc_path = candidate
                    break
                search_dir = os.path.dirname(search_dir)
        if not os.path.exists(trc_path):
            # 3. Broad search under ~/Desktop and ~/Downloads as last resort
            for search_root in [os.path.expanduser('~/Desktop'), os.path.expanduser('~/Downloads')]:
                for dirpath, _, fnames in os.walk(search_root):
                    if trc_name in fnames and 'MarkerData' in dirpath:
                        trc_path = os.path.join(dirpath, trc_name)
                        break
        trc_data = optimizer.load_trc_file(trc_path) if os.path.exists(trc_path) else None

        diagnosis = optimizer.comprehensive_diagnosis(kinematics, filename, trc_data=trc_data)

        # Add skeleton frames for visualization (downsampled key markers during swing window)
        if trc_data is not None:
            try:
                diagnosis['skeleton_frames'] = _extract_skeleton_frames(trc_data, kinematics)
            except Exception:
                pass

        model_path = payload.get('model_path', DEFAULT_MODEL)
        _run_id(file_path, model_path, bat_mass_kg, bat_length_m, diagnosis)

        return JSONResponse(content={"filename": filename, "success": True, "data": diagnosis})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/style.css")
def serve_css():
    return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))

@app.get("/app.js")
def serve_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "app.js"))

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
