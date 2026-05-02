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

        trc_path = file_path.replace('Kinematics', 'MarkerData').replace('.mot', '.trc')
        trc_data = optimizer.load_trc_file(trc_path) if os.path.exists(trc_path) else None

        diagnosis = optimizer.comprehensive_diagnosis(kinematics, filename, trc_data=trc_data)

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
