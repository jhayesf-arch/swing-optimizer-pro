import os
import glob
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from analyzer import RefinedHittingOptimizer

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

# Important: StaticFiles will return 404 if the directory does not exist or index is missing
# So we mount only a specific path for assets, and serve index.html separately.

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@app.post("/api/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    height_m: float = Form(1.83),
    weight_kg: float = Form(82.0),
    skill_level: str = Form('high_school'),
):
    if not file.filename.endswith('.mot'):
        return JSONResponse(status_code=400, content={"success": False, "error": "File must be a .mot file"})
        
    file_path = os.path.join(TMP_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        optimizer = RefinedHittingOptimizer(body_mass_kg=weight_kg, body_height_m=height_m, skill_level=skill_level)
        kinematics = optimizer.load_mot_file(file_path)
        
        if kinematics is None or len(kinematics) == 0:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid .mot file or empty data"})

        diagnosis = optimizer.comprehensive_diagnosis(kinematics, file.filename)
        return JSONResponse(content={"filename": file.filename, "success": True, "data": diagnosis})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.get("/api/scan-downloads")
def scan_downloads():
    try:
        downloads_path = os.path.expanduser("~/Downloads")
        mot_files = glob.glob(os.path.join(downloads_path, "**/*.mot"), recursive=True)
        swing_files = [{"filename": os.path.basename(f), "filepath": f} for f in mot_files]
        return JSONResponse(content={"success": True, "files": swing_files})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/api/analyze/local")
def analyze_local(payload: dict):
    if "filepath" not in payload:
        return JSONResponse(status_code=400, content={"success": False, "error": "Filepath required"})
        
    file_path = payload["filepath"]
    filename = payload.get("filename", os.path.basename(file_path))
    
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"success": False, "error": "File not found in Downloads"})
        
    try:
        height_m = float(payload.get('height_m', 1.83))
        weight_kg = float(payload.get('weight_kg', 82.0))
        skill_level = str(payload.get('skill_level', 'high_school'))
        optimizer = RefinedHittingOptimizer(body_mass_kg=weight_kg, body_height_m=height_m, skill_level=skill_level)
        kinematics = optimizer.load_mot_file(file_path)
        
        trc_path = file_path.replace('Kinematics', 'MarkerData').replace('.mot', '.trc')
        trc_data = None
        if os.path.exists(trc_path):
            trc_data = optimizer.load_trc_file(trc_path)
            
        diagnosis = optimizer.comprehensive_diagnosis(kinematics, filename, trc_data=trc_data)
        return JSONResponse(content={"filename": filename, "success": True, "data": diagnosis})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

