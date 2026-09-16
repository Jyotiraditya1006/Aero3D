import shutil
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from aero3d.pipeline import run_pipeline
from aero3d.synthetic import generate_synthetic_mission

app = FastAPI(title="Aero3D Web Backend")

# We will serve the static outputs directly
OUTPUTS_DIR = Path("outputs/web")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="website"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
def read_index():
    return FileResponse("website/index.html")

@app.post("/generate")
def generate_model(
    video: Optional[UploadFile] = File(None),
    telemetry: Optional[UploadFile] = File(None),
    is_synthetic: bool = Form(False)
):
    run_dir = OUTPUTS_DIR / "current_run"
    
    try:
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        if is_synthetic:
            info = generate_synthetic_mission(run_dir / "mission")
            vid_path = info["video"]
            tel_path = info["telemetry"]
        else:
            if not video:
                return JSONResponse(status_code=400, content={"error": "Upload a video."})
            
            vid_path = run_dir / video.filename
            with open(vid_path, "wb") as f:
                f.write(video.file.read())
                
            if telemetry:
                tel_path = run_dir / telemetry.filename
                with open(tel_path, "wb") as f:
                    f.write(telemetry.file.read())
            else:
                tel_path = vid_path  # Pipeline will extract SRT from video
        
        out_model_dir = run_dir / "model"
        
        # Run pipeline
        result = run_pipeline(vid_path, tel_path, out_model_dir)
        
        # Zip the results
        zip_path = run_dir / "aero3d_model.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            if (out_model_dir / "model_cloud.ply").exists():
                zf.write(out_model_dir / "model_cloud.ply", "model_cloud.ply")
            if (out_model_dir / "model_mesh.ply").exists():
                zf.write(out_model_dir / "model_mesh.ply", "model_mesh.ply")
            if (out_model_dir / "report.json").exists():
                zf.write(out_model_dir / "report.json", "report.json")
                
        return {"status": "success", "download_url": "/download/aero3d_model.zip", "metrics": result.metrics}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = OUTPUTS_DIR / "current_run" / filename
    if file_path.exists():
        return FileResponse(file_path, filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
