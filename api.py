import io
import time
import cv2
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import urllib.request
import os

from sg_terra.seg.sam2_wrapper import SAM2BaseWrapper
from sg_terra.topo.depth_wrapper import DepthAnythingV2Wrapper
from sg_terra.curv.curvature import CurvatureAnalyzer
from sg_terra.match.engine import KnowledgeEngine

# Global objects to hold models
models: Dict[str, Any] = {}

def download_models_if_needed():
    """Download the models if they don't exist, based on DEPLOY_ENV"""
    deploy_env = os.environ.get("DEPLOY_ENV", "local")
    
    if deploy_env == "cloud":
        sam2_cfg = "sam2_hiera_s.yaml"
        sam2_ckpt = "models/sam2/sam2_hiera_small.pt"
        sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
        
        depth_encoder = "vits"
        depth_ckpt = "models/depth_anything_v2/depth_anything_v2_vits.pth"
        depth_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
    else:
        sam2_cfg = "sam2_hiera_l.yaml"
        sam2_ckpt = "models/sam2/sam2_hiera_large.pt"
        sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
        
        depth_encoder = "vitl"
        depth_ckpt = "models/depth_anything_v2/depth_anything_v2_vitl.pth"
        depth_url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
        
    os.makedirs("models/sam2", exist_ok=True)
    os.makedirs("models/depth_anything_v2", exist_ok=True)
    
    if not os.path.exists(sam2_ckpt):
        print(f"Downloading SAM 2 ({deploy_env})...")
        urllib.request.urlretrieve(sam2_url, sam2_ckpt)
        
    if not os.path.exists(depth_ckpt):
        print(f"Downloading Depth-Anything-V2 ({deploy_env})...")
        urllib.request.urlretrieve(depth_url, depth_ckpt)
    
    return sam2_cfg, sam2_ckpt, depth_encoder, depth_ckpt

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading SG-TERRA AI Models...")
    sam2_cfg, sam2_ckpt, depth_encoder, depth_ckpt = download_models_if_needed()
    
    sam_wrapper = SAM2BaseWrapper(model_cfg=sam2_cfg, checkpoint_path=sam2_ckpt)
    depth_wrapper = DepthAnythingV2Wrapper(encoder=depth_encoder, checkpoint_path=depth_ckpt)
    curv_analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    match_engine = KnowledgeEngine(db_path="data/database/film_properties.csv")
    
    sam_wrapper.load_model()
    depth_wrapper.load_model()
    
    models["sam"] = sam_wrapper
    models["depth"] = depth_wrapper
    models["curv"] = curv_analyzer
    models["match"] = match_engine
    
    print("All models loaded successfully.")
    yield
    # Shutdown
    models.clear()

app = FastAPI(title="SG-TERRA Headless API", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": len(models) > 0}

@app.post("/api/v1/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    ref_length_mm: float = Form(100.0),
    roughness: float = Form(1.0),
    click_x: Optional[str] = Form(None),
    click_y: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Invalid image file")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")

    start_total = time.time()
    
    prompt_points = None
    prompt_labels = None
    pixel_to_mm = 1.0
    
    # Process clicks if provided (expects comma separated values if multiple)
    if click_x and click_y:
        xs = [int(x.strip()) for x in click_x.split(",")]
        ys = [int(y.strip()) for y in click_y.split(",")]
        pts = list(zip(xs, ys))
        
        if len(pts) >= 2:
            p1, p2 = pts[0], pts[1]
            dist_px = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if dist_px > 0:
                pixel_to_mm = ref_length_mm / dist_px
            # Center of the first two points
            cx, cy = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
            prompt_points = np.array([[cx, cy]])
            prompt_labels = np.array([1])
        elif len(pts) == 1:
            prompt_points = np.array([[pts[0][0], pts[0][1]]])
            prompt_labels = np.array([1])
            
    # If no prompt provided, we just rely on SAM 2 fallback behavior
    # Step 1: Segmentation
    t0 = time.time()
    target_mask = models["sam"].segment_target(img_rgb, prompt_points=prompt_points, prompt_labels=prompt_labels)
    t_seg = time.time() - t0
    
    # Step 2: Depth
    t0 = time.time()
    depth_map = models["depth"].estimate_depth(img_rgb, mask=target_mask)
    t_depth = time.time() - t0
    
    # Step 3: Curvature
    t0 = time.time()
    gaussian_curv = models["curv"].calculate_gaussian_curvature(depth_map, mask=target_mask)
    critical_vals, critical_coords = models["curv"].find_critical_points(gaussian_curv, mask=target_mask, top_k=1)
    t_curv = time.time() - t0
    
    highest_stress_raw = float(critical_vals[0])
    y_max, x_max = int(critical_coords[0][0]), int(critical_coords[0][1])
    
    r_pixel = 1.0 / np.sqrt(np.abs(highest_stress_raw)) if highest_stress_raw != 0 else 0
    estimated_r_mm = round(float(r_pixel * pixel_to_mm), 2)
    
    # Step 4: Matching
    t0 = time.time()
    recommendations = models["match"].recommend(measured_curvature=estimated_r_mm, measured_roughness=roughness)
    t_match = time.time() - t0
    
    t_total = time.time() - start_total
    
    return JSONResponse(content={
        "status": "success",
        "latency_ms": {
            "segmentation": round(t_seg * 1000, 2),
            "depth": round(t_depth * 1000, 2),
            "curvature": round(t_curv * 1000, 2),
            "matching": round(t_match * 1000, 2),
            "total": round(t_total * 1000, 2)
        },
        "metrics": {
            "max_gaussian_curvature_raw": highest_stress_raw,
            "estimated_radius_mm": estimated_r_mm,
            "critical_point_coords": {"x": x_max, "y": y_max},
            "pixel_to_mm_scale": round(pixel_to_mm, 4)
        },
        "recommendations": recommendations
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
