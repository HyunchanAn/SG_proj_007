import os
import time
import numpy as np
import cv2

# import pipeline modules
from sg_terra.seg.sam2_wrapper import SAM2BaseWrapper
from sg_terra.topo.depth_wrapper import DepthAnythingV2Wrapper
from sg_terra.curv.curvature import CurvatureAnalyzer

def main():
    print("==================================================")
    print("SG-TERRA Pipeline Initialization (Mock Test)")
    print("==================================================")
    
    # 1. Load Dummy Image (1080p)
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    print("[1] Loaded dummy image (1920x1080)")
    
    # 2. Module Initialization
    print("[2] Initializing Modules...")
    sam_wrapper = SAM2BaseWrapper()
    depth_wrapper = DepthAnythingV2Wrapper()
    curv_analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    

    
    print("\n--------------------------------------------------")
    print("Executing Pipeline...")
    start_total = time.time()
    
    # Step 1: Target Segmentation (SAM 2)
    t0 = time.time()
    target_mask = sam_wrapper.segment_target(dummy_img)
    t_seg = time.time() - t0
    
    # Step 2: Depth Estimation (Depth-Anything-V2)
    # Using Resolution Cropping Strategy (passing the mask to ignore background)
    t0 = time.time()
    depth_map = depth_wrapper.estimate_depth(dummy_img, mask=target_mask)
    t_depth = time.time() - t0
    
    # Step 3: Curvature Analysis
    t0 = time.time()
    # Calculate Gaussian Curvature
    gaussian_curv = curv_analyzer.calculate_gaussian_curvature(depth_map, mask=target_mask)
    
    # Find top stress concentration points
    # In a real scenario, we map this curvature map value to a physical scale (mm)
    # Here, we will just use a mock scaling for demonstration
    critical_vals, critical_coords = curv_analyzer.find_critical_points(gaussian_curv, mask=target_mask, top_k=1)
    
    t_curv = time.time() - t0
    
    # convert highest curvature to a mock physical radius R (mm)
    # For testing, we assume the highest stress point translates to a 3.5mm radius
    highest_stress_val = critical_vals[0]
    estimated_r_mm = 3.5 
    print(f"\n[Analysis Result] Max Stress Point found at {critical_coords[0]} with raw curvature {highest_stress_val:.4f}")
    print(f"[Analysis Result] Estimated Minimum Curvature Radius (R) = {estimated_r_mm} mm")
    

    
    end_total = time.time()
    
    print("\n--------------------------------------------------")

    
    print("\n--------------------------------------------------")
    print(f"Pipeline Latency Summary:")
    print(f" - Segmentation (SAM 2)     : {t_seg*1000:.2f} ms")
    print(f" - Depth Est. (Depth-V2)    : {t_depth*1000:.2f} ms")
    print(f" - Curvature Analysis       : {t_curv*1000:.2f} ms")

    print(f" = Total Pipeline Execution : {(end_total - start_total)*1000:.2f} ms")

if __name__ == "__main__":
    main()
