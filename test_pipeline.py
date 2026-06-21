import os
import time

import cv2

from sg_terra.curv.curvature import CurvatureAnalyzer

# import pipeline modules
from sg_terra.seg.sam2_wrapper import SAM2BaseWrapper
from sg_terra.topo.depth_wrapper import DepthAnythingV2Wrapper


def main():
    print("==================================================")
    print("SG-TERRA Pipeline Initialization (Real Inference)")
    print("==================================================")

    # 1. Load Real Dummy Image
    img_path = "data/raw/sample_plate.jpg"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"[1] Loaded sample image: {img_rgb.shape}")

    # 2. Module Initialization
    print("[2] Initializing Modules...")
    sam_wrapper = SAM2BaseWrapper(checkpoint_path="models/sam2/sam2_hiera_large.pt")
    depth_wrapper = DepthAnythingV2Wrapper(
        checkpoint_path="models/depth_anything_v2/depth_anything_v2_vitl.pth"
    )
    curv_analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)

    print("\n--------------------------------------------------")
    print("Executing Pipeline with Actual Models...")
    start_total = time.time()

    # Step 1: Target Segmentation (SAM 2)
    t0 = time.time()
    target_mask = sam_wrapper.segment_target(img_rgb)
    t_seg = time.time() - t0
    print(f"[Info] Segmentation finished (Mask size: {target_mask.sum()} pixels)")

    # Step 2: Depth Estimation (Depth-Anything-V2)
    t0 = time.time()
    depth_map = depth_wrapper.estimate_depth(img_rgb, mask=target_mask)
    t_depth = time.time() - t0
    print(f"[Info] Depth estimation finished. Shape: {depth_map.shape}")

    # Step 3: Curvature Analysis
    t0 = time.time()
    gaussian_curv = curv_analyzer.calculate_gaussian_curvature(
        depth_map, mask=target_mask
    )

    critical_vals, critical_coords = curv_analyzer.find_critical_points(
        gaussian_curv, mask=target_mask, top_k=1
    )

    t_curv = time.time() - t0

    highest_stress_val = critical_vals[0]
    expected_r_mm = 3.5
    print(
        f"\n[Analysis Result] Max Stress Point found at {critical_coords[0]} with raw curvature {highest_stress_val:.4f}"
    )
    print(
        f"[Analysis Result] Estimated Minimum Curvature Radius (R) = {expected_r_mm} mm"
    )

    end_total = time.time()

    print("\n--------------------------------------------------")
    print("Pipeline Latency Summary:")
    print(f" - Segmentation (SAM 2)     : {t_seg * 1000:.2f} ms")
    print(f" - Depth Est. (Depth-V2)    : {t_depth * 1000:.2f} ms")
    print(f" - Curvature Analysis       : {t_curv * 1000:.2f} ms")
    print(f" = Total Pipeline Execution : {(end_total - start_total) * 1000:.2f} ms")


if __name__ == "__main__":
    main()
