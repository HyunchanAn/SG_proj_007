import os
import sys
import numpy as np
import cv2
import torch
import time

# Add src to path
sys.path.append(os.getcwd())

from src.topo.depth_wrapper import DepthAnythingV2Wrapper
from src.curv.curvature import CurvatureAnalyzer

def calculate_rms_roughness(depth_map):
    """지형의 RMS 거칠기 계산 (평면 피팅 후 잔차)"""
    h, w = depth_map.shape
    y, x = np.indices((h, w))
    
    # Plane fitting: Z = aX + bY + c
    A = np.column_stack([x.ravel(), y.ravel(), np.ones(h*w)])
    B = depth_map.ravel()
    
    # Least squares
    coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    a, b, c = coeffs
    
    # Target plane
    fitted_plane = a*x + b*y + c
    
    # Residuals
    residuals = depth_map - fitted_plane
    rms = np.sqrt(np.mean(residuals**2))
    return rms

def main():
    test_img_path = "260409 007 test images/KakaoTalk_20260409_135200065.jpg"
    if not os.path.exists(test_img_path):
        print(f"Test image not found at {test_img_path}")
        return

    print(f"--- Phase 7 Verification: {os.path.basename(test_img_path)} ---")
    
    # 1. Load Model
    dw = DepthAnythingV2Wrapper()
    dw.load_model()
    
    # 2. Load Image
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Raw Depth Estimation
    print("Estimating Raw Depth...")
    raw_depth = dw.estimate_depth(img)
    
    # 4. Calculate Raw RMS
    raw_rms = calculate_rms_roughness(raw_depth)
    print(f"Raw Depth RMS Roughness: {raw_rms:.6f}")
    
    # 5. Apply Phase 7 Bilateral Filter
    print("Applying Phase 7 High-Precision Bilateral Filter...")
    # CurvatureAnalyzer를 사용하여 필터링 (sigma=2.0, type='bilateral')
    analyzer = CurvatureAnalyzer(smoothing_sigma=2.0, smoothing_type='bilateral')
    
    # calculate_gaussian_curvature 내부의 필터링 로직을 시뮬레이션하거나 직접 호출
    # 여기서는 필터링된 Z를 얻기 위해 내부 로직 분리 호출 (또는 analyzer의 특정 메서드 이용)
    # Z = cv2.bilateralFilter(Z_norm, d=9, sigmaColor=75, sigmaSpace=2.0)
    
    d_min, d_max = raw_depth.min(), raw_depth.max()
    Z_norm = (raw_depth - d_min) / (d_max - d_min + 1e-7) * 255.0
    filtered_norm = cv2.bilateralFilter(Z_norm.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=2.0)
    filtered_depth = (filtered_norm / 255.0) * (d_max - d_min) + d_min
    
    # 6. Calculate Filtered RMS
    filtered_rms = calculate_rms_roughness(filtered_depth)
    print(f"Filtered (Phase 7) Depth RMS Roughness: {filtered_rms:.6f}")
    
    improvement = (raw_rms - filtered_rms) / raw_rms * 100
    print(f"\n[Result] Surface Noise Reduction: {improvement:.2f}%")
    print(f"[Verdict] RMS Improvement: {raw_rms:.6f} -> {filtered_rms:.6f}")

if __name__ == "__main__":
    main()
