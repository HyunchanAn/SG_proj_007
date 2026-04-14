import numpy as np
import cv2
import scipy.ndimage as ndimage
import pandas as pd

def create_spherical_cap(R, size=512, pixel_size=0.1):
    """
    R (mm) 반경을 가진 구형 캡 생성
    pixel_size: 픽셀당 mm
    """
    x = np.linspace(-size/2 * pixel_size, size/2 * pixel_size, size)
    y = np.linspace(-size/2 * pixel_size, size/2 * pixel_size, size)
    xx, yy = np.meshgrid(x, y)
    dist_sq = xx**2 + yy**2
    
    mask = dist_sq <= R**2
    z = np.zeros((size, size), dtype=np.float32)
    z[mask] = np.sqrt(R**2 - dist_sq[mask])
    
    # 엣지 노이즈 방지를 위해 80% 영역으로 마스크 축소
    eval_mask = dist_sq <= (R * 0.8)**2
    return z, eval_mask

def estimate_radius_laplacian(z_map, mask, pixel_size=0.1):
    """
    Zxx, Zyy를 이용한 R 추정
    """
    # 2차 편미분 (Laplacian)
    # Zxx + Zyy
    lap = cv2.Laplacian(z_map, cv2.CV_32F, ksize=3)
    # 단위 보정 (pixel_size^2)
    lap_phys = lap / (pixel_size**2)
    
    mean_lap = np.abs(np.mean(lap[mask]))
    if mean_lap < 1e-9: return float('inf')
    
    # 구의 경우 Laplacian ~ 2/R (근사)
    return 2.0 / mean_lap

def simulate_pipeline_bias(R_list):
    results = []
    pixel_size = 0.5 # mm/px
    
    for R_target in R_list:
        z_true, eval_mask = create_spherical_cap(R_target, pixel_size=pixel_size)
        r_true_est = estimate_radius_laplacian(z_true, eval_mask, pixel_size=pixel_size)
        
        # 1. Add typical Noise
        z_noisy = z_true.copy()
        spikes = np.random.rand(*z_true.shape) < 0.01
        z_noisy[spikes] += 20.0 # 20mm spike
        
        # 2. Phase 8 filtering (Median 5x5 + MAD Clipping)
        z_filtered = cv2.medianBlur(z_noisy, 5)
        # Standardize for clipping
        z_med = np.median(z_filtered)
        z_mad = np.median(np.abs(z_filtered - z_med))
        trim = 3.0 * (z_mad * 1.4826 + 1e-7)
        z_clipped = np.clip(z_filtered, z_med - trim, z_med + trim)
        
        r_filtered = estimate_radius_laplacian(z_clipped, eval_mask, pixel_size=pixel_size)
        
        error = (r_filtered - R_target) / R_target * 100
        results.append({
            "Target_R": R_target,
            "Raw_Recovery_R": r_true_est,
            "Filtered_R": r_filtered,
            "Error_Pct": error
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    test_radii = [50, 100, 200, 300, 500]
    print(f"\n[Phase 9 Analysis: Smoothing Bias Linearity Check]")
    df = simulate_pipeline_bias(test_radii)
    print(df.to_markdown(index=False))
    
    # Calculate Calibration Factor (Systematic Bias)
    avg_error = df["Error_Pct"].mean()
    print(f"\nSystematic Index (Avg Overestimation): {avg_error:.2f}%")
    print(f"Proposed Calibration Factor: {100/(100+avg_error):.4f}")
