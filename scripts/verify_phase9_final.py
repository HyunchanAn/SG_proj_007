import numpy as np
import cv2

def sphere_fit(points):
    """
    Least Squares Sphere Fit
    points: (N, 3) 
    (x-a)^2 + (y-b)^2 + (z-c)^2 = R^2
    """
    A = np.zeros((len(points), 4))
    A[:, 0] = points[:, 0] * 2
    A[:, 1] = points[:, 1] * 2
    A[:, 2] = points[:, 2] * 2
    A[:, 3] = 1
    
    f = points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2
    
    # Solve C = (A^T * A)^-1 * A^T * f
    C, resid, rank, s = np.linalg.lstsq(A, f, rcond=None)
    
    a, b, c = C[0], C[1], C[2]
    R = np.sqrt(C[3] + a**2 + b**2 + c**2)
    return a, b, c, R

def simulate_verification():
    h, w = 256, 256
    pixel_size = 0.2 # mm/px
    radii = [50, 100, 200, 300, 500]
    
    print(f"{'Target R (mm)':>15} | {'Measured R (mm)':>15} | {'Error (%)':>10}")
    print("-" * 46)
    
    for R_true in radii:
        # 1. Create true surface
        y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
        dist_sq = (x*pixel_size)**2 + (y*pixel_size)**2
        mask = dist_sq < (R_true*0.9)**2
        z = np.zeros((h, w), dtype=np.float32)
        z[mask] = np.sqrt(R_true**2 - dist_sq[mask])
        
        # 2. Add realistic Noise
        z_noisy = z.copy()
        spikes = np.random.rand(h, w) < 0.01
        z_noisy[spikes] += 10.0 # 10mm spikes
        
        # 3. Apply Phase 9 Adaptive Filtering (Simulated)
        # Median -> Adaptive MAD -> (Poisson logic is modeled by high-res fit)
        z_filtered = cv2.medianBlur(z_noisy, 5)
        
        # MAD Clipping
        z_med = np.median(z_filtered)
        z_mad = np.median(np.abs(z_filtered - z_med))
        z_clipped = np.clip(z_filtered, z_med - 4.0*z_mad*1.4826, z_med + 4.0*z_mad*1.4826)
        
        # Extract points for fitting
        xv, yv = np.meshgrid(x * pixel_size, y * pixel_size)
        pts_x = xv[mask]
        pts_y = yv[mask]
        pts_z = z_clipped[mask]
        pts = np.column_stack((pts_x, pts_y, pts_z))
        
        _, _, _, R_measured = sphere_fit(pts)
        error = (R_measured - R_true) / R_true * 100
        
        print(f"{R_true:15.2f} | {R_measured:15.2f} | {error:10.2f}%")

if __name__ == "__main__":
    simulate_verification()
