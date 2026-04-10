import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class CurvatureAnalyzer:
    def __init__(self, smoothing_sigma: float = 2.0, smoothing_type: str = 'bilateral'):
        """
        Depth Map을 기반으로 3D 표면의 곡률(Gaussian/Mean Curvature)을 분석하는 모듈.
        :param smoothing_sigma: 노이즈 저감을 위한 Smoothing 강도 (Gaussian sigma 또는 Bilateral sigmaSpace)
        :param smoothing_type: 'gaussian' 또는 'bilateral'
        """
        self.sigma = smoothing_sigma
        self.type = smoothing_type

    def smooth_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        사용자 제안 및 전문가 피드백이 반영된 고정밀 뎁스 평활화 로직.
        - float32 정밀도 유지
        - 0~255 선형 사영을 통한 수치 안정성 확보
        - 데이터 표준편차 기반의 동적 sigmaColor 산출
        """
        depth_f32 = depth_map.astype(np.float32)
        d_min, d_max = depth_f32.min(), depth_f32.max()
        
        if d_max <= d_min:
            return depth_f32
            
        # 0 ~ 255 스케일로 정규화
        Z_norm = (depth_f32 - d_min) / (d_max - d_min + 1e-7) * 255.0
        
        # Phase 8: 침엽수림(Impulse Noise) 제거를 위한 비선형 필터링
        # 1. Median Blur (5x5): 뾰족한 스파이크 노이즈 물리적 제거
        Z_norm = cv2.medianBlur(Z_norm, 5)
        
        # 2. 통계적 이상치 클리핑 (MAD 기반) 전문가 피드백 반영
        # 헤비 테일 노이즈에 강건한(Robust) MAD(Median Absolute Deviation) 기준 적용
        z_median = np.median(Z_norm)
        # MAD = median(|x - median(x)|)
        z_mad = np.median(np.abs(Z_norm - z_median))
        # 3.0 * (MAD * 1.4826)는 대략 3시그마와 유사하지만 이상치에 훨씬 덜 영향받음
        trim_range = 3.0 * (z_mad * 1.4826 + 1e-7)
        Z_norm = np.clip(Z_norm, z_median - trim_range, z_median + trim_range)

        if self.type == 'bilateral':
            # 전문가 피드백: 데이터의 표준편차를 기반으로 sigmaColor 동적 결정
            data_std = np.std(Z_norm)
            dynamic_sigma_color = np.clip(data_std * 0.8, 30, 100)
            
            Z_filtered = cv2.bilateralFilter(Z_norm, d=9, sigmaColor=dynamic_sigma_color, sigmaSpace=self.sigma)
            Z = (Z_filtered / 255.0) * (d_max - d_min) + d_min
        else:
            Z_filtered = gaussian_filter(Z_norm, sigma=self.sigma)
            Z = (Z_filtered / 255.0) * (d_max - d_min) + d_min
            
        return Z
        """
        Depth Map 식으로부터 Gaussian Curvature(K)를 계산.
        K = (Zxx * Zyy - Zxy^2) / (1 + Zx^2 + Zy^2)^2
        단, 편미분 계산 전 고주파 노이즈 제거를 위해 Smoothing 적용.
        
        :param depth_map: Depth-Anything-V2에서 추출된 2D Depth 배열
        :param mask: 강판 타겟 영역 마스크 (배경 배제)
        :return: 각 픽셀별 Gaussian Curvature 배열
        """
        # 32비트 고정밀 전처리 적용
        Z = self.smooth_depth(depth_map)
        
        # 1차 편미분 (x, y 방향의 Gradient)
        Zy, Zx = np.gradient(Z)
        
        # 2차 편미분 (Hessian Matrix 성분)
        Zyy, Zyx = np.gradient(Zy)
        Zxy, Zxx = np.gradient(Zx)
        
        # Gaussian Curvature 연산 (K)
        # 평면 = 0, 볼록/오목 = 양수, 말안장(Saddle) = 음수
        numerator = (Zxx * Zyy) - (Zxy ** 2)
        denominator = (1 + Zx**2 + Zy**2) ** 2
        
        K = numerator / denominator
        
        # 마스크 바깥(배경) 영역의 곡률은 0으로 날림
        if mask is not None:
            K[~mask] = 0.0
            
        return K

    def calculate_mean_curvature(self, depth_map: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Depth Map 식으로부터 Mean Curvature(H)를 계산.
        H = ((1 + Zx^2)*Zyy - 2*Zx*Zy*Zxy + (1 + Zy^2)*Zxx) / (2*(1 + Zx^2 + Zy^2)^(3/2))
        """
        # 32비트 고정밀 전처리 적용
        Z = self.smooth_depth(depth_map)
        
        Zy, Zx = np.gradient(Z)
        Zyy, Zyx = np.gradient(Zy)
        Zxy, Zxx = np.gradient(Zx)
        
        numerator = (1 + Zx**2)*Zyy - 2*Zx*Zy*Zxy + (1 + Zy**2)*Zxx
        denominator = 2 * (1 + Zx**2 + Zy**2)**1.5
        
        # 방어 로직: 분모가 0이 되는 것을 방지
        H = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        if mask is not None:
            H[~mask] = 0.0
            
        return H
        
    def find_critical_points(self, curvature_map: np.ndarray, mask: np.ndarray = None, top_k: int = 5):
        """
        곡률이 가장 심한 임계 영역(Stress Concentration)의 픽셀 좌표를 도출.
        절대값이 클수록 급격하게 꺾임.
        :param curvature_map: Gaussian 또는 Mean Curvature 배열
        :return: (상위 k개의 최대 곡률값, 해당 좌표 리스트)
        """
        search_space = np.abs(curvature_map)
        
        if mask is not None:
            search_space[~mask] = 0.0

        # 가장 곡률 절대값이 높은 인덱스 찾기
        flat_indices = np.argsort(search_space, axis=None)[::-1][:top_k]
        
        # 1D 인덱스를 2D y, x 좌표로 변환
        coords = [np.unravel_index(idx, search_space.shape) for idx in flat_indices]
        values = [curvature_map[y, x] for y, x in coords]
        
        return values, coords

# 테스트 블럭 (직접 실행 시)
if __name__ == "__main__":
    analyzer = CurvatureAnalyzer(smoothing_sigma=2.0)
    
    # 가상의 반구형 뎁스맵 생성 (중심이 가장 볼록함)
    h, w = 100, 100
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    # 반경 30인 구의 상단부 (Z = sqrt(R^2 - x^2 - y^2))
    r2 = 30**2
    mask = (x**2 + y**2) < r2
    dummy_depth = np.zeros((h, w), dtype=np.float32)
    dummy_depth[mask] = np.sqrt(r2 - x[mask]**2 - y[mask]**2)
    
    # 계산
    gaussian_c = analyzer.calculate_gaussian_curvature(dummy_depth, mask=mask)
    
    # 결과 출력
    critical_vals, critical_coords = analyzer.find_critical_points(gaussian_c, mask=mask, top_k=3)
    print("\n[Curvature Analysis Mock Results]")
    for i, (val, coord) in enumerate(zip(critical_vals, critical_coords), 1):
         print(f"Top {i} Stress Point at {coord}: Curvature = {val:.4f}")
