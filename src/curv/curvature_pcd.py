import numpy as np
import open3d as o3d

class PCDCurvatureAnalyzer:
    def __init__(self, knn=30):
        """
        3D Point Cloud 데이터를 기반으로 표면의 곡률(Surface Variation)을 분석.
        :param knn: 로컬 공분산 행렬 계산을 위한 이웃 포인트 개수
        """
        self.knn = knn

    def calculate_curvature(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        PCA 기반의 Surface Variation (Curvature) 계산.
        sigma = lambda_0 / (lambda_0 + lambda_1 + lambda_2)
        """
        # KDTree 구축
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        num_points = len(points)
        curvatures = np.zeros(num_points)

        for i in range(num_points):
            # i번째 포인트의 knn 이웃 검색
            [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], self.knn)
            
            # 이웃 포인트들 추출
            neighbors = points[idx, :]
            
            if len(neighbors) < 3:
                curvatures[i] = 0
                continue
            
            # 로컬 공분산 행렬 계산
            cov = np.cov(neighbors.T)
            
            # 고윳값 추출 (오름차순 정렬)
            eigenvalues = np.linalg.eigvalsh(cov)
            
            # Surface Variation 계산
            sum_lambda = np.sum(eigenvalues)
            if sum_lambda > 0:
                curvatures[i] = eigenvalues[0] / sum_lambda
            else:
                curvatures[i] = 0
                
        return curvatures

    def estimate_min_radius(self, curvatures: np.ndarray, scale_factor: float = 1.0) -> float:
        """
        계산된 곡률(variation) 수치를 기반으로 물리적 최소 곡률 반경(R)을 추정.
        참고: 정교한 매핑을 위해 실험적 상수 또는 기하학적 관계식 필요.
        현재는 휴리스틱 모델을 사용함.
        """
        # 상위 1% 급격한 곡률의 평균값 사용 (노이즈 방지)
        sorted_curv = np.sort(curvatures)[::-1]
        critical_idx = max(int(len(sorted_curv) * 0.01), 1)
        max_curv_val = np.mean(sorted_curv[:critical_idx])
        
        if max_curv_val <= 0:
            return 1000.0  # 평면인 경우 매우 큰 반경 반환
            
        # 휴리스틱: 곡률이 클수록 반경은 작음 (R = K / curvature)
        # K는 캘리브레이션된 스케일에 따라 조정될 수 있는 실험 상수
        K = 2.0 
        estimated_r = K / (max_curv_val + 1e-6)
        
        # 최소 물리적 한계치 보정
        return max(estimated_r, 0.5) 

# 간단한 검증 테스트
if __name__ == "__main__":
    # 구형(Sphere) PCD 생성 테스트
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10.0)
    pcd = mesh_sphere.sample_points_uniformly(number_of_points=2000)
    
    analyzer = PCDCurvatureAnalyzer(knn=50)
    curvs = analyzer.calculate_curvature(pcd)
    r = analyzer.estimate_min_radius(curvs)
    
    print(f"Sphere (R=10) Estimated R: {r:.2f}")
    print(f"Max Curvature Value: {np.max(curvs):.4f}")
