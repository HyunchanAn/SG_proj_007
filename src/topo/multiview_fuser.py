import numpy as np
import cv2
import open3d as o3d
import copy

class MultiViewFuser:
    def __init__(self, voxel_size=0.05):
        """
        초정밀 Multi-View 3D Point Cloud 퓨전 시스템
        :param voxel_size: 포인트 클라우드 병합 최적화를 위한 Voxel 크기
        """
        self.voxel_size = voxel_size
        
        # SIFT/ORB 기반 피처 매처 생성
        self.sift = cv2.SIFT_create()
        # BF 매치 (KDTree 사용)
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_features(self, image):
        """이미지에서 특징점(Keypoints)과 기술자(Descriptors) 추출"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        return kp, des

    def calculate_sharpness(self, image):
        """라플라시안 분산을 이용한 이미지 선명도 측정"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_entropy(self, image):
        """이미지의 정보량(Entropy) 측정"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist /= hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        return entropy

    def select_reference_view(self, images):
        """선명도(0.7)와 정보량(0.3) 가중합을 통한 최적의 레퍼런스 뷰 선정"""
        scores = []
        for img in images:
            sharp = self.calculate_sharpness(img)
            ent = self.calculate_entropy(img)
            # 정규화 (배치 내 상대값)
            scores.append((sharp, ent))
        
        sharp_vals = np.array([s[0] for s in scores])
        ent_vals = np.array([s[1] for s in scores])
        
        # Min-Max Normalization
        s_norm = (sharp_vals - sharp_vals.min()) / (sharp_vals.max() - sharp_vals.min() + 1e-7)
        e_norm = (ent_vals - ent_vals.min()) / (ent_vals.max() - ent_vals.min() + 1e-7)
        
        final_scores = s_norm * 0.7 + e_norm * 0.3
        return np.argmax(final_scores)

    def match_histograms(self, source, reference):
        """상대적 뎁스 스케일 통일을 위한 히스토그램 매칭 (Histogram Matching)"""
        # 0.0 ~ 1.0 범위를 가정하고 처리 (또는 uint8 변환 후 처리)
        s_flat = source.flatten()
        r_flat = reference.flatten()
        
        s_values, bin_idx, s_counts = np.unique(s_flat, return_inverse=True, return_counts=True)
        r_values, r_counts = np.unique(r_flat, return_counts=True)
        
        s_quantiles = np.cumsum(s_counts).astype(np.float64) / s_flat.size
        r_quantiles = np.cumsum(r_counts).astype(np.float64) / r_flat.size
        
        interp_values = np.interp(s_quantiles, r_quantiles, r_values)
        return interp_values[bin_idx].reshape(source.shape)

    def create_point_cloud(self, rgb, depth, scale_factor=1.0, z_scale=1.0, fov_scale=1.0, fx=1200, fy=1200, cx=None, cy=None, anchor_pt=None):
        """RGB와 Depth 맵을 기반으로 Open3D Point Cloud 객체 생성 (실제 스케일 반영)"""
        h, w = depth.shape
        if cx is None: cx = w / 2
        if cy is None: cy = h / 2
        
        # Apply FOV Scale to focal length (default 1200 for smartphone-like)
        fx_adj = fx * fov_scale
        fy_adj = fy * fov_scale
        
        # Intrinsic Matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx_adj, fy_adj, cx, cy)
        
        # 8비트 RGB
        rgb_o3d = o3d.geometry.Image(rgb)
        
        # Depth-Anything의 상대적 깊이를 물리적 mm 단위로 정규화 및 스케일링
        # (상대 깊이 0.0~1.0 가정 -> scale_factor를 곱해 실제 mm로 변환)
        depth_mm = depth.astype(np.float32) * scale_factor * z_scale
        depth_o3d = o3d.geometry.Image(depth_mm.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 다운샘플링 및 노이즈 제거 (통계적 방식)
        pcd = pcd.voxel_down_sample(self.voxel_size)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.0)
        pcd = pcd.select_by_index(ind)
        
        # 추가: 반경 기반 노이즈 제거 (고립된 점 제거)
        cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=self.voxel_size * 2)
        pcd = pcd.select_by_index(ind)
        
        # Optional Anchor Point 3D extraction
        a_3d = None
        if anchor_pt is not None:
            u, v = anchor_pt
            # Bounds check
            u_idx = int(np.clip(u, 0, w - 1))
            v_idx = int(np.clip(v, 0, h - 1))
            z_a = depth_mm[v_idx, u_idx]
            x_a = (u - cx) * z_a / fx_adj
            y_a = (v - cy) * z_a / fy_adj
            a_3d = np.array([x_a, y_a, z_a])
            
        return pcd, a_3d

    def extract_fpfh(self, pcd):
        """기하 구조 특징점(FPFH) 추출 최적화 (저질감 대응)"""
        # 법선 추정 범위 확장 (부드러운 평면에서의 일관성 확보)
        radius_normal = self.voxel_size * 5
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # FPFH 특징점 추출 범위 확장
        radius_feature = self.voxel_size * 10
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return fpfh

    def register_icp(self, source, target, source_anchor=None, target_anchor=None):
        """FPFH 기반 전역 정합 후 정밀 ICP 수행 (Global-to-Local 전략)"""
        
        # 1. 초기 추정치 계산
        initial_trans = np.eye(4)
        distance_threshold = self.voxel_size * 1.5
        
        if source_anchor is not None and target_anchor is not None:
            # Anchor 간의 차이로 일차 정밀 정렬
            diff = target_anchor - source_anchor
            initial_trans[:3, 3] = diff
        else:
            # 수동 앵커가 없으면 FPFH 기반 전역 정합 (Shape-based Matching)
            source_fpfh = self.extract_fpfh(source)
            target_fpfh = self.extract_fpfh(target)
            
            # RANSAC 전역 정합
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source, target, source_fpfh, target_fpfh, True,
                distance_threshold * 2,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold * 2)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )
            initial_trans = result_ransac.transformation

        # 2. 로컬 ICP 정밀 보정 (Generalized ICP 스타일)
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, initial_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result_icp.transformation, result_icp.fitness

    def fuse_views(self, rgb_list, depth_list, scale_factor=1.0, z_scale=1.0, fov_scale=1.0, anchor_coords=None):
        """지능형 레퍼런스 선정 및 히스토그램 매칭 기반 다중 뷰 정합"""
        if len(rgb_list) < 1:
            raise ValueError("최소 1개 이상의 이미지가 필요합니다.")
            
        # 1. 최적의 레퍼런스 뷰 자동 선정
        ref_idx = self.select_reference_view(rgb_list)
        print(f"[MultiView] Best Reference Selected: Index {ref_idx}")
        
        # 2. 히스토그램 매칭을 통한 전체 뷰의 뎁스 스케일 동기화
        synced_depths = []
        ref_depth = depth_list[ref_idx]
        for i, d in enumerate(depth_list):
            if i == ref_idx:
                synced_depths.append(d)
            else:
                synced_depths.append(self.match_histograms(d, ref_depth))
        
        # 3. 정합 베이스 생성 (선정된 레퍼런스 뷰 기준)
        ref_anchor_uv = anchor_coords[ref_idx] if anchor_coords else None
        accumulated_pcd, ref_anchor_3d = self.create_point_cloud(
            rgb_list[ref_idx], synced_depths[ref_idx], 
            scale_factor=scale_factor, z_scale=z_scale, fov_scale=fov_scale,
            anchor_pt=ref_anchor_uv
        )
        accumulated_pcd.estimate_normals()
        
        if len(rgb_list) == 1:
            return accumulated_pcd, 1.0
        
        total_fitness = 1.0
        for i in range(len(rgb_list)):
            if i == ref_idx: continue
            
            print(f"[MultiView] Aligning view {i} to reference...")
            cur_anchor_uv = anchor_coords[i] if anchor_coords else None
            source_pcd, cur_anchor_3d = self.create_point_cloud(
                rgb_list[i], synced_depths[i], 
                scale_factor=scale_factor, z_scale=z_scale, fov_scale=fov_scale,
                anchor_pt=cur_anchor_uv
            )
            source_pcd.estimate_normals()
            
            transform_mat, fitness = self.register_icp(source_pcd, accumulated_pcd, source_anchor=cur_anchor_3d, target_anchor=ref_anchor_3d)
            source_pcd.transform(transform_mat)
            total_fitness = min(total_fitness, fitness)
            
            accumulated_pcd += source_pcd
            accumulated_pcd = accumulated_pcd.voxel_down_sample(self.voxel_size)
            
        # 4. 포아송 재구성을 통한 고품질 표면 평활화 (Phase 8 최적화)
        # 전문가 피드백: 연산력을 활용하여 해상도(depth)를 9로 복구 (고해상도 지형 정보 유지)
        print("[MultiView] Performing Poisson Surface Reconstruction (Depth=9)...")
        accumulated_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # 포아송 재구성 수행 (지능형 트리밍 수반)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(accumulated_pcd, depth=9)
        
        # 전문가 피드백: 밀도(Density) 기반 적응형 트리밍 알고리즘
        # 단순히 하위 %를 자르는 것이 아니라, 평균 밀도와의 거리를 고려하여 고립된 평탄면 유실 방지
        densities = np.asanyarray(densities)
        density_mean = np.mean(densities)
        density_std = np.std(densities)
        # 평균 대비 현저히 낮은 밀도(예: mean - 1.5*std) 영역만 선택적 제거
        trim_threshold = max(density_mean - 1.5 * density_std, np.quantile(densities, 0.05))
        vertices_to_remove = densities < trim_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Mesh Smoothing (Taubin): 부피 수축 없는 평활화
        # 반복 횟수를 15회로 증량하여 노이즈 제거력 강화하되, R값 왜곡 방지를 위해 lambda/mu 조절
        mesh = mesh.filter_smooth_taubin(number_of_iterations=15, lamb=0.5, mu=-0.53)
        
        # 최종 포인트 클라우드로 다시 샘플링 (원본 밀도 유지 및 R값 정밀도 확보)
        final_pcd = mesh.sample_points_uniformly(number_of_points=len(accumulated_pcd.points))
        
        return final_pcd, total_fitness
