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
        return result_icp.transformation

    def fuse_views(self, rgb_list, depth_list, scale_factor=1.0, z_scale=1.0, fov_scale=1.0, anchor_coords=None):
        """다중 뷰 이미지 리스트와 깊이 리스트를 통째로 정합 (스케일 반영 및 수동 앵커 지원)"""
        if len(rgb_list) < 1:
            raise ValueError("최소 1개 이상의 이미지가 필요합니다.")
            
        print(f"[MultiView] Calibration Scale(factor={scale_factor:.4f}) applying...")
        
        # Reference Anchor
        ref_anchor_uv = anchor_coords[0] if anchor_coords else None
        accumulated_pcd, ref_anchor_3d = self.create_point_cloud(
            rgb_list[0], depth_list[0], 
            scale_factor=scale_factor, z_scale=z_scale, fov_scale=fov_scale,
            anchor_pt=ref_anchor_uv
        )
        accumulated_pcd.estimate_normals()
        
        # 뷰가 1개뿐이면 정합 과정을 건너뛰고 바로 반환
        if len(rgb_list) == 1:
            print("[MultiView] 단일 뷰 스케일 보정 완료.")
            return accumulated_pcd
        
        for i in range(1, len(rgb_list)):
            print(f"[MultiView] {i+1}th view refined matching (ICP with Anchor)...")
            cur_anchor_uv = anchor_coords[i] if anchor_coords else None
            source_pcd, cur_anchor_3d = self.create_point_cloud(
                rgb_list[i], depth_list[i], 
                scale_factor=scale_factor, z_scale=z_scale, fov_scale=fov_scale,
                anchor_pt=cur_anchor_uv
            )
            source_pcd.estimate_normals()
            
            # 정합 진행 (앵커 3D 좌표 전달)
            transform_mat = self.register_icp(source_pcd, accumulated_pcd, source_anchor=cur_anchor_3d, target_anchor=ref_anchor_3d)
            source_pcd.transform(transform_mat)
            
            # 합체 및 다운샘플링
            accumulated_pcd += source_pcd
            accumulated_pcd = accumulated_pcd.voxel_down_sample(self.voxel_size)
            
            # 병합 후 노이즈 재필터링 (정합 오차로 인한 가시 현상 방지)
            accumulated_pcd, _ = accumulated_pcd.remove_statistical_outlier(nb_neighbors=40, std_ratio=1.0)
            # 포인트 클라우드는 Laplacian Smoothing을 직접 지원하지 않으므로 voxel_down_sample로 밀도 정규화
            accumulated_pcd = accumulated_pcd.voxel_down_sample(self.voxel_size)
            
            print(f"[MultiView] View {i+1} fused. Accumulated points: {len(accumulated_pcd.points)}")
            
        return accumulated_pcd
