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

    def create_point_cloud(self, rgb, depth, scale_factor=1.0, fx=1000, fy=1000, cx=None, cy=None):
        """RGB와 Depth 맵을 기반으로 Open3D Point Cloud 객체 생성 (실제 스케일 반영)"""
        h, w = depth.shape
        if cx is None: cx = w / 2
        if cy is None: cy = h / 2
        
        # Intrinsic Matrix
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        
        # 8비트 RGB
        rgb_o3d = o3d.geometry.Image(rgb)
        
        # Depth-Anything의 상대적 깊이를 물리적 mm 단위로 정규화 및 스케일링
        # (상대 깊이 0.0~1.0 가정 -> scale_factor를 곱해 실제 mm로 변환)
        depth_scaled = depth.astype(np.float32) * scale_factor * 10.0 # 10.0은 기본 뎁스 강조 계수
        depth_o3d = o3d.geometry.Image(depth_scaled)
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 2D 평면 스케일 보정 (X, Y축에 scale_factor 적용)
        points = np.asarray(pcd.points)
        points[:, 0] *= scale_factor
        points[:, 1] *= scale_factor
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 다운샘플링 및 노이즈 제거
        pcd = pcd.voxel_down_sample(self.voxel_size)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd.select_by_index(ind)

    def extract_fpfh(self, pcd):
        """FPFH 추출"""
        radius_normal = self.voxel_size * 2
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = self.voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return fpfh

    def register_icp(self, source, target):
        """전역 정합 후 ICP 수행 (30도 이내 각도 최적화)"""
        source_fpfh = self.extract_fpfh(source)
        target_fpfh = self.extract_fpfh(target)
        
        # 30도 이내 환경이므로 정합 임계치를 다소 타이트하게 설정하여 노이즈 방지
        distance_threshold = self.voxel_size * 2.0
        
        # 1. RANSAC 전역 정합
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        # 2. 로컬 ICP 정밀 보정 (Point-to-Plane)
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold / 2, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result_icp.transformation

    def fuse_views(self, rgb_list, depth_list, scale_factor=1.0):
        """다중 뷰 이미지 리스트와 깊이 리스트를 통째로 정합 (스케일 반영)"""
        if len(rgb_list) < 1:
            raise ValueError("최소 1개 이상의 이미지가 필요합니다.")
            
        print(f"[MultiView] Calibration Scale(factor={scale_factor:.4f}) applying...")
        accumulated_pcd = self.create_point_cloud(rgb_list[0], depth_list[0], scale_factor=scale_factor)
        accumulated_pcd.estimate_normals()
        
        # 뷰가 1개뿐이면 정합 과정을 건너뛰고 바로 반환
        if len(rgb_list) == 1:
            print("[MultiView] 단일 뷰 스케일 보정 완료.")
            return accumulated_pcd
        
        for i in range(1, len(rgb_list)):
            print(f"[MultiView] {i+1}th view refined matching (ICP)...")
            source_pcd = self.create_point_cloud(rgb_list[i], depth_list[i], scale_factor=scale_factor)
            source_pcd.estimate_normals()
            
            # 정합 진행
            transform_mat = self.register_icp(source_pcd, accumulated_pcd)
            source_pcd.transform(transform_mat)
            
            # 합체 및 다운샘플링
            accumulated_pcd += source_pcd
            accumulated_pcd = accumulated_pcd.voxel_down_sample(self.voxel_size)
            print(f"[MultiView] View {i+1} fused. Accumulated points: {len(accumulated_pcd.points)}")
            
        return accumulated_pcd
