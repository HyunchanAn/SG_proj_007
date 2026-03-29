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

    def create_point_cloud(self, rgb, depth, fx=1000, fy=1000, cx=None, cy=None):
        """RGB와 Depth 맵을 기반으로 Open3D Point Cloud 객체 생성"""
        h, w = depth.shape
        if cx is None: cx = w / 2
        if cy is None: cy = h / 2
        
        # Intrinsic Matrix (가변 카메라 고려)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        
        # 8비트 RGB를 맞게 변환
        rgb_o3d = o3d.geometry.Image(rgb)
        
        # Float32 Depth를 미터법(Metric) 스케일로 변환 가정
        depth_norm = cv2.normalize(depth, None, 0, 10, cv2.NORM_MINMAX)
        depth_o3d = o3d.geometry.Image(depth_norm.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
        )
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 다운샘플링 수행 및 노이즈 제거
        pcd = pcd.voxel_down_sample(self.voxel_size)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd.select_by_index(ind)

    def extract_fpfh(self, pcd):
        """빠른 전역 정합(Fast Global Registration)을 위한 FPFH 추출"""
        radius_normal = self.voxel_size * 2
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        radius_feature = self.voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return fpfh

    def register_icp(self, source, target):
        """전역 정합 후 Iterative Closest Point(ICP) 수행"""
        # FPFH 특징 추출
        source_fpfh = self.extract_fpfh(source)
        target_fpfh = self.extract_fpfh(target)
        
        distance_threshold = self.voxel_size * 1.5
        
        # 1. RANSAC 전역 정합
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        # 2. 로컬 ICP 정밀 보정
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold / 2, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        return result_icp.transformation

    def fuse_views(self, rgb_list, depth_list):
        """다중 뷰 이미지 리스트와 깊이 리스트를 통째로 정합(ICP Iteration)"""
        if len(rgb_list) < 2:
            raise ValueError("최소 2개 이상의 뷰가 필요합니다.")
            
        print("[MultiView] 첫 번째 타겟 뷰 초기화...")
        accumulated_pcd = self.create_point_cloud(rgb_list[0], depth_list[0])
        accumulated_pcd.estimate_normals()
        
        results = [accumulated_pcd]
        
        for i in range(1, len(rgb_list)):
            print(f"[MultiView] {i+1}번째 뷰 정합 계산 중 (ICP)...")
            source_pcd = self.create_point_cloud(rgb_list[i], depth_list[i])
            source_pcd.estimate_normals()
            
            # 소스와 누적본 정합 진행
            transform_mat = self.register_icp(source_pcd, accumulated_pcd)
            source_pcd.transform(transform_mat)
            
            # 합체
            accumulated_pcd += source_pcd
            
            # Voxel 필터로 균일화 처리
            accumulated_pcd = accumulated_pcd.voxel_down_sample(self.voxel_size)
            print(f"[MultiView] 뷰 {i+1} 병합 완료. 누적 포인트 갯수: {len(accumulated_pcd.points)}개")
            
        print("[MultiView] 전역 병합 최종 Smoothing 완료.")
        return accumulated_pcd
