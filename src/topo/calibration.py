import numpy as np

class ScaleCalibrator:
    def __init__(self):
        """
        이미지 내 픽셀 좌표와 실제 실측 거리를 매핑하여 
        3D 공간의 스케일(Pixels to mm)을 환산하는 모듈.
        """
        self.pixel_distance = None
        self.real_mm = None
        self.scale_factor = None  # mm/pixel

    def calibrate(self, p1: tuple, p2: tuple, real_mm: float):
        """
        두 점 사이의 픽셀 거리와 실제 거리를 기반으로 스케일 팩터 계산.
        :param p1: (x, y) 픽셀 좌표
        :param p2: (x, y) 픽셀 좌표
        :param real_mm: 실측 거리 (mm)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        self.pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if self.pixel_distance == 0:
            raise ValueError("두 점이 동일합니다. 보정을 위해 서로 다른 두 지점을 선택해 주세요.")
            
        self.real_mm = real_mm
        self.scale_factor = self.real_mm / self.pixel_distance
        
        print(f"Calibration Success: 1 pixel around {self.scale_factor:.4f} mm")
        return self.scale_factor

    def apply_scale(self, point_cloud_np: np.ndarray) -> np.ndarray:
        """
        Numpy 배열 형태의 포인트 클라우드(N, 3)에 스케일 적용.
        """
        if self.scale_factor is None:
            return point_cloud_np
            
        return point_cloud_np * self.scale_factor

    def pixel_to_mm(self, pixels: float) -> float:
        """픽셀 값을 mm 단위로 환산."""
        if self.scale_factor is None:
            return pixels
        return pixels * self.scale_factor

# 간단한 검증 테스트
if __name__ == "__main__":
    calibrator = ScaleCalibrator()
    # 100 픽셀이 10mm인 경우 (1px = 0.1mm)
    factor = calibrator.calibrate((0, 0), (100, 0), 10.0)
    assert abs(factor - 0.1) < 1e-6
    print("Calibration logic test passed.")
