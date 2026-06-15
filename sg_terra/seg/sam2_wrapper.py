import torch
import numpy as np
import cv2
import time
from typing import Optional, Any

class SAM2BaseWrapper:
    def __init__(self, model_cfg: str = "sam2_hiera_l.yaml", 
                 checkpoint_path: str = "../../models/sam2/sam2_hiera_large.pt", 
                 device: Optional[str] = None):
        """
        SAM 2 기반 Target Segmentation ?�이?�라??구성.
        :param model_cfg: SAM 2 모델 구조 ?�정 ?�일 (기본: Hiera-Large)
        :param checkpoint_path: 가중치 ?�일 경로
        :param device: ?�산 ?�바?�스 (cuda, mps, cpu ?�동 ?�당)
        """
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Initializing SAM 2 Wrapper on: {self.device}")
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        
        # 모델 빌드 �?로드 로직 (?�제 sam2 ?�키지가 ?�치????매핑)
        self.model: Any = None
        self.predictor: Any = None
        
        # PyTorch FP16/BF16 Mixed Precision 추론 최적??(NVIDIA GPU ?�정)
        if self.device.type == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8: # Ampere ?�키?�처 ?�상 (RTX 5080 ?? TF32 ?�용
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def load_model(self):
        """
        모델 ?�라미터�?메모리에 ?�재 (추후 지??로딩???�해 분리)
        """
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        print("Loading SAM 2 checkpoints...")
        self.model = build_sam2(self.model_cfg, self.checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print("SAM 2 loaded successfully.")

    def segment_target(self, image: np.ndarray, prompt_points: Optional[np.ndarray] = None, prompt_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ?�력 ?��?지?�서 강판(Target)만을 Segmentation 처리.
        :param image: HxWxC ?�맷??numpy ?��?지 (RGB)
        :param prompt_points: Nx2 ?�태???��? 좌표 배열
        :param prompt_labels: N ?�태???�이�?(1: Target, 0: Background)
        :return: 마스????boolean 2D 배열
        """
        if self.predictor is None:
            self.load_model()
            
        self.predictor.set_image(image)  # type: ignore
        
        if prompt_points is None or prompt_labels is None:
            # Default to center point if no prompts provided
            h, w = image.shape[:2]
            prompt_points = np.array([[w//2, h//2]])
            prompt_labels = np.array([1])
            
        masks, scores, logits = self.predictor.predict(  # type: ignore
            point_coords=prompt_points,
            point_labels=prompt_labels,
            multimask_output=False
        )
        
        return masks[0].astype(bool)

# ?�스??블럭 (직접 ?�행 ??
if __name__ == "__main__":
    wrapper = SAM2BaseWrapper()
    wrapper.load_model()
    
    # 1080p 가???��?지�?Latency ?�스??
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    start = time.time()
    mask = wrapper.segment_target(dummy_img)
    end = time.time()
    
    print(f"Mock Output Mask Shape: {mask.shape}")
    print(f"Inference Latency: {(end - start)*1000:.2f} ms")
