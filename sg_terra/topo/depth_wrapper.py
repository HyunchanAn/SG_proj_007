import torch
import numpy as np
import cv2
import time
from typing import Optional, Any

class DepthAnythingV2Wrapper:
    def __init__(self, encoder: str = 'vitl', 
                 checkpoint_path: str = "../../models/depth_anything_v2/depth_anything_v2_vitl.pth", 
                 device: Optional[str] = None):
        """
        Depth-Anything-V2 ê¸°ë°˜ 3D Topography ?¬êµ¬???Œì´?„ë¼??
        :param encoder: ëª¨ë¸ ?¸ì½”???¬ê¸° ('vits', 'vitb', 'vitl'). ê¸°ë³¸ê°’ì? Large.
        :param checkpoint_path: ê°€ì¤‘ì¹˜ ?Œì¼ ê²½ë¡œ
        :param device: ?°ì‚° ?”ë°”?´ìŠ¤ (cuda, mps, cpu ?ë™ ? ë‹¹)
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
            
        print(f"Initializing Depth-Anything-V2 Wrapper on: {self.device}")
        self.encoder = encoder
        self.checkpoint_path = checkpoint_path
        self.model: Any = None
        
        # PyTorch FP16/BF16 Mixed Precision ì¶”ë¡  ìµœì ??(NVIDIA GPU ?œì •)
        if self.device.type == "cuda":
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8: # Ampere ?„í‚¤?ì²˜ ?´ìƒ TF32 ?ˆìš©
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def load_model(self):
        """
        ëª¨ë¸ ?Œë¼ë¯¸í„°ë¥?ë©”ëª¨ë¦¬ì— ?ì¬.
        """
        from sg_terra.topo.depth_anything_v2.dpt import DepthAnythingV2
        
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }
        print("Loading Depth-Anything-V2 checkpoints...")
        self.model = DepthAnythingV2(**model_configs[self.encoder])
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        print("Depth-Anything-V2 loaded successfully.")

    def estimate_depth(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        ?¨ì•ˆ ?´ë?ì§€?ì„œ ?´ìƒ???’ì? Depth Map ì¶”ì •.
        SAM ë§ˆìŠ¤?¬ë? ?œìš©?˜ì—¬ ROI ?ì—­??ì§‘ì¤‘, Latency?€ ?¸ì´ì¦?ìµœì ??
        :param image: HxWxC ?¬ë§·??numpy ?´ë?ì§€ (RGB)
        :param mask: SAM 2?ì„œ ì¶”ì¶œ??ë¶ˆë¦¬??2D ë§ˆìŠ¤??
        :return: ?½ì?ë³??ë???ê¹Šì´ê°?ë°°ì—´ (Z-axis)
        """
        if self.model is None:
            self.load_model()
            
        # SAM ë§ˆìŠ¤?¬ê? ?œê³µ??ê²½ìš° ë°°ê²½ ?½ì? ?´ë Œì§•ìœ¼ë¡??¸ì´ì¦?ìµœì†Œ??
        if mask is not None:
            mask_bool = np.array(mask, dtype=bool).squeeze()
            proc_img = image.copy()
            proc_img[~mask_bool] = 0 
        else:
            proc_img = image
            
        # Real inference
        depth = self.model.infer_image(proc_img)
        return depth

# ?ŒìŠ¤??ë¸”ëŸ­ (ì§ì ‘ ?¤í–‰ ??
if __name__ == "__main__":
    wrapper = DepthAnythingV2Wrapper()
    wrapper.load_model()
    
    # 1080p ê°€???´ë?ì§€ë¡??ŒìŠ¤??
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_mask = np.ones((1080, 1920), dtype=bool) 
    # ?½ê°„???¬ë°± ?ì„±
    dummy_mask[:100, :] = False
    dummy_mask[-100:, :] = False
    
    start = time.time()
    depth_map = wrapper.estimate_depth(dummy_img, mask=dummy_mask)
    end = time.time()
    
    print(f"Mock Output Depth Map Shape: {depth_map.shape}")
    print(f"Inference Latency: {(end - start)*1000:.2f} ms")
