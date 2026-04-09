import torch
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

try:
    from src.topo.depth_anything_v2.dpt import DepthAnythingV2
    print(f"Import Success: {DepthAnythingV2}")
    
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    model = DepthAnythingV2(**model_configs['vitl'])
    print(f"Model created: {type(model)}")
    print(f"Has load_state_dict: {hasattr(model, 'load_state_dict')}")
    
    # Try omegaconf check
    import omegaconf
    print(f"Is model DictConfig? {isinstance(model, omegaconf.DictConfig)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
