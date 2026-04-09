import torch
import sys
import os

# Set PYTHONPATH
sys.path.append(os.getcwd())

print("Testing full pipeline imports...")

try:
    from src.seg.sam2_wrapper import SAM2BaseWrapper
    from src.topo.depth_wrapper import DepthAnythingV2Wrapper
    
    print("Initializing wrappers...")
    sam_w = SAM2BaseWrapper()
    dw = DepthAnythingV2Wrapper()
    
    print("Loading SAM2 model...")
    # sam_w.load_model() # This might be heavy, let's see if just import/init is enough
    
    print("Loading Depth-Anything-V2 model...")
    dw.load_model()
    
    print(f"Success! Depth model type: {type(dw.model)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
