#!/bin/bash

# SG-TERRA Model Weights Download Script
# This script downloads the pre-trained weights for SAM 2 and Depth-Anything-V2.

echo "=========================================================="
echo "Downloading Model Weights for SG-TERRA"
echo "=========================================================="

# Create directories if they don't exist
mkdir -p models/sam2
mkdir -p models/depth_anything_v2

# 1. SAM 2 Checkpoints
# Using the Hiera-Large model as it provides the best performance for RTX 5080/M2 Pro
echo "1. Downloading SAM 2 (Hiera-Large)..."
curl -L -o models/sam2/sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# 2. Depth-Anything-V2 Checkpoints
# Using the ViT-Large model to ensure high-resolution depth extraction
echo "2. Downloading Depth-Anything-V2 (ViT-Large)..."
curl -L -o models/depth_anything_v2/depth_anything_v2_vitl.pth https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

echo "=========================================================="
echo "Download Completed!"
echo "Model weights are stored in 'models/sam2/' and 'models/depth_anything_v2/'."
