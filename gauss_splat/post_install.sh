#!/usr/bin/env bash
set -euo pipefail

# PyTorch with CUDA 12.1 (install before nerfstudio so it picks up GPU support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# nerfstudio pulls in gsplat, tinycudann bindings, and other deps
pip install nerfstudio

# Grounding DINO (via HuggingFace transformers) + SAM2 for object segmentation
pip install transformers
pip install sam2
