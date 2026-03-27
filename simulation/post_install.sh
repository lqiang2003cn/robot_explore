#!/usr/bin/env bash
set -euo pipefail

export OMNI_KIT_ACCEPT_EULA=YES

echo "Installing Isaac Sim 6.0.0 from NVIDIA PyPI..."
pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com
pip install "imageio[ffmpeg]"
