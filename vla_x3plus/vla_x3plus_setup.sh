#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/common_setup.sh"

# ── Main ───────────────────────────────────────────────────────
local_yml="$SCRIPT_DIR/vla_x3plus/environment.yml"
env_name=$(grep '^name:' "$local_yml" | awk '{print $2}')

log "━━━ vla_x3plus ━━━  env=${env_name}"

if [[ "$DRY_RUN" == true ]]; then
  log "  Would create env '$env_name' and run post-install steps"
  exit 0
fi

mkdir -p "$SCRIPT_DIR/vla_x3plus/third_party" "$SCRIPT_DIR/vla_x3plus/output"

setup_conda_env vla_x3plus

log "  Installing JAX 0.4.20 with CUDA 12..."
conda run -n "$env_name" pip install "jax[cuda12_pip]==0.4.20" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

log "  Installing MuJoCo + Gymnasium..."
conda run -n "$env_name" pip install mujoco
conda run -n "$env_name" pip install "gymnasium[mujoco]"

log "  Cloning and installing Octo VLA..."
OCTO_DIR="$SCRIPT_DIR/vla_x3plus/third_party/octo"
if [[ ! -d "$OCTO_DIR" ]]; then
  log "    Cloning octo-models/octo..."
  git clone https://github.com/octo-models/octo.git "$OCTO_DIR"
else
  ok "Octo already cloned, skipping"
fi
conda run -n "$env_name" pip install -e "$OCTO_DIR"
conda run -n "$env_name" pip install -r "$OCTO_DIR/requirements.txt"

log "  Installing imageio[ffmpeg] for video rendering..."
conda run -n "$env_name" pip install "imageio[ffmpeg]"

ok "vla_x3plus post-install complete"
