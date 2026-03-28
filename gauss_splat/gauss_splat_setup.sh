#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/common_setup.sh"

# ── COLMAP (required for SfM) ─────────────────────────────────
install_colmap() {
  if command -v colmap &>/dev/null; then
    ok "COLMAP already installed ($(colmap -h 2>&1 | head -1 || echo 'found'))"
    return 0
  fi

  log "Installing COLMAP..."
  if [[ "$DRY_RUN" == true ]]; then
    log "  Would install colmap via apt"
    return 0
  fi

  if apt-get update -qq && apt-get install -y -qq colmap; then
    ok "COLMAP installed"
  else
    err "Failed to install COLMAP"
    return 1
  fi
}

# ── Main ───────────────────────────────────────────────────────
install_colmap
echo ""

local_yml="$SCRIPT_DIR/gauss_splat/environment.yml"
env_name=$(grep '^name:' "$local_yml" | awk '{print $2}')

log "━━━ gauss_splat ━━━  env=${env_name}"

if [[ "$DRY_RUN" == true ]]; then
  log "  Would create env '$env_name' and run post-install steps"
  exit 0
fi

mkdir -p "$SCRIPT_DIR/gauss_splat/input" "$SCRIPT_DIR/gauss_splat/output"

setup_conda_env gauss_splat

log "  Installing PyTorch with CUDA 12.1..."
conda run -n "$env_name" pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121

log "  Installing nerfstudio..."
conda run -n "$env_name" pip install nerfstudio

log "  Installing Grounding DINO (transformers) + SAM2..."
conda run -n "$env_name" pip install transformers
conda run -n "$env_name" pip install sam2

ok "gauss_splat post-install complete"
