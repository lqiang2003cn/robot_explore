#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/common_setup.sh"

# ── System-level dependencies for Isaac Sim RTX renderer ──────
install_system_deps() {
  local pkgs=(
    libglu1-mesa   # libGLU.so.1  — required by neuray (MDL-SDK / RTX renderer)
    libxt6          # libXt.so.6   — required by MaterialX rendering
  )

  local missing=()
  for pkg in "${pkgs[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "System dependencies already installed"
    return 0
  fi

  log "Installing system packages: ${missing[*]}"
  if [[ "$DRY_RUN" == true ]]; then
    log "  Would run: apt-get install -y ${missing[*]}"
    return 0
  fi

  if apt-get update -qq && apt-get install -y -qq "${missing[@]}"; then
    ok "System dependencies installed"
  else
    err "Failed to install system packages (are you root / do you have sudo?)"
    return 1
  fi
}

# ── Main ───────────────────────────────────────────────────────
install_system_deps

local_yml="$SCRIPT_DIR/simulation/environment.yml"
env_name=$(grep '^name:' "$local_yml" | awk '{print $2}')

log "━━━ simulation ━━━  env=${env_name}"

if [[ "$DRY_RUN" == true ]]; then
  log "  Would create env '$env_name' and install Isaac Sim 6.0.0"
  exit 0
fi

setup_conda_env simulation

log "  Installing Isaac Sim 6.0.0..."
export OMNI_KIT_ACCEPT_EULA=YES
if conda run -n "$env_name" pip install "isaacsim[all,extscache]==6.0.0" \
    --extra-index-url https://pypi.nvidia.com \
  && conda run -n "$env_name" pip install "imageio[ffmpeg]"; then
  ok "Isaac Sim 6.0.0 installed"
else
  err "Isaac Sim installation FAILED"
  exit 1
fi
