#!/usr/bin/env bash
# Shared helpers and common system-level setup for robot_explore.
# Sourced by setup_envs.sh and individual component setup scripts.
set -euo pipefail

# Guard against double-sourcing
if [[ "${_COMMON_SETUP_LOADED:-}" == "1" ]]; then
  return 0 2>/dev/null || true
fi
_COMMON_SETUP_LOADED=1

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
CLEAN="${CLEAN:-false}"
DRY_RUN="${DRY_RUN:-false}"
FORCE_REINSTALL="${FORCE_REINSTALL:-false}"

# ── Colors ─────────────────────────────────────────────────────
if [[ -t 1 ]]; then
  GREEN='\033[0;32m'; RED='\033[0;31m'
  YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
else
  GREEN=''; RED=''; YELLOW=''; CYAN=''; NC=''
fi

log()  { echo -e "${CYAN}[robot_explore]${NC} $*"; }
ok()   { echo -e "${GREEN}  ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠ $*${NC}"; }
err()  { echo -e "${RED}  ✗ $*${NC}"; }

# ── Ensure conda is available (install Miniconda if missing) ──
install_miniconda() {
  local installer="Miniconda3-latest-Linux-x86_64.sh"
  local url="https://repo.anaconda.com/miniconda/$installer"
  local prefix="$HOME/miniconda3"

  log "conda not found — installing Miniconda to $prefix ..."
  curl -fsSL "$url" -o "/tmp/$installer"
  bash "/tmp/$installer" -b -p "$prefix"
  rm -f "/tmp/$installer"

  eval "$("$prefix/bin/conda" shell.bash hook)"
  conda init bash >/dev/null 2>&1 || true
  ok "Miniconda installed at $prefix"
}

ensure_conda() {
  if [[ "$FORCE_REINSTALL" == true && -d "$HOME/miniconda3" ]]; then
    log "Removing existing Miniconda at $HOME/miniconda3 ..."
    rm -rf "$HOME/miniconda3"
    hash -r 2>/dev/null || true
  fi

  if ! command -v conda &>/dev/null; then
    if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
      eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    else
      install_miniconda
    fi
  fi

  eval "$(conda shell.bash hook)"

  # Miniconda 26.x ships with 'defaults' channels that require interactive
  # ToS acceptance. Write a clean config using only conda-forge.
  find "$(conda info --base)" -name ".condarc" -o -name "condarc" 2>/dev/null | xargs rm -f 2>/dev/null || true
  cat > "$HOME/.condarc" << 'CONDARC'
channels:
  - conda-forge
default_channels: []
auto_activate_base: false
CONDARC

  log "conda found: $(conda --version)"
}

# ── GitHub CLI ─────────────────────────────────────────────────
install_gh_cli() {
  if command -v gh &>/dev/null; then
    ok "GitHub CLI already installed ($(gh --version | head -1))"
    return 0
  fi

  log "Installing GitHub CLI..."
  if [[ "$DRY_RUN" == true ]]; then
    log "  Would install gh via apt"
    return 0
  fi

  if apt-get update -qq && apt-get install -y -qq gh; then
    ok "GitHub CLI installed ($(gh --version | head -1))"
  else
    err "Failed to install GitHub CLI"
    return 1
  fi
}

# ── Generic conda env create / update ─────────────────────────
# Usage: setup_conda_env <component>
# Expects <component>/environment.yml to exist.
# Does NOT run post-install hooks — callers handle that inline.
setup_conda_env() {
  local comp="$1"
  local yml="$SCRIPT_DIR/$comp/environment.yml"

  if [[ ! -f "$yml" ]]; then
    warn "Skipping $comp — no environment.yml found"
    return 1
  fi

  local env_name
  env_name=$(grep '^name:' "$yml" | awk '{print $2}')
  local py_ver
  py_ver=$(grep 'python=' "$yml" | head -1 | sed 's/.*python=//' | tr -d ' ')

  log "━━━ ${comp} ━━━  env=${env_name}  python=${py_ver}"

  if [[ "$DRY_RUN" == true ]]; then
    log "  Would create/update env '$env_name' from $yml"
    return 0
  fi

  local env_exists=false
  if conda env list | grep -qw "$env_name"; then
    env_exists=true
  fi

  if [[ "$CLEAN" == true && "$env_exists" == true ]]; then
    log "  Removing existing env '$env_name'..."
    conda env remove -n "$env_name" -y
    env_exists=false
  fi

  if [[ "$env_exists" == true ]]; then
    log "  Updating existing env '$env_name'..."
    if conda env update -f "$yml" --prune; then
      ok "$env_name updated"
    else
      err "$env_name update FAILED"
      return 1
    fi
  else
    log "  Creating env '$env_name'..."
    if conda env create -f "$yml"; then
      ok "$env_name created"
    else
      err "$env_name creation FAILED"
      return 1
    fi
  fi

  return 0
}

# ── Run common setup steps ─────────────────────────────────────
ensure_conda
install_gh_cli
echo ""
