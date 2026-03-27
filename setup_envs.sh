#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Component order (explicit for dependency control) ──────────
COMPONENTS=(
  simulation
)

# ── Options ────────────────────────────────────────────────────
CLEAN=false
PARALLEL=false
DRY_RUN=false
FORCE_REINSTALL=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [component ...]

Set up conda environments for robot_explore components.

Options:
  --clean             Remove and recreate environments from scratch
  --force-reinstall   Remove existing Miniconda and reinstall from scratch
  --parallel          Create environments in parallel (faster, noisier output)
  --dry-run           Show what would be done without executing
  -h, --help          Show this help message

If component names are given, only those environments are set up.
Otherwise all components are set up.

Examples:
  $(basename "$0")                     # set up everything
  $(basename "$0") --clean simulation  # recreate simulation env
EOF
  exit 0
}

# ── Parse arguments ────────────────────────────────────────────
SELECTED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)   CLEAN=true;            shift ;;
    --force-reinstall) FORCE_REINSTALL=true; shift ;;
    --parallel) PARALLEL=true;        shift ;;
    --dry-run) DRY_RUN=true;          shift ;;
    -h|--help) usage ;;
    -*) echo "Unknown option: $1"; usage ;;
    *)  SELECTED+=("$1");     shift ;;
  esac
done

if [[ ${#SELECTED[@]} -gt 0 ]]; then
  COMPONENTS=("${SELECTED[@]}")
fi

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

  # Make conda available in this session
  eval "$("$prefix/bin/conda" shell.bash hook)"
  conda init bash >/dev/null 2>&1 || true
  ok "Miniconda installed at $prefix"
}

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

# Miniconda 26.x ships with 'defaults' channels that require ToS acceptance.
# Remove them and use only conda-forge.
conda config --remove channels defaults 2>/dev/null || true
conda config --add channels conda-forge 2>/dev/null || true

log "conda found: $(conda --version)"
log "Components to set up: ${COMPONENTS[*]}"
[[ "$CLEAN" == true ]] && log "Mode: CLEAN (environments will be removed and recreated)"
[[ "$DRY_RUN" == true ]] && log "Mode: DRY RUN (no changes will be made)"
echo ""

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

install_gh_cli
echo ""

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

# Only install system deps when the simulation component is requested
for _comp in "${COMPONENTS[@]}"; do
  if [[ "$_comp" == "simulation" ]]; then
    install_system_deps
    break
  fi
done
unset _comp
echo ""

# ── Setup function ─────────────────────────────────────────────
setup_env() {
  local comp="$1"
  local yml="$comp/environment.yml"

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

  # Post-install hook
  if [[ -f "$comp/post_install.sh" ]]; then
    log "  Running post_install.sh..."
    if ! conda run -n "$env_name" bash "$comp/post_install.sh"; then
      err "$env_name post_install FAILED"
      return 1
    fi
    ok "$env_name post_install done"
  fi

  return 0
}

# ── Main loop ──────────────────────────────────────────────────
FAILED=()
SUCCEEDED=()

if [[ "$PARALLEL" == true && "$DRY_RUN" == false ]]; then
  log "Running in parallel mode..."
  pids=()
  for comp in "${COMPONENTS[@]}"; do
    setup_env "$comp" &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      SUCCEEDED+=("${COMPONENTS[$i]}")
    else
      FAILED+=("${COMPONENTS[$i]}")
    fi
  done
else
  for comp in "${COMPONENTS[@]}"; do
    if setup_env "$comp"; then
      SUCCEEDED+=("$comp")
    else
      FAILED+=("$comp")
    fi
  done
fi

# ── Summary ────────────────────────────────────────────────────
echo ""
log "━━━ Summary ━━━"

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
  ok "Succeeded: ${SUCCEEDED[*]}"
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  err "Failed: ${FAILED[*]}"
  echo ""
  log "To retry a failed component:  ./setup_envs.sh ${FAILED[0]}"
  exit 1
fi

echo ""
log "All environments ready. Activate with:"
for comp in "${SUCCEEDED[@]}"; do
  local_name=$(grep '^name:' "$comp/environment.yml" | awk '{print $2}')
  echo "  conda activate $local_name"
done
