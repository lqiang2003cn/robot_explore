#!/usr/bin/env bash
set -euo pipefail

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Component order (explicit for dependency control) ──────────
ALL_COMPONENTS=(
  simulation
  ros2_stack
  meshroom
  gauss_splat
  vla_x3plus
)
DEFAULT_COMPONENTS=(ros2_stack vla_x3plus)

# ── Options ────────────────────────────────────────────────────
export CLEAN=false
PARALLEL=false
export DRY_RUN=false
export FORCE_REINSTALL=false
ALL=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [component ...]

Set up environments for robot_explore components.

Options:
  --all               Set up all components (simulation, ros2_stack, meshroom, gauss_splat, vla_x3plus)
  --clean             Remove and recreate environments from scratch
  --force-reinstall   Remove existing Miniconda and reinstall from scratch
  --parallel          Create environments in parallel (faster, noisier output)
  --dry-run           Show what would be done without executing
  -h, --help          Show this help message

By default only ros2_stack and vla_x3plus are set up.
Pass component names or --all to set up additional components.

Components: ${ALL_COMPONENTS[*]}

Examples:
  $(basename "$0")                     # set up ros2_stack + vla_x3plus
  $(basename "$0") --all               # set up everything
  $(basename "$0") simulation          # set up simulation only
  $(basename "$0") --clean simulation  # recreate simulation env
EOF
  exit 0
}

# ── Parse arguments ────────────────────────────────────────────
SELECTED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)     ALL=true;              shift ;;
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
elif [[ "$ALL" == true ]]; then
  COMPONENTS=("${ALL_COMPONENTS[@]}")
else
  COMPONENTS=("${DEFAULT_COMPONENTS[@]}")
fi

# ── Common setup (always runs) ─────────────────────────────────
source "$SCRIPT_DIR/common_setup.sh"

log "Components to set up: ${COMPONENTS[*]}"
[[ "$CLEAN" == true ]] && log "Mode: CLEAN (environments will be removed and recreated)"
[[ "$DRY_RUN" == true ]] && log "Mode: DRY RUN (no changes will be made)"
echo ""

# ── Run component setup scripts ───────────────────────────────
FAILED=()
SUCCEEDED=()

run_component() {
  local comp="$1"
  local script="$SCRIPT_DIR/$comp/${comp}_setup.sh"

  if [[ ! -f "$script" ]]; then
    echo -e "\033[0;31m  ✗ No setup script found: $script\033[0m"
    return 1
  fi

  bash "$script"
}

if [[ "$PARALLEL" == true && "$DRY_RUN" == false ]]; then
  log "Running in parallel mode..."
  pids=()
  for comp in "${COMPONENTS[@]}"; do
    run_component "$comp" &
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
    if run_component "$comp"; then
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
  if [[ "$comp" == "ros2_stack" || "$comp" == "meshroom" ]]; then
    echo "  source activate_env.sh $comp"
  elif [[ -f "$comp/environment.yml" ]]; then
    local_name=$(grep '^name:' "$comp/environment.yml" | awk '{print $2}')
    echo "  conda activate $local_name"
  fi
done
