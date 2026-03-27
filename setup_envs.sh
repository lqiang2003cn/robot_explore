#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Component order (explicit for dependency control) ──────────
COMPONENTS=(
  simulation
  ros2_stack
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

# Miniconda 26.x ships with 'defaults' channels (pkgs/main, pkgs/r) that
# require interactive ToS acceptance. Purge all condarc files and write a
# clean one that uses only conda-forge.
find "$(conda info --base)" -name ".condarc" -o -name "condarc" 2>/dev/null | xargs rm -f 2>/dev/null || true
cat > "$HOME/.condarc" << 'CONDARC'
channels:
  - conda-forge
default_channels: []
auto_activate_base: false
CONDARC

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

# ── ROS 2 Jazzy + robotics stack ──────────────────────────────
install_ros2_jazzy() {
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    ok "ROS 2 Jazzy already installed"
  else
    log "Installing ROS 2 Jazzy and robotics stack..."

    if [[ "$DRY_RUN" == true ]]; then
      log "  Would install ROS 2 Jazzy + ros2_control + MoveIt2 + BehaviorTree.CPP"
      return 0
    fi

    apt-get update -qq
    apt-get install -y -qq software-properties-common curl

    # ROS 2 GPG key + repository
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
      > /etc/apt/sources.list.d/ros2.list

    apt-get update -qq

    # Core ROS 2 desktop
    apt-get install -y -qq ros-jazzy-desktop

    ok "ROS 2 Jazzy base installed"
  fi

  # Robotics stack packages
  local stack_pkgs=(
    ros-jazzy-ros2-control
    ros-jazzy-ros2-controllers
    ros-jazzy-moveit
    ros-jazzy-moveit-py
    ros-jazzy-moveit-resources-panda-moveit-config
    ros-jazzy-moveit-resources-panda-description
    ros-jazzy-behaviortree-cpp
    ros-jazzy-simulation-interfaces
    ros-jazzy-ros-testing
    python3-colcon-common-extensions
  )

  local missing=()
  for pkg in "${stack_pkgs[@]}"; do
    if ! dpkg -s "$pkg" &>/dev/null; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    ok "ROS 2 robotics stack already installed"
  elif [[ "$DRY_RUN" == true ]]; then
    log "  Would install: ${missing[*]}"
  else
    log "Installing ROS 2 stack packages: ${missing[*]}"
    apt-get install -y -qq "${missing[@]}"
    ok "ROS 2 robotics stack installed"
  fi

  # py_trees for Python BehaviorTree orchestration
  if python3 -c "import py_trees" &>/dev/null; then
    ok "py_trees already installed"
  elif [[ "$DRY_RUN" == true ]]; then
    log "  Would pip-install py_trees"
  else
    log "Installing py_trees..."
    pip install --break-system-packages py_trees
    ok "py_trees installed"
  fi
}

# Only install ROS 2 when the ros2_stack component is requested
for _comp in "${COMPONENTS[@]}"; do
  if [[ "$_comp" == "ros2_stack" ]]; then
    install_ros2_jazzy
    break
  fi
done
unset _comp
echo ""

# ── Build ros2_stack colcon workspace ─────────────────────────
setup_ros2_stack() {
  log "━━━ ros2_stack ━━━  (native, no conda env)"

  if [[ "$DRY_RUN" == true ]]; then
    log "  Would build ros2_stack colcon workspace"
    return 0
  fi

  # ROS 2 setup scripts reference variables that may be unset
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u

  local ws_dir="$SCRIPT_DIR/ros2_stack/ws"
  local src_dir="$ws_dir/src"
  mkdir -p "$src_dir"

  # Clone topic_based_ros2_control if not present
  if [[ ! -d "$src_dir/topic_based_ros2_control" ]]; then
    log "  Cloning topic_based_ros2_control..."
    git clone https://github.com/PickNikRobotics/topic_based_ros2_control.git \
      "$src_dir/topic_based_ros2_control"
    ok "topic_based_ros2_control cloned"
  else
    ok "topic_based_ros2_control already present"
  fi

  if [[ "$CLEAN" == true ]]; then
    log "  Cleaning previous build..."
    rm -rf "$ws_dir/build" "$ws_dir/install" "$ws_dir/log"
  fi

  log "  Building colcon workspace..."
  cd "$ws_dir"
  if colcon build; then
    ok "ros2_stack workspace built"
  else
    err "ros2_stack build FAILED"
    return 1
  fi

  cd "$SCRIPT_DIR"
  return 0
}

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
    if [[ "$comp" == "ros2_stack" ]]; then
      setup_ros2_stack &
    else
      setup_env "$comp" &
    fi
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
    if [[ "$comp" == "ros2_stack" ]]; then
      if setup_ros2_stack; then
        SUCCEEDED+=("$comp")
      else
        FAILED+=("$comp")
      fi
    else
      if setup_env "$comp"; then
        SUCCEEDED+=("$comp")
      else
        FAILED+=("$comp")
      fi
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
  if [[ "$comp" == "ros2_stack" ]]; then
    echo "  source activate_env.sh ros2_stack"
  elif [[ -f "$comp/environment.yml" ]]; then
    local_name=$(grep '^name:' "$comp/environment.yml" | awk '{print $2}')
    echo "  conda activate $local_name"
  fi
done
