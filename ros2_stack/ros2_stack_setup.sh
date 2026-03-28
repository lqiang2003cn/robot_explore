#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/common_setup.sh"

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

    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
      > /etc/apt/sources.list.d/ros2.list

    apt-get update -qq
    apt-get install -y -qq ros-jazzy-desktop

    ok "ROS 2 Jazzy base installed"
  fi

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

# ── Build colcon workspace ────────────────────────────────────
setup_ros2_workspace() {
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

  # Isolate from Miniconda so colcon/CMake use the system Python
  local _orig_path="$PATH"
  PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)
  hash -r
  conda deactivate 2>/dev/null || true
  local _saved_conda_prefix="${CONDA_PREFIX:-}"
  local _saved_conda_exe="${CONDA_EXE:-}"
  local _saved_conda_python="${CONDA_PYTHON_EXE:-}"
  unset CONDA_PREFIX CONDA_EXE CONDA_PYTHON_EXE CONDA_DEFAULT_ENV

  log "  Building colcon workspace..."
  cd "$ws_dir"
  if colcon build --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3; then
    ok "ros2_stack workspace built"
  else
    err "ros2_stack build FAILED"
    PATH="$_orig_path"; hash -r
    [[ -n "$_saved_conda_prefix" ]] && export CONDA_PREFIX="$_saved_conda_prefix"
    [[ -n "$_saved_conda_exe" ]] && export CONDA_EXE="$_saved_conda_exe"
    [[ -n "$_saved_conda_python" ]] && export CONDA_PYTHON_EXE="$_saved_conda_python"
    exit 1
  fi

  PATH="$_orig_path"; hash -r
  [[ -n "$_saved_conda_prefix" ]] && export CONDA_PREFIX="$_saved_conda_prefix"
  [[ -n "$_saved_conda_exe" ]] && export CONDA_EXE="$_saved_conda_exe"
  [[ -n "$_saved_conda_python" ]] && export CONDA_PYTHON_EXE="$_saved_conda_python"
  cd "$SCRIPT_DIR"
}

# ── Main ───────────────────────────────────────────────────────
install_ros2_jazzy
echo ""
setup_ros2_workspace
