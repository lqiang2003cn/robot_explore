#!/usr/bin/env bash
# Convenience wrapper: source this to activate a component's env and cd into it.
#
# Usage:  source activate_env.sh <component>
# Example: source activate_env.sh simulation

if [[ $# -lt 1 ]]; then
  echo "Usage: source activate_env.sh <component>"
  echo "Available: simulation, ros2_stack, meshroom, gauss_splat, vla_x3plus"
  return 1 2>/dev/null || exit 1
fi

COMP="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ros2_stack uses native ROS 2, not conda
if [[ "$COMP" == "ros2_stack" ]]; then
  if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
    echo "Error: ROS 2 Jazzy not found. Run ./setup_envs.sh ros2_stack first."
    return 1 2>/dev/null || exit 1
  fi
  source /opt/ros/jazzy/setup.bash
  local_ws="$SCRIPT_DIR/ros2_stack/ws/install/setup.bash"
  if [[ -f "$local_ws" ]]; then
    source "$local_ws"
  fi
  cd "$SCRIPT_DIR/ros2_stack"
  echo "Sourced ROS 2 Jazzy + workspace — now in $(pwd)"
  return 0 2>/dev/null || exit 0
fi

# meshroom uses a prebuilt binary, not conda
if [[ "$COMP" == "meshroom" ]]; then
  MESHROOM_VERSION=$(grep '^version:' "$SCRIPT_DIR/meshroom/config.yml" 2>/dev/null | awk '{print $2}' | tr -d '"'"'")
  if [[ -z "$MESHROOM_VERSION" ]]; then
    echo "Error: meshroom/config.yml must contain 'version: X.Y.Z'"
    return 1 2>/dev/null || exit 1
  fi
  MESHROOM_DIR="$SCRIPT_DIR/meshroom/Meshroom-${MESHROOM_VERSION}"
  if [[ ! -d "$MESHROOM_DIR" ]]; then
    echo "Error: Meshroom not found at $MESHROOM_DIR. Run ./setup_envs.sh meshroom first."
    return 1 2>/dev/null || exit 1
  fi
  export PATH="$MESHROOM_DIR:$PATH"
  export ALICEVISION_SENSOR_DB="$MESHROOM_DIR/aliceVision/share/aliceVision/cameraSensors.db"
  cd "$SCRIPT_DIR/meshroom"
  echo "Meshroom ${MESHROOM_VERSION} on PATH — now in $(pwd)"
  return 0 2>/dev/null || exit 0
fi

YML="$SCRIPT_DIR/$COMP/environment.yml"

if [[ ! -f "$YML" ]]; then
  echo "Error: $YML not found"
  return 1 2>/dev/null || exit 1
fi

ENV_NAME=$(grep '^name:' "$YML" | awk '{print $2}')

if ! command -v conda &>/dev/null; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  else
    echo "Error: conda not found. Run ./setup_envs.sh first to install Miniconda."
    return 1 2>/dev/null || exit 1
  fi
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
cd "$SCRIPT_DIR/$COMP"
echo "Activated '$ENV_NAME' — now in $(pwd)"
