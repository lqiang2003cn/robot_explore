#!/usr/bin/env bash
# Launch MuJoCo bridge + BT pick-and-place pipeline in one shot.
#
# Usage:
#   ./run_pick_place.sh                         # headless, records video
#   ./run_pick_place.sh --no-video              # skip video recording
#   ./run_pick_place.sh --video path/out.mp4    # custom video path
#
# Prerequisites:
#   - ROS 2 Jazzy installed  (/opt/ros/jazzy/setup.bash)
#   - conda env 'roboex-vla-x3plus' created    (./setup_envs.sh vla_x3plus)
#   - x3plus_pick_place built in ros2_stack/ws  (colcon build)

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VIDEO_PATH="$SCRIPT_DIR/output/bt_pick_place.mp4"
RECORD_VIDEO=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-video)   RECORD_VIDEO=false; shift ;;
    --video)      VIDEO_PATH="$2"; shift 2 ;;
    *)            echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Source ROS 2 ──────────────────────────────────────────────────────────────
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ERROR: ROS 2 Jazzy not found at /opt/ros/jazzy/setup.bash"
  exit 1
fi
source /opt/ros/jazzy/setup.bash

# Source the colcon workspace so ros2 launch can find x3plus_pick_place
COLCON_SETUP="$REPO_ROOT/ros2_stack/ws/install/setup.bash"
if [[ ! -f "$COLCON_SETUP" ]]; then
  echo "ERROR: colcon workspace not built — run 'colcon build' in ros2_stack/ws first"
  exit 1
fi
source "$COLCON_SETUP"

# ── Activate conda env for MuJoCo bridge ─────────────────────────────────────
if ! command -v conda &>/dev/null; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  else
    echo "ERROR: conda not found"
    exit 1
  fi
fi
eval "$(conda shell.bash hook)"
conda activate roboex-vla-x3plus

# ── Cleanup on exit ──────────────────────────────────────────────────────────
BRIDGE_PID=""
PP_PID=""

cleanup() {
  echo ""
  echo "Shutting down..."
  [[ -n "$PP_PID" ]]     && kill "$PP_PID"     2>/dev/null && wait "$PP_PID"     2>/dev/null || true
  [[ -n "$BRIDGE_PID" ]] && kill "$BRIDGE_PID" 2>/dev/null && wait "$BRIDGE_PID" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

# ── Start MuJoCo bridge ─────────────────────────────────────────────────────
BRIDGE_ARGS=()
if $RECORD_VIDEO; then
  mkdir -p "$(dirname "$VIDEO_PATH")"
  BRIDGE_ARGS+=(--record-video "$VIDEO_PATH")
fi

echo "==> Starting MuJoCo bridge node ..."
(cd "$SCRIPT_DIR" && python -m src.mujoco_bridge_node "${BRIDGE_ARGS[@]}") &
BRIDGE_PID=$!

# Wait for the bridge to start publishing before launching the BT pipeline.
echo "==> Waiting for /joint_states topic ..."
until ros2 topic info /joint_states &>/dev/null; do
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "ERROR: MuJoCo bridge exited prematurely"
    exit 1
  fi
  sleep 0.5
done
echo "    /joint_states available"

# ── Start pick-and-place BT pipeline ────────────────────────────────────────
echo "==> Launching x3plus_pick_place pick-and-place pipeline ..."
ros2 launch x3plus_pick_place pick_place.launch.py &
PP_PID=$!

# ── Wait for both processes ──────────────────────────────────────────────────
# The bridge exits on its own after saving video (when task_complete fires).
# The BT node exits after the tree succeeds or fails.
wait "$BRIDGE_PID" "$PP_PID" 2>/dev/null || true
