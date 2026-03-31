#!/usr/bin/env bash
# Fine-tune SmolVLA on the X3Plus pick-and-place dataset.
#
# Usage:
#   bash src/train.sh                          # defaults from config.yml
#   STEPS=5000 bash src/train.sh               # override training steps
#   DATASET_ID=local/x3plus_v2 bash src/train.sh
#
# Prerequisites:
#   1. source activate_env.sh vla_x3plus
#   2. Collect demonstrations via one of:
#      - python -m src.record_dataset                         (direct MuJoCo, delta actions)
#      - python -m src.record_dataset_ros2 --num-episodes 100 (ROS2 topics, absolute actions)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "roboex-vla-x3plus" ]]; then
  echo "Activating vla_x3plus environment ..."
  source "$REPO_ROOT/activate_env.sh" vla_x3plus
fi

# Read defaults from config.yml via Python (keeps the shell script simple)
read_cfg() {
  python3 -c "
import yaml, sys
with open('$SCRIPT_DIR/config.yml') as f:
    cfg = yaml.safe_load(f)
t = cfg.get('training', {})
d = cfg.get('dataset', {})
print(d.get('repo_id', 'local/x3plus_pick_cube'))
print(t.get('pretrained_path', 'lerobot/smolvla_base'))
print(t.get('batch_size', 32))
print(t.get('steps', 20000))
print(t.get('output_dir', 'outputs/train/smolvla_x3plus'))
"
}

IFS=$'\n' read -r -d '' CFG_DATASET CFG_PRETRAINED CFG_BATCH CFG_STEPS CFG_OUTDIR < <(read_cfg && printf '\0') || true

DATASET_ID="${DATASET_ID:-$CFG_DATASET}"
PRETRAINED="${PRETRAINED:-$CFG_PRETRAINED}"
BATCH_SIZE="${BATCH_SIZE:-$CFG_BATCH}"
STEPS="${STEPS:-$CFG_STEPS}"
OUTPUT_DIR="${OUTPUT_DIR:-$CFG_OUTDIR}"

echo "=== SmolVLA Fine-Tuning ==="
echo "  Dataset    : $DATASET_ID"
echo "  Pretrained : $PRETRAINED"
echo "  Batch size : $BATCH_SIZE"
echo "  Steps      : $STEPS"
echo "  Output     : $OUTPUT_DIR"
echo ""

exec lerobot-train \
  --policy.type smolvla \
  --policy.pretrained_path "$PRETRAINED" \
  --dataset.repo_id "$DATASET_ID" \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --output_dir "$OUTPUT_DIR"
