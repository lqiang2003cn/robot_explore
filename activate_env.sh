#!/usr/bin/env bash
# Convenience wrapper: source this to activate a component's env and cd into it.
#
# Usage:  source activate_env.sh <component>
# Example: source activate_env.sh planning

if [[ $# -lt 1 ]]; then
  echo "Usage: source activate_env.sh <component>"
  echo "Available: simulation"
  return 1 2>/dev/null || exit 1
fi

COMP="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
