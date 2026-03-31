#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/common_setup.sh"

# ── Main ───────────────────────────────────────────────────────
local_yml="$SCRIPT_DIR/vla_x3plus/environment.yml"
env_name=$(grep '^name:' "$local_yml" | awk '{print $2}')

log "━━━ vla_x3plus ━━━  env=${env_name}"

if [[ "$DRY_RUN" == true ]]; then
  log "  Would create env '$env_name' and run post-install steps"
  exit 0
fi

mkdir -p "$SCRIPT_DIR/vla_x3plus/output"

setup_conda_env vla_x3plus

lerobot_ver=$(conda run -n "$env_name" python -c "import lerobot; print(lerobot.__version__)")
if [[ "$lerobot_ver" == "0.5.0" ]]; then
  log "  Patching lerobot $lerobot_ver (remove duplicate @dataclass on GR00TN15Config)..."
  lerobot_site=$(conda run -n "$env_name" python -c "import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).parent)")
  patch -N -d "$(dirname "$lerobot_site")" -p1 < "$SCRIPT_DIR/vla_x3plus/patches/lerobot_groot_n1_dataclass.patch" \
    || log "  (patch already applied)"
else
  log "  Skipping lerobot patch (installed $lerobot_ver, patch targets 0.5.0)"
fi

log "  Installing MuJoCo + Gymnasium..."
conda run -n "$env_name" pip install mujoco "gymnasium[mujoco]"

log "  Installing imageio[ffmpeg] for video rendering..."
conda run -n "$env_name" pip install "imageio[ffmpeg]"

ok "vla_x3plus post-install complete"
