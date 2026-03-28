#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$SCRIPT_DIR/common_setup.sh"

# ── Meshroom (AliceVision photogrammetry) ─────────────────────
# Version is read from meshroom/config.yml (single source of truth).
# The Linux binary is hosted on Zenodo; the download URL is resolved
# dynamically from the GitHub release notes so that changing the version
# in config.yml is all that's needed for an upgrade.

MESHROOM_CONFIG="$SCRIPT_DIR/meshroom/config.yml"
MESHROOM_VERSION=$(grep '^version:' "$MESHROOM_CONFIG" 2>/dev/null | awk '{print $2}' | tr -d '"'"'")
if [[ -z "$MESHROOM_VERSION" ]]; then
  warn "meshroom/config.yml missing 'version:' — falling back to 2025.1.0"
  MESHROOM_VERSION="2025.1.0"
fi
MESHROOM_DIR="$SCRIPT_DIR/meshroom/Meshroom-${MESHROOM_VERSION}"

resolve_meshroom_url() {
  local version="$1"
  local api_url="https://api.github.com/repos/alicevision/Meshroom/releases/tags/v${version}"
  curl -fsSL "$api_url" 2>/dev/null \
    | grep -oP 'https://zenodo\.org/records/[0-9]+/files/Meshroom-[^"]*-Linux\.tar\.gz' \
    | head -1
}

install_meshroom() {
  if [[ -d "$MESHROOM_DIR" ]]; then
    ok "Meshroom ${MESHROOM_VERSION} already installed at $MESHROOM_DIR"
    return 0
  fi

  log "Installing Meshroom ${MESHROOM_VERSION}..."

  if [[ "$DRY_RUN" == true ]]; then
    log "  Would download and extract Meshroom-${MESHROOM_VERSION}-Linux.tar.gz"
    return 0
  fi

  if [[ "$CLEAN" == true && -d "$MESHROOM_DIR" ]]; then
    log "  Removing existing Meshroom installation..."
    rm -rf "$MESHROOM_DIR"
  fi

  log "  Resolving download URL from GitHub release v${MESHROOM_VERSION}..."
  local download_url
  download_url=$(resolve_meshroom_url "$MESHROOM_VERSION")

  if [[ -z "$download_url" ]]; then
    err "Could not resolve download URL for Meshroom ${MESHROOM_VERSION}"
    err "Verify the release exists: https://github.com/alicevision/Meshroom/releases/tag/v${MESHROOM_VERSION}"
    return 1
  fi

  ok "Resolved URL: $download_url"

  local tarball
  tarball=$(basename "$download_url")
  local dest="$SCRIPT_DIR/meshroom"
  local tarball_path="$dest/${tarball}"

  mkdir -p "$dest"
  log "  Downloading ${tarball} (very large; curl -C supports resume) ..."
  curl -fSL --retry 3 --retry-delay 10 -C - -o "$tarball_path" "$download_url"

  log "  Extracting to ${dest} ..."
  tar -xzf "$tarball_path" -C "$dest"
  rm -f "$tarball_path"

  if [[ -d "$MESHROOM_DIR" ]]; then
    ok "Meshroom ${MESHROOM_VERSION} installed at $MESHROOM_DIR"
  else
    err "Meshroom extraction failed — expected directory $MESHROOM_DIR not found"
    return 1
  fi
}

# ── Main ───────────────────────────────────────────────────────
install_meshroom

log "━━━ meshroom ━━━  (prebuilt binary, no conda env)"

if [[ "$DRY_RUN" == true ]]; then
  log "  Would verify Meshroom installation"
  exit 0
fi

if [[ ! -d "$MESHROOM_DIR" ]]; then
  err "Meshroom not found at $MESHROOM_DIR — install_meshroom may have failed"
  exit 1
fi

mkdir -p "$SCRIPT_DIR/meshroom/input" "$SCRIPT_DIR/meshroom/output"
ok "Meshroom ${MESHROOM_VERSION} ready"
