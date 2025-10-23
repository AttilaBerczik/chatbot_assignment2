#!/usr/bin/env bash
set -euo pipefail

RUNTIME_CACHE_DIR="${MODELS_CACHE_DIR:-/app/models_cache}"
BAKED_CACHE_DIR="/image_models_cache"

# Ensure runtime cache directory exists
mkdir -p "$RUNTIME_CACHE_DIR"

# If runtime cache is empty but baked cache has content, seed it
if [ -d "$BAKED_CACHE_DIR" ] && [ -z "$(ls -A "$RUNTIME_CACHE_DIR")" ] && [ -n "$(ls -A "$BAKED_CACHE_DIR" || true)" ]; then
  echo "Seeding models cache from baked image cache..."
  cp -a "$BAKED_CACHE_DIR"/. "$RUNTIME_CACHE_DIR"/
  echo "Seeding complete."
fi

# Show what's in the cache for visibility
echo "Models cache directory: $RUNTIME_CACHE_DIR"
ls -lah "$RUNTIME_CACHE_DIR" || true

# If no args provided, default to running the app
if [ "$#" -eq 0 ]; then
  exec python -u app.py
else
  exec "$@"
fi

