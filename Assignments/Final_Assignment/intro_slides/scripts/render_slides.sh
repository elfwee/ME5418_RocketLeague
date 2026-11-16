#!/usr/bin/env bash
# Render every slide of slides.html to a PNG with headless Chrome.
#   scripts/render_slides.sh            -> png/slide_0N.png at 1920x1080
#   SCALE=2 scripts/render_slides.sh    -> 3840x2160 (4K) instead
set -euo pipefail
cd "$(dirname "$0")/.."
SCALE="${SCALE:-1}"
CHROME="${CHROME:-google-chrome}"
NAMES=(title environment learners curriculum loopholes results up_next)
mkdir -p png
for i in "${!NAMES[@]}"; do
  n=$((i + 1))
  out="png/slide_0${n}_${NAMES[$i]}.png"
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor="$SCALE" \
    --window-size=1920,1080 --virtual-time-budget=10000 \
    --screenshot="$PWD/$out" "file://$PWD/slides.html?s=$n" >/dev/null 2>&1
  echo "$out"
done
