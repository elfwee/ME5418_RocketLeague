#!/usr/bin/env bash
# Build the deliverables from a 120 fps master render.
#   ./finish.sh MASTER_120.mp4 SFX.wav
# Each 60 fps output frame averages two 120 fps frames (180° shutter motion blur).
set -euo pipefail
cd "$(dirname "$0")"
MASTER="$1"; SFX="$2"
BLUR="tmix=frames=2:weights='1 1',select='mod(n\,2)',setpts=N/60/TB"
ENC=(-c:v libx264 -preset slow -crf 14 -pix_fmt yuv420p -profile:v high -r 60
     -color_primaries bt709 -color_trc bt709 -colorspace bt709 -movflags +faststart)

ffmpeg -v error -y -i "$MASTER" -i "$SFX" -vf "$BLUR" "${ENC[@]}" \
  -c:a aac -b:a 320k -ar 48000 -shortest car_soccer_intro.mp4
ffmpeg -v error -y -i car_soccer_intro.mp4 -c copy -an car_soccer_intro_silent.mp4
cp "$SFX" car_soccer_intro_sound.wav
ffmpeg -v error -y -ss 2.3 -i car_soccer_intro.mp4 -frames:v 1 car_soccer_intro_poster.png
ls -la car_soccer_intro*
