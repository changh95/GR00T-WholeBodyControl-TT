#!/usr/bin/env bash
# record_demo.sh: screen capture of the live demo, out of process (PLAN.md WP-1.7, R16, §4.3).
#
#   record_demo.sh [--out FILE.mp4] [--display :1] [--size WxH] [--fps 30] [--max-width 1920]
#                  [--duration S] [--dry-run]
#
# ffmpeg x11grab -> libx264 ultrafast, pinned to CPUs 6-7 at nice 10 with 2 encoder threads, so it never
# competes with the tick thread (CPU 2/18) or its helpers (CPU 3). In-loop video costs 17-87 ms per frame
# (U:sim), which is why recording lives here. ffmpeg is looked up in $FFMPEG, then PATH, then the sim venv's
# imageio_ffmpeg binary (has x11grab + libx264); with none, or no reachable X display, this script reports
# why and exits 3 without recording (demo.sh carries on). SIGINT/SIGTERM make ffmpeg finish the file; the
# fragmented MP4 below stays playable up to the last 2 s fragment even if it is killed first.
set -u -o pipefail

RT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$RT_DIR/env.sh"

OUT="$RT_DIR/runs/demo_logs/$(date +%Y%m%d_%H%M%S)_demo.mp4"
DISP=${DISPLAY:-:1}; SIZE=auto; FPS=30; MAXW=1920; DUR=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=$2; shift 2 ;;
    --display) DISP=$2; shift 2 ;;
    --size) SIZE=$2; shift 2 ;;
    --fps) FPS=$2; shift 2 ;;
    --max-width) MAXW=$2; shift 2 ;;
    --duration) DUR=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "record_demo: unknown option $1" >&2; exit 2 ;;
  esac
done
export DISPLAY=$DISP
export XAUTHORITY=${XAUTHORITY:-/run/user/1000/gdm/Xauthority}

report() { echo "record_demo: not recording: $*" >&2; exit 3; }

find_ffmpeg() {
  local c
  for c in "${FFMPEG:-}" "$(command -v ffmpeg 2>/dev/null)" \
           "$(dirname "$SIM_PYTHON")"/../lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-*; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

FF=$(find_ffmpeg) || FF=""
CMD_SIZE=$SIZE
if [ "$DRY" = 0 ]; then
  [ -n "$FF" ] || report "no ffmpeg (looked at \$FFMPEG, PATH and the sim venv's imageio_ffmpeg)"
  "$FF" -hide_banner -devices 2>/dev/null | grep -q x11grab || report "$FF has no x11grab input device"
  if [ "$SIZE" = auto ]; then
    geo=$(xwininfo -root -display "$DISP" 2>/dev/null) || report "X display $DISP is not reachable (DISPLAY/XAUTHORITY)"
    w=$(awk '/Width:/ {print $2}' <<<"$geo"); h=$(awk '/Height:/ {print $2}' <<<"$geo")
    [ -n "$w" ] && [ -n "$h" ] || report "cannot read the size of $DISP"
    CMD_SIZE="${w}x${h}"
  fi
  mkdir -p "$(dirname "$OUT")"
fi

# shellcheck disable=SC2054  # "6,7" is taskset's CPU list, not two array elements
CMD=(taskset -c 6,7 nice -n 10 "${FF:-ffmpeg}" -hide_banner -loglevel warning -nostdin -y
     -f x11grab -draw_mouse 0 -framerate "$FPS" -video_size "$CMD_SIZE" -i "${DISP}+0,0"
     -vf "scale='min($MAXW,iw)':-2" -c:v libx264 -preset ultrafast -tune zerolatency -crf 23
     -pix_fmt yuv420p -threads 2 -g "$((FPS * 2))" -movflags +frag_keyframe+empty_moov+default_base_moof)
# Fragmented MP4 (a keyframe and a fragment every 2 s): the file plays even if ffmpeg is killed before it can
# finish. A plain MP4 needs the trailer written at exit, and at 2560x2880 the encoder's backlog took longer than
# demo.sh's grace period, so every recording up to T-2.6 lost its moov atom.
[ -n "$DUR" ] && CMD+=(-t "$DUR")
CMD+=("$OUT")

if [ "$DRY" = 1 ]; then
  [ -n "$FF" ] || echo "# NOTE: no ffmpeg found; a live run would report and exit 3"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
echo "record_demo: $DISP ${CMD_SIZE} @ ${FPS} fps -> $OUT ($FF)" >&2
exec "${CMD[@]}"
