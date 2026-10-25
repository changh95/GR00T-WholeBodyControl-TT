"""Cut one successful episode out of a demo recording and lay the MuJoCo viewer and the HUD side by side (1920x1080).

usage: cut_success.py RUN_DIR DEMO_MP4 EPISODE OUT.mp4 --rec-start-epoch S --ff FFMPEG [--wbc-caption TEXT]
       [--tail-s T]
Tick times (CLOCK_MONOTONIC) map to recording time through the wall-clock offset and the recorder's start time
(S = the recording's birth time, `stat -c %W demo.mp4`); run it on the machine that recorded, in the same boot.
The crop boxes assume scripts/video/arrange_windows.py's layout on a 2560x2880 screen recorded at 1920 wide.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np

run, mp4, ep, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
rec_start = float(sys.argv[sys.argv.index("--rec-start-epoch") + 1])
FF = sys.argv[sys.argv.index("--ff") + 1]
n = int((run / "ticks.len").read_text().split()[0])
t = np.load(run / "ticks.npy", mmap_mode="r")[:n]
m = np.asarray(t["ep"]) == ep
k = np.asarray(t["k"])[m]; ts = np.asarray(t["t_start_ns"], np.int64)[m]; succ = np.asarray(t["success"])[m]
mono2wall = time.time() - time.monotonic()
t0 = ts[0] / 1e9 + mono2wall - rec_start - 2.0  # 2 s of lead-in
i_s = int(np.argmax(succ)) if succ.any() else len(ts) - 1
tail = float(sys.argv[sys.argv.index("--tail-s") + 1]) if "--tail-s" in sys.argv else 4.0
t1 = ts[i_s] / 1e9 + mono2wall - rec_start + tail  # after the success tick (the env resets right after it)
eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
e = next(x for x in eps if x["ep"] == ep)
s = 0.75  # the recorder scales 2560 -> 1920 wide
vx, vy, vw, vh = round(120 * s), round(114 * s), round(1706 * s), round(1920 * s)
hx, hy, hw, hh = round(120 * s), round(2137 * s), round(1280 * s), round(720 * s)
from PIL import Image, ImageDraw, ImageFont
cap = Image.new("RGBA", (960, 300), (18, 18, 24, 255))
dr = ImageDraw.Draw(cap)
fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
fr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
dr.text((24, 20), "One Tenstorrent Blackhole p150a", font=fb, fill=(255, 255, 255))
dr.text((24, 78), "VLA   GR00T N1.6 on the Tensix grid", font=fr, fill=(230, 230, 230))
wbc = sys.argv[sys.argv.index("--wbc-caption") + 1] if "--wbc-caption" in sys.argv else "gear_wbc MLP on the on-die X280 RISC-V"
dr.text((24, 112), "WBC   " + wbc, font=fr, fill=(230, 230, 230))
dr.text((24, 160), f"Unitree G1, apple to plate - seed {e['seed']}", font=fr, fill=(200, 200, 200))
dr.text((24, 194), f"{e['end']} at tick {e['k_end']}  (50 Hz real-time, async VLA chunks)", font=fr, fill=(140, 230, 140))
dr.text((24, 240), "MuJoCo viewer (left) and the rtloop HUD (right), recorded live", font=fr, fill=(150, 150, 160))
capp = out.with_suffix(".caption.png"); cap.save(capp)
fc = (f"[0:v]crop={vw}:{vh}:{vx}:{vy},scale=-2:1080,pad=1920:1080:(960-iw)/2:0:color=0x121218[v];"
      f"[0:v]crop={hw}:{hh}:{hx}:{hy},scale=960:-2[h];"
      f"[v][h]overlay=960:90[vh];[vh][1:v]overlay=960:690[o]")
cmd = [FF, "-v", "error", "-y", "-ss", f"{t0:.2f}", "-t", f"{t1 - t0:.2f}", "-i", str(mp4), "-i", str(capp), "-filter_complex", fc,
       "-map", "[o]", "-s", "1920x1080", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", "-threads", "2", str(out)]
print(f"episode {ep} seed {e['seed']} {e['end']} k={e['k_end']}: recording {t0:.1f}-{t1:.1f} s", flush=True)
subprocess.run(cmd, check=True)
print(out, out.stat().st_size)
