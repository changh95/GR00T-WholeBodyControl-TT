"""Per-tick overhead of VideoRecordingWrapper (h264 encode in the control loop)."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.pop("GR00T_WBC_SERVER", None)
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import numpy as np
from gr00t.eval.sim.wrapper.video_recording_wrapper import VideoRecorder, VideoRecordingWrapper
from sim_timing import hold_action, make_env
out = Path(os.path.dirname(__file__)) / "video_tmp"
env = VideoRecordingWrapper(make_env(True), VideoRecorder.create_h264(fps=20, codec="h264", input_pix_fmt="rgb24", crf=22, thread_type="FRAME", thread_count=1),
                            video_dir=out, steps_per_render=2, max_episode_steps=1440, overlay_text=True)
obs, _ = env.reset()
ts = []
for i in range(300):
    t = time.perf_counter(); obs, *_ = env.step(hold_action(obs)); ts.append(time.perf_counter() - t)
ts = np.array(ts[10:]) * 1e3
print(f"tick with VideoRecordingWrapper: mean {ts.mean():.2f} ms p50 {np.median(ts):.2f} p99 {np.percentile(ts,99):.2f} max {ts.max():.2f}")
print(f"even ticks (no frame) {np.mean(ts[1::2]):.2f} ms, odd ticks (frame) {np.mean(ts[0::2]):.2f} ms")
env.video_recorder.stop()
