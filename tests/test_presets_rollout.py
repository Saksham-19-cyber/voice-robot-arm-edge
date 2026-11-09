import os
import sys
import subprocess
import numpy as np
from PIL import Image

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.nlu.parser import VoiceCommandParser
from src.rl.rollout import run_onnx_rollout
import imageio_ffmpeg


def test_three_presets_rollout_visuals_and_video_encoding():
    nlu = VoiceCommandParser()
    presets = [
        "reach to the left",
        "move up by 10 cm",
        "move down towards table",
    ]

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    for cmd in presets:
        intent_res = nlu.parse(cmd)
        assert intent_res.goal_xyz is not None, f"Failed NLU for {cmd}"
        goal = np.array(intent_res.goal_xyz, dtype=np.float32)

        res = run_onnx_rollout(
            goal_xyz=goal,
            onnx_path="models/fetch_reach_actor_int8.onnx",
            max_steps=25,
            early_termination=True,
            render_width=320,
            render_height=320,
        )

        # Expected keys check
        required_keys = [
            "video_path", "video_bytes", "gif_path", "gif_bytes", "frames",
            "n_frames", "steps_taken", "succeeded", "first_success_step",
            "final_distance_cm", "start_pos_xyz", "goal_xyz", "clipped",
            "distances_m", "ee_positions", "actions", "rollout_time_ms",
            "encoding_time_ms", "render_time_per_frame_ms"
        ]
        for k in required_keys:
            assert k in res, f"Missing key {k} in rollout result for {cmd}"

        # Frames check: not all black and motion occurred
        frames = res["frames"]
        assert len(frames) >= 2, f"Expected at least 2 frames for {cmd}"
        f0 = frames[0]
        flast = frames[-1]

        assert np.mean(f0) > 50.0, f"Initial frame appears black for {cmd}"
        assert np.mean(flast) > 50.0, f"Final frame appears black for {cmd}"

        pixel_diff = float(np.mean(np.abs(flast.astype(float) - f0.astype(float))))
        assert pixel_diff > 1.0, f"No visual motion detected for {cmd} (diff={pixel_diff})"

        # Verify MP4 codec (h264) and pixel format (yuv420p)
        assert os.path.exists(res["video_path"])
        p = subprocess.run([ffmpeg_exe, "-i", res["video_path"]], capture_output=True, text=True)
        ffmpeg_log = p.stderr
        assert "h264" in ffmpeg_log, f"MP4 not encoded with h264 for {cmd}: {ffmpeg_log}"
        assert "yuv420p" in ffmpeg_log, f"MP4 not yuv420p for {cmd}: {ffmpeg_log}"
        assert "320x320" in ffmpeg_log, f"MP4 resolution mismatch for {cmd}: {ffmpeg_log}"

