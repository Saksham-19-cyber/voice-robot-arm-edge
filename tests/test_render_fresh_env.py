"""Test that fresh env instantiation across threads renders valid non-black frames."""
import os
import sys
import threading
import numpy as np

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.nlu.parser import VoiceCommandParser
from src.rl.rollout import run_onnx_rollout


def test_fresh_env_render_threads():
    nlu = VoiceCommandParser()
    commands = [
        ("Reach Left", "reach to the left"),
        ("Reach Right", "reach to the right"),
        ("Move Up 10cm", "move up by 10 cm"),
    ]

    results = {}
    errors = {}

    def run_command_in_thread(name, cmd_text):
        intent_res = nlu.parse(cmd_text)
        assert intent_res.goal_xyz is not None, f"Parsing failed for {cmd_text}"
        goal = np.array(intent_res.goal_xyz, dtype=np.float32)

        try:
            res = run_onnx_rollout(
                goal_xyz=goal,
                onnx_path="models/fetch_reach_actor_int8.onnx",
                max_steps=50,
                render_width=320,
                render_height=320,
                env=None,
            )
            results[name] = res
        except Exception as e:
            errors[name] = e

    # Call the rollout 3 times in a row, each in a new threading.Thread
    for name, cmd in commands:
        t = threading.Thread(target=run_command_in_thread, args=(name, cmd))
        t.start()
        t.join()

    # Assert no errors occurred in thread execution
    assert not errors, f"Encountered errors in threaded rollouts: {errors}"

    # Assert frames are not black and final end-effector is within 2 cm of requested goal
    for name, _ in commands:
        res = results[name]
        frames = res["frames"]
        assert len(frames) > 0, f"No frames produced for {name}"
        assert np.mean(frames[0]) > 5.0, f"First frame is black for {name} (mean: {np.mean(frames[0])})"
        assert np.mean(frames[-1]) > 5.0, f"Last frame is black for {name} (mean: {np.mean(frames[-1])})"
        assert res["final_distance_cm"] <= 2.0, (
            f"Final distance for {name} was {res['final_distance_cm']} cm, expected <= 2.0 cm"
        )

    # Assert first frame of Reach Left differs from Reach Right
    f0_left = results["Reach Left"]["frames"][0]
    f0_right = results["Reach Right"]["frames"][0]
    assert not np.array_equal(f0_left, f0_right), "First frame of Reach Left matches Reach Right exactly"
    pixel_diff = float(np.mean(np.abs(f0_left.astype(float) - f0_right.astype(float))))
    assert pixel_diff > 0.0, f"Expected visual difference between Reach Left and Reach Right, got diff {pixel_diff}"

