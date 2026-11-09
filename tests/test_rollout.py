"""
Unit tests for ONNX policy rollout and MuJoCo environment interaction.
Verifies FP32 and INT8 ONNX models execute closed-loop rollouts cleanly.
"""

import os
import pytest
import numpy as np
import onnxruntime as ort

from src.rl.rollout import run_onnx_rollout
from src.rl.env_utils import WORKSPACE_CENTER


def test_fp32_inference_session():
    path = "models/fetch_reach_actor_fp32.onnx"
    assert os.path.exists(path), f"FP32 model missing: {path}"
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    
    # FetchReach input dim is 16
    dummy_obs = np.random.randn(1, 16).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_obs})[0]
    
    assert output.shape == (1, 4)
    # Action bounds in Fetch are [-1, 1]
    assert np.all(output >= -1.0) and np.all(output <= 1.0)


def test_int8_inference_session():
    path = "models/fetch_reach_actor_int8.onnx"
    assert os.path.exists(path), f"INT8 model missing: {path}"
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    
    dummy_obs = np.random.randn(1, 16).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: dummy_obs})[0]
    
    assert output.shape == (1, 4)
    assert np.all(output >= -1.0) and np.all(output <= 1.0)


def test_closed_loop_rollout_reaches_goal_fp32():
    target = WORKSPACE_CENTER.copy()
    res = run_onnx_rollout(
        goal_xyz=target,
        onnx_path="models/fetch_reach_actor_fp32.onnx",
        max_steps=25,
        render_width=160,
        render_height=120,
        output_format="mp4",
    )
    assert res["succeeded"] is True
    assert res["first_success_step"] is not None and res["first_success_step"] <= 10
    assert res["steps_taken"] <= 25
    assert res["final_distance_m"] <= 0.05
    assert os.path.exists(res["video_path"])
    # Clean up generated video
    if os.path.exists(res["video_path"]):
        os.remove(res["video_path"])


def test_closed_loop_rollout_reaches_goal_int8():
    target = WORKSPACE_CENTER.copy()
    res = run_onnx_rollout(
        goal_xyz=target,
        onnx_path="models/fetch_reach_actor_int8.onnx",
        max_steps=25,
        render_width=160,
        render_height=120,
        output_format="mp4",
    )
    assert res["succeeded"] is True
    assert res["first_success_step"] is not None and res["first_success_step"] <= 10
    assert res["steps_taken"] <= 25
    assert res["final_distance_m"] <= 0.05
    assert os.path.exists(res["video_path"])
    if os.path.exists(res["video_path"]):
        os.remove(res["video_path"])
