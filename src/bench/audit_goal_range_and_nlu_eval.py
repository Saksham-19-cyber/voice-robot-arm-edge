"""Audit FetchReach training goal range vs workspace bounds and evaluate NLU goals."""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
else:
    os.environ.setdefault("MUJOCO_GL", "glfw")

import gymnasium as gym
import gymnasium_robotics
import onnxruntime as ort

from src.rl.env_utils import (
    make_fetch_env,
    WORKSPACE_BOUNDS,
    WORKSPACE_CENTER,
    clip_to_workspace,
    GoalOverrideWrapper,
)
from src.rl.rollout import run_onnx_rollout
from src.nlu.rule_parser import RuleBasedParser, DIRECTION_VECTORS, ANCHORS


def audit_goal_range():
    print("=== AUDITING FETCHREACH TRAINING GOAL RANGE ===", flush=True)
    gym.register_envs(gymnasium_robotics)
    raw_env = gym.make("FetchReach-v4")
    unwrapped = raw_env.unwrapped

    target_range = float(unwrapped.target_range)
    init_gripper = unwrapped.initial_gripper_xpos[:3].astype(np.float64)
    raw_env.close()

    train_min = (init_gripper - target_range).tolist()
    train_max = (init_gripper + target_range).tolist()

    training_range = {
        "target_range": target_range,
        "initial_gripper_xpos": [round(float(c), 4) for c in init_gripper],
        "bounds": {
            "x": [round(train_min[0], 4), round(train_max[0], 4)],
            "y": [round(train_min[1], 4), round(train_max[1], 4)],
            "z": [round(train_min[2], 4), round(train_max[2], 4)],
        },
    }

    print(f"Initial Gripper: {training_range['initial_gripper_xpos']}")
    print(f"Training Range: +/- {target_range} m")
    print(f"X: {training_range['bounds']['x']}")
    print(f"Y: {training_range['bounds']['y']}")
    print(f"Z: {training_range['bounds']['z']}")

    # Check NLU goals against training bounds
    parser = RuleBasedParser()
    test_commands = [
        # Presets
        "reach to the left",
        "reach to the right",
        "reach up",
        "point arm towards table",
        "touch the block",
        "touch the cube",
        # Metric offsets in each direction
        "move left 5cm", "move left 10cm", "move left 20cm", "move left 30cm",
        "move right 5cm", "move right 10cm", "move right 20cm", "move right 30cm",
        "move up 5cm", "move up 10cm", "move up 20cm", "move up 30cm",
        "move down 5cm", "move down 10cm", "move down 20cm", "move down 30cm",
        "move forward 5cm", "move forward 10cm", "move forward 20cm", "move forward 30cm",
        "move backward 5cm", "move backward 10cm", "move backward 20cm", "move backward 30cm",
        # Edge clipped commands
        "move forward 10m",
        "move backward 10m",
        "move left 10m",
        "move right 10m",
        "move up 10m",
        "move down 10m",
    ]

    out_of_range_goals = []
    in_range_goals = []

    for cmd in test_commands:
        res = parser.parse(cmd)
        if res.goal_xyz is None:
            continue
        gx, gy, gz = res.goal_xyz
        is_outside = False
        reasons = []

        if gx < train_min[0]:
            is_outside = True
            reasons.append(f"x={gx:.3f} < min_x={train_min[0]:.3f}")
        elif gx > train_max[0]:
            is_outside = True
            reasons.append(f"x={gx:.3f} > max_x={train_max[0]:.3f}")

        if gy < train_min[1]:
            is_outside = True
            reasons.append(f"y={gy:.3f} < min_y={train_min[1]:.3f}")
        elif gy > train_max[1]:
            is_outside = True
            reasons.append(f"y={gy:.3f} > max_y={train_max[1]:.3f}")

        if gz < train_min[2]:
            is_outside = True
            reasons.append(f"z={gz:.3f} < min_z={train_min[2]:.3f}")
        elif gz > train_max[2]:
            is_outside = True
            reasons.append(f"z={gz:.3f} > max_z={train_max[2]:.3f}")

        entry = {
            "command": cmd,
            "goal_xyz": res.goal_xyz,
            "target": res.target,
            "was_clipped": res.metadata.get("was_clipped", False),
        }
        if is_outside:
            entry["out_of_range_reasons"] = reasons
            out_of_range_goals.append(entry)
        else:
            in_range_goals.append(entry)

    report = {
        "training_range": training_range,
        "workspace_bounds_declared": WORKSPACE_BOUNDS,
        "bounds_comparison": {
            "x_bounds": {"declared": list(WORKSPACE_BOUNDS["x"]), "training": training_range["bounds"]["x"]},
            "y_bounds": {"declared": list(WORKSPACE_BOUNDS["y"]), "training": training_range["bounds"]["y"]},
            "z_bounds": {"declared": list(WORKSPACE_BOUNDS["z"]), "training": training_range["bounds"]["z"]},
        },
        "total_commands_evaluated": len(test_commands),
        "in_range_count": len(in_range_goals),
        "out_of_range_count": len(out_of_range_goals),
        "out_of_range_goals": out_of_range_goals,
        "gl_backend": os.environ.get("MUJOCO_GL", "default"),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/goal_range.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Goal range audit saved to results/goal_range.json ({len(out_of_range_goals)} out-of-range commands)", flush=True)
    return report, train_min, train_max


def run_fast_rollout(env, session, goal_xyz, max_steps=50, seed=42):
    target_goal = np.array(goal_xyz, dtype=np.float32).copy()
    if hasattr(env, "set_goal"):
        env.set_goal(target_goal)
    obs_dict, _ = env.reset(seed=seed)
    
    # Strictly ensure env.unwrapped.goal and obs_dict match target_goal exactly
    env.unwrapped.goal = target_goal.copy()
    obs_dict["desired_goal"] = target_goal.copy()
    if hasattr(env.unwrapped, "_render_callback"):
        env.unwrapped._render_callback()

    assert np.allclose(env.unwrapped.goal, target_goal, atol=1e-4), (
        f"Goal mismatch: env.unwrapped.goal {env.unwrapped.goal} != target_goal {target_goal}"
    )

    succeeded_first = False
    first_step = max_steps
    final_dist = float(np.linalg.norm(obs_dict["achieved_goal"] - target_goal))
    
    for step_idx in range(max_steps):
        ob = obs_dict["observation"]
        ag = obs_dict["achieved_goal"]
        dg = obs_dict["desired_goal"]
        obs_flat = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)
        action = session.run(None, {"observation_flat": obs_flat})[0][0]
        obs_dict, _, _, _, info = env.step(action)
        dist = float(np.linalg.norm(obs_dict["achieved_goal"] - target_goal))
        final_dist = dist
        if (info.get("is_success", 0.0) == 1.0 or dist < 0.05) and not succeeded_first:
            succeeded_first = True
            first_step = step_idx + 1

    succeeded_final = bool(final_dist < 0.05)
    return succeeded_first, succeeded_final, first_step, final_dist


def evaluate_nlu_goals(n_episodes=20):
    print(f"\n=== EVALUATING INT8 & FP32 POLICIES ON NLU GOALS ({n_episodes} EPISODES EACH) ===", flush=True)
    fp32_model = "models/fetch_reach_actor_fp32.onnx"
    int8_model = "models/fetch_reach_actor_int8.onnx"

    parser = RuleBasedParser()
    goal_categories = [
        # Presets
        ("preset_reach_left", "reach to the left"),
        ("preset_reach_right", "reach to the right"),
        ("preset_reach_up", "reach up"),
        # Metric offsets
        ("offset_5cm_left", "move left 5cm"),
        ("offset_10cm_left", "move left 10cm"),
        ("offset_20cm_left", "move left 20cm"),
        ("offset_30cm_left", "move left 30cm"),
        ("offset_5cm_right", "move right 5cm"),
        ("offset_10cm_right", "move right 10cm"),
        ("offset_20cm_right", "move right 20cm"),
        ("offset_30cm_right", "move right 30cm"),
        ("offset_5cm_up", "move up 5cm"),
        ("offset_10cm_up", "move up 10cm"),
        ("offset_20cm_up", "move up 20cm"),
        ("offset_30cm_up", "move up 30cm"),
        ("offset_5cm_down", "move down 5cm"),
        ("offset_10cm_down", "move down 10cm"),
        ("offset_20cm_down", "move down 20cm"),
        ("offset_30cm_down", "move down 30cm"),
        ("offset_5cm_forward", "move forward 5cm"),
        ("offset_10cm_forward", "move forward 10cm"),
        ("offset_20cm_forward", "move forward 20cm"),
        ("offset_30cm_forward", "move forward 30cm"),
        ("offset_5cm_backward", "move backward 5cm"),
        ("offset_10cm_backward", "move backward 10cm"),
        ("offset_20cm_backward", "move backward 20cm"),
        ("offset_30cm_backward", "move backward 30cm"),
        # Edge goals (clipped to declared workspace)
        ("edge_clipped_forward_max", "move forward 10m"),
        ("edge_clipped_backward_min", "move backward 10m"),
        ("edge_clipped_left_max", "move left 10m"),
        ("edge_clipped_right_min", "move right 10m"),
        ("edge_clipped_up_max", "move up 10m"),
        ("edge_clipped_down_min", "move down 10m"),
    ]

    results = {
        "n_episodes_per_goal": n_episodes,
        "gl_backend": os.environ.get("MUJOCO_GL", "default"),
        "goals": {},
    }

    env = make_fetch_env(render_mode="rgb_array")
    sess_fp32 = ort.InferenceSession(fp32_model, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_model, providers=["CPUExecutionProvider"])

    for cat_name, cmd in goal_categories:
        parsed = parser.parse(cmd)
        target_goal = np.array(parsed.goal_xyz, dtype=np.float32)

        # Evaluate FP32
        fp32_first_succ = []
        fp32_final_succ = []
        fp32_steps = []
        fp32_distances = []
        for ep in range(n_episodes):
            s_first, s_final, st, dist = run_fast_rollout(
                env=env,
                session=sess_fp32,
                goal_xyz=target_goal,
                max_steps=50,
                seed=ep * 100 + 7,
            )
            fp32_first_succ.append(float(s_first))
            fp32_final_succ.append(float(s_final))
            fp32_steps.append(float(st))
            fp32_distances.append(float(dist))

        # Evaluate INT8
        int8_first_succ = []
        int8_final_succ = []
        int8_steps = []
        int8_distances = []
        sample_logs = []
        for ep in range(n_episodes):
            s_first, s_final, st, dist = run_fast_rollout(
                env=env,
                session=sess_int8,
                goal_xyz=target_goal,
                max_steps=50,
                seed=ep * 100 + 7,
            )
            int8_first_succ.append(float(s_first))
            int8_final_succ.append(float(s_final))
            int8_steps.append(float(st))
            int8_distances.append(float(dist))

            if ep < 10:
                sample_logs.append({
                    "episode": ep,
                    "requested_goal": [round(float(c), 4) for c in target_goal],
                    "env_goal_after_reset": [round(float(c), 4) for c in env.unwrapped.goal],
                    "achieved_final_position": [round(float(c), 4) for c in env.unwrapped.data.mocap_pos[0]],
                    "distance_to_requested_cm": round(float(dist * 100), 2),
                    "steps_to_goal": st,
                })

        results["goals"][cat_name] = {
            "command": cmd,
            "target_goal": [round(float(c), 4) for c in target_goal],
            "was_clipped": parsed.metadata.get("was_clipped", False),
            "fp32": {
                "success_rate_first_reached_within_5cm": round(float(np.mean(fp32_first_succ)), 4),
                "success_rate_final_step": round(float(np.mean(fp32_final_succ)), 4),
                "success_rate": round(float(np.mean(fp32_first_succ)), 4),
                "mean_steps_to_goal": round(float(np.mean(fp32_steps)), 2),
                "mean_final_distance_m": round(float(np.mean(fp32_distances)), 4),
                "max_final_distance_m": round(float(np.max(fp32_distances)), 4),
            },
            "int8": {
                "success_rate_first_reached_within_5cm": round(float(np.mean(int8_first_succ)), 4),
                "success_rate_final_step": round(float(np.mean(int8_final_succ)), 4),
                "success_rate": round(float(np.mean(int8_first_succ)), 4),
                "mean_steps_to_goal": round(float(np.mean(int8_steps)), 2),
                "mean_final_distance_m": round(float(np.mean(int8_distances)), 4),
                "max_final_distance_m": round(float(np.max(int8_distances)), 4),
                "sample_episodes_log": sample_logs,
            },
        }
        print(f"[{cat_name:28s}] FP32 SR: {results['goals'][cat_name]['fp32']['success_rate']*100:5.1f}% | INT8 SR: {results['goals'][cat_name]['int8']['success_rate']*100:5.1f}% | steps: {results['goals'][cat_name]['int8']['mean_steps_to_goal']:.1f} | dist: {results['goals'][cat_name]['int8']['mean_final_distance_m']*100:.2f} cm", flush=True)

    env.close()

    # Aggregate summaries
    all_fp32_first_sr = [g["fp32"]["success_rate_first_reached_within_5cm"] for g in results["goals"].values()]
    all_fp32_final_sr = [g["fp32"]["success_rate_final_step"] for g in results["goals"].values()]
    all_int8_first_sr = [g["int8"]["success_rate_first_reached_within_5cm"] for g in results["goals"].values()]
    all_int8_final_sr = [g["int8"]["success_rate_final_step"] for g in results["goals"].values()]
    all_int8_steps = [g["int8"]["mean_steps_to_goal"] for g in results["goals"].values()]
    all_int8_dists = [g["int8"]["mean_final_distance_m"] for g in results["goals"].values()]

    results["summary"] = {
        "mean_fp32_success_rate": round(float(np.mean(all_fp32_first_sr)), 4),
        "mean_int8_success_rate": round(float(np.mean(all_int8_first_sr)), 4),
        "mean_int8_success_rate_final_step": round(float(np.mean(all_int8_final_sr)), 4),
        "mean_int8_steps_to_goal": round(float(np.mean(all_int8_steps)), 2),
        "mean_int8_final_distance_cm": round(float(np.mean(all_int8_dists) * 100), 2),
        "presets_int8_success_rate": round(float(np.mean([results["goals"][k]["int8"]["success_rate"] for k in results["goals"] if k.startswith("preset")])), 4),
        "offsets_int8_success_rate": round(float(np.mean([results["goals"][k]["int8"]["success_rate"] for k in results["goals"] if k.startswith("offset")])), 4),
        "edges_int8_success_rate": round(float(np.mean([results["goals"][k]["int8"]["success_rate"] for k in results["goals"] if k.startswith("edge")])), 4),
    }

    with open("results/nlu_goal_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"NLU goal evaluation saved to results/nlu_goal_eval.json. Overall INT8 SR: {results['summary']['mean_int8_success_rate']*100:.1f}%", flush=True)
    return results


def audit_action_differences(n_random_states=1000):
    print(f"\n=== AUDITING INT8 VS FP32 ACTION DIFFERENCES ({n_random_states} RANDOM STATES + ROLLOUT STATES) ===", flush=True)
    fp32_sess = ort.InferenceSession("models/fetch_reach_actor_fp32.onnx", providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", providers=["CPUExecutionProvider"])
    in_name = fp32_sess.get_inputs()[0].name

    # 1. Random observation states (16 dims: 10 ob + 3 ag + 3 dg)
    np.random.seed(42)
    random_states = np.random.uniform(-1.5, 1.5, size=(n_random_states, 16)).astype(np.float32)

    diffs_random = []
    for i in range(n_random_states):
        obs = random_states[i:i+1]
        act_fp32 = fp32_sess.run(None, {in_name: obs})[0]
        act_int8 = int8_sess.run(None, {in_name: obs})[0]
        diffs_random.append(np.abs(act_fp32 - act_int8))

    diffs_random = np.concatenate(diffs_random, axis=0)  # shape (1000, 4)
    max_err_random = float(np.max(diffs_random))
    mean_err_random = float(np.mean(diffs_random))

    # 2. Rollout visited states
    env = make_fetch_env(render_mode="rgb_array")
    diffs_rollout = []
    for seed in range(20):
        obs_dict, _ = env.reset(seed=seed)
        for _ in range(50):
            ob = obs_dict["observation"]
            ag = obs_dict["achieved_goal"]
            dg = obs_dict["desired_goal"]
            inp = np.concatenate([ob, ag, dg], axis=0).astype(np.float32).reshape(1, 16)

            act_fp32 = fp32_sess.run(None, {in_name: inp})[0]
            act_int8 = int8_sess.run(None, {in_name: inp})[0]
            diffs_rollout.append(np.abs(act_fp32 - act_int8))

            obs_dict, _, done, truncated, _ = env.step(act_int8[0])
            if done or truncated:
                break
    env.close()

    diffs_rollout = np.concatenate(diffs_rollout, axis=0)
    max_err_rollout = float(np.max(diffs_rollout))
    mean_err_rollout = float(np.mean(diffs_rollout))

    action_report = {
        "random_states": {
            "n_states": n_random_states,
            "mean_absolute_action_error": round(mean_err_random, 5),
            "max_absolute_action_error": round(max_err_random, 5),
        },
        "rollout_visited_states": {
            "n_states": len(diffs_rollout),
            "mean_absolute_action_error": round(mean_err_rollout, 5),
            "max_absolute_action_error": round(max_err_rollout, 5),
        },
        "gl_backend": os.environ.get("MUJOCO_GL", "default"),
    }

    with open("results/action_difference_eval.json", "w") as f:
        json.dump(action_report, f, indent=2)
    print("Action difference report saved to results/action_difference_eval.json:")
    print(json.dumps(action_report, indent=2), flush=True)
    return action_report


if __name__ == "__main__":
    audit_goal_range()
    evaluate_nlu_goals(n_episodes=20)
    audit_action_differences(n_random_states=1000)

