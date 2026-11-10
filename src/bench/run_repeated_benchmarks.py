"""Driver to execute policy, rollout FPS, and pipeline benchmarks 5 times each.

Records system execution conditions (High Performance power plan, AC plugged in)
and writes each run to results/runs/, then writes median and min-max ranges
into results/policy_benchmark.json, results/rollout_fps_audit.json, and results/pipeline_latency.json.
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
os.makedirs("results/runs", exist_ok=True)

from src.bench.run_policy_benchmark import benchmark_policy_isolated, bootstrap_p50_diff
from src.bench.audit_rollout_fps import audit_rollout_fps
from src.bench.measure_pipeline_latency import benchmark_pipeline_latency

SYSTEM_STATE = {
    "power_plan": "High Performance",
    "ac_power_plugged_in": True,
    "background_apps_closed": True,
    "hardware": "Intel Core i5-1235U, 12 threads, Windows 11",
    "gl_backend": "glfw",
}


def run_5x_policy_benchmark():
    print("=== Running Policy Benchmark 5x (20k iterations each) ===")
    runs = []
    fp32_p50s = []
    int8_p50s = []
    diffs = []

    for run_idx in range(1, 6):
        print(f"  Policy run {run_idx}/5...")
        fp32_metrics = benchmark_policy_isolated("models/fetch_reach_actor_fp32.onnx")
        int8_metrics = benchmark_policy_isolated("models/fetch_reach_actor_int8.onnx")

        # Isolated run difference
        diff_val = round(fp32_metrics["p50_us"] - int8_metrics["p50_us"], 2)

        run_summary = {
            "run_index": run_idx,
            "fp32_p50_us": fp32_metrics["p50_us"],
            "fp32_p95_us": fp32_metrics["p95_us"],
            "int8_p50_us": int8_metrics["p50_us"],
            "int8_p95_us": int8_metrics["p95_us"],
            "diff_p50_us": diff_val,
        }
        with open(f"results/runs/policy_run_{run_idx}.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        runs.append(run_summary)
        fp32_p50s.append(fp32_metrics["p50_us"])
        int8_p50s.append(int8_metrics["p50_us"])
        diffs.append(diff_val)

    # Load main file
    with open("results/policy_benchmark.json", "r") as f:
        main_data = json.load(f)

    main_data["system_execution_state"] = SYSTEM_STATE
    main_data["repeated_runs"] = {
        "n_runs": 5,
        "fp32_p50_us": {
            "median": round(float(np.median(fp32_p50s)), 1),
            "min": round(float(np.min(fp32_p50s)), 1),
            "max": round(float(np.max(fp32_p50s)), 1),
            "range": f"[{np.min(fp32_p50s):.1f}, {np.max(fp32_p50s):.1f}]",
            "values": fp32_p50s,
        },
        "int8_p50_us": {
            "median": round(float(np.median(int8_p50s)), 1),
            "min": round(float(np.min(int8_p50s)), 1),
            "max": round(float(np.max(int8_p50s)), 1),
            "range": f"[{np.min(int8_p50s):.1f}, {np.max(int8_p50s):.1f}]",
            "values": int8_p50s,
        },
        "p50_difference_us": {
            "median": round(float(np.median(diffs)), 1),
            "min": round(float(np.min(diffs)), 1),
            "max": round(float(np.max(diffs)), 1),
            "range": f"[{np.min(diffs):.1f}, {np.max(diffs):.1f}]",
            "values": diffs,
        },
    }

    with open("results/policy_benchmark.json", "w") as f:
        json.dump(main_data, f, indent=2)
    print("Policy benchmark 5x complete.")


def run_5x_rollout_fps():
    print("=== Running Rollout FPS Benchmark 5x ===")
    runs = []
    mujoco_alone_ms = []
    st_fps_list = []
    st_step_ms_list = []
    mt_fps_list = []
    mt_step_ms_list = []

    for run_idx in range(1, 6):
        print(f"  Rollout FPS run {run_idx}/5...")
        res = audit_rollout_fps()

        m_ms = res["pure_mujoco_step_alone_ms"]
        st_fps = res["closed_loop_single_thread_ort"]["effective_fps"]
        st_ms = res["closed_loop_single_thread_ort"]["ms_per_step"]
        mt_fps = res["closed_loop_default_thread_ort"]["effective_fps"]
        mt_ms = res["closed_loop_default_thread_ort"]["ms_per_step"]

        run_summary = {
            "run_index": run_idx,
            "pure_mujoco_step_alone_ms": m_ms,
            "closed_loop_single_thread_ort_fps": st_fps,
            "closed_loop_single_thread_ort_ms": st_ms,
            "closed_loop_default_thread_ort_fps": mt_fps,
            "closed_loop_default_thread_ort_ms": mt_ms,
        }
        with open(f"results/runs/rollout_fps_run_{run_idx}.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        runs.append(run_summary)
        mujoco_alone_ms.append(m_ms)
        st_fps_list.append(st_fps)
        st_step_ms_list.append(st_ms)
        mt_fps_list.append(mt_fps)
        mt_step_ms_list.append(mt_ms)

    with open("results/rollout_fps_audit.json", "r") as f:
        main_data = json.load(f)

    main_data["system_execution_state"] = SYSTEM_STATE
    main_data["repeated_runs"] = {
        "n_runs": 5,
        "pure_mujoco_step_alone_ms": {
            "median": round(float(np.median(mujoco_alone_ms)), 3),
            "min": round(float(np.min(mujoco_alone_ms)), 3),
            "max": round(float(np.max(mujoco_alone_ms)), 3),
            "range": f"[{np.min(mujoco_alone_ms):.3f}, {np.max(mujoco_alone_ms):.3f}]",
            "values": mujoco_alone_ms,
        },
        "closed_loop_single_thread_ort_fps": {
            "median": round(float(np.median(st_fps_list)), 1),
            "min": round(float(np.min(st_fps_list)), 1),
            "max": round(float(np.max(st_fps_list)), 1),
            "range": f"[{np.min(st_fps_list):.1f}, {np.max(st_fps_list):.1f}]",
            "values": st_fps_list,
        },
        "closed_loop_single_thread_ort_step_ms": {
            "median": round(float(np.median(st_step_ms_list)), 3),
            "min": round(float(np.min(st_step_ms_list)), 3),
            "max": round(float(np.max(st_step_ms_list)), 3),
            "range": f"[{np.min(st_step_ms_list):.3f}, {np.max(st_step_ms_list):.3f}]",
            "values": st_step_ms_list,
        },
        "closed_loop_default_thread_ort_fps": {
            "median": round(float(np.median(mt_fps_list)), 1),
            "min": round(float(np.min(mt_fps_list)), 1),
            "max": round(float(np.max(mt_fps_list)), 1),
            "range": f"[{np.min(mt_fps_list):.1f}, {np.max(mt_fps_list):.1f}]",
            "values": mt_fps_list,
        },
    }

    with open("results/rollout_fps_audit.json", "w") as f:
        json.dump(main_data, f, indent=2)
    print("Rollout FPS benchmark 5x complete.")


def run_5x_pipeline_latency():
    print("=== Running Pipeline Latency Benchmark 5x ===")
    runs = []
    env_recreates = []
    env_warms = []
    speedups = []
    et_wait_p50s = []
    full_wait_p50s = []

    for run_idx in range(1, 6):
        print(f"  Pipeline latency run {run_idx}/5...")
        # 10 runs per benchmark execution to keep total wall clock reasonable
        res = benchmark_pipeline_latency(n_runs=10)

        erc = res["env_reuse_comparison"]["env_recreate_with_first_render_mean_ms"]
        ew = res["env_reuse_comparison"]["env_warm_reset_with_render_mean_ms"]
        sp = res["env_reuse_comparison"]["speedup_factor"]
        et_p50 = res["early_termination"]["total_user_wait_ms"]["p50"]
        full_p50 = res["full_50_steps"]["total_user_wait_ms"]["p50"]

        run_summary = {
            "run_index": run_idx,
            "env_recreate_mean_ms": erc,
            "env_warm_reset_mean_ms": ew,
            "speedup_factor": sp,
            "early_termination_p50_ms": et_p50,
            "full_50_steps_p50_ms": full_p50,
        }
        with open(f"results/runs/pipeline_run_{run_idx}.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        runs.append(run_summary)
        env_recreates.append(erc)
        env_warms.append(ew)
        speedups.append(sp)
        et_wait_p50s.append(et_p50)
        full_wait_p50s.append(full_p50)

    with open("results/pipeline_latency.json", "r") as f:
        main_data = json.load(f)

    main_data["system_execution_state"] = SYSTEM_STATE
    main_data["repeated_runs"] = {
        "n_runs": 5,
        "env_recreate_mean_ms": {
            "median": round(float(np.median(env_recreates)), 1),
            "min": round(float(np.min(env_recreates)), 1),
            "max": round(float(np.max(env_recreates)), 1),
            "range": f"[{np.min(env_recreates):.1f}, {np.max(env_recreates):.1f}]",
            "values": env_recreates,
        },
        "env_warm_reset_mean_ms": {
            "median": round(float(np.median(env_warms)), 1),
            "min": round(float(np.min(env_warms)), 1),
            "max": round(float(np.max(env_warms)), 1),
            "range": f"[{np.min(env_warms):.1f}, {np.max(env_warms):.1f}]",
            "values": env_warms,
        },
        "speedup_factor": {
            "median": round(float(np.median(speedups)), 1),
            "min": round(float(np.min(speedups)), 1),
            "max": round(float(np.max(speedups)), 1),
            "range": f"[{np.min(speedups):.1f}, {np.max(speedups):.1f}]",
            "values": speedups,
        },
        "early_termination_wait_p50_ms": {
            "median": round(float(np.median(et_wait_p50s)), 1),
            "min": round(float(np.min(et_wait_p50s)), 1),
            "max": round(float(np.max(et_wait_p50s)), 1),
            "range": f"[{np.min(et_wait_p50s):.1f}, {np.max(et_wait_p50s):.1f}]",
            "values": et_wait_p50s,
        },
    }

    with open("results/pipeline_latency.json", "w") as f:
        json.dump(main_data, f, indent=2)
    print("Pipeline latency benchmark 5x complete.")


def run_all_repeated():
    run_5x_policy_benchmark()
    run_5x_rollout_fps()
    run_5x_pipeline_latency()


if __name__ == "__main__":
    run_all_repeated()
