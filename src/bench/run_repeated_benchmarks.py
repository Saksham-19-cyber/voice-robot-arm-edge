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
    print("=== Running Synchronized Rollout FPS and Step Breakdown Benchmarks 5x ===", flush=True)
    from src.rl.env_utils import make_fetch_env
    from src.bench.timing import profile_rollout_step_and_breakdown
    import onnxruntime as ort

    opts1 = ort.SessionOptions()
    opts1.intra_op_num_threads = 1
    opts1.inter_op_num_threads = 1

    runs_st = []
    runs_mt = []

    phys_list = []
    wrap_list = []
    onnx_list = []
    obs_list = []
    total_st_list = []
    fps_st_list = []

    total_mt_list = []
    fps_mt_list = []

    for run_idx in range(1, 6):
        print(f"  Rollout & Step Breakdown run {run_idx}/5...", flush=True)
        env = make_fetch_env("FetchReach-v4", render_mode="rgb_array")
        session_st = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", sess_options=opts1, providers=["CPUExecutionProvider"])
        session_mt = ort.InferenceSession("models/fetch_reach_actor_int8.onnx", providers=["CPUExecutionProvider"])

        res_st = profile_rollout_step_and_breakdown(env, session_st, n_steps=500, n_warmup=50)
        res_mt = profile_rollout_step_and_breakdown(env, session_mt, n_steps=500, n_warmup=50)
        env.close()

        # Save individual run files
        with open(f"results/runs/step_breakdown_run_{run_idx}.json", "w", encoding="utf-8") as f:
            json.dump(res_st, f, indent=2)

        fps_run_data = {
            "run_index": run_idx,
            "pure_mujoco_step_alone_ms": res_st["pure_mujoco_physics_ms"]["median_ms"],
            "closed_loop_single_thread_ort_fps": res_st["closed_loop_step_ms"]["effective_fps"],
            "closed_loop_single_thread_ort_ms": res_st["closed_loop_step_ms"]["median_ms"],
            "closed_loop_default_thread_ort_fps": res_mt["closed_loop_step_ms"]["effective_fps"],
            "closed_loop_default_thread_ort_ms": res_mt["closed_loop_step_ms"]["median_ms"],
        }
        with open(f"results/runs/rollout_fps_run_{run_idx}.json", "w", encoding="utf-8") as f:
            json.dump(fps_run_data, f, indent=2)

        runs_st.append(res_st)
        runs_mt.append(res_mt)

        phys_list.append(res_st["pure_mujoco_physics_ms"]["median_ms"])
        wrap_list.append(res_st["gymnasium_wrapper_ms"]["median_ms"])
        onnx_list.append(res_st["onnx_inference_ms"]["median_ms"])
        obs_list.append(res_st["obs_formatting_ms"]["median_ms"])
        total_st_list.append(res_st["closed_loop_step_ms"]["median_ms"])
        fps_st_list.append(res_st["closed_loop_step_ms"]["effective_fps"])

        total_mt_list.append(res_mt["closed_loop_step_ms"]["median_ms"])
        fps_mt_list.append(res_mt["closed_loop_step_ms"]["effective_fps"])

    med_phys = round(float(np.median(phys_list)), 3)
    med_wrap = round(float(np.median(wrap_list)), 3)
    med_onnx = round(float(np.median(onnx_list)), 3)
    med_obs = round(float(np.median(obs_list)), 3)
    med_total_st = round(float(np.median(total_st_list)), 3)
    med_fps_st = round(float(np.median(fps_st_list)), 1)

    med_total_mt = round(float(np.median(total_mt_list)), 3)
    med_fps_mt = round(float(np.median(fps_mt_list)), 1)

    sum_meds = round(med_phys + med_wrap + med_onnx + med_obs, 3)
    closure_diff_pct = round(abs(sum_meds - med_total_st) / med_total_st * 100, 1)

    pct_phys = round(med_phys / med_total_st * 100, 1)
    pct_wrap = round(med_wrap / med_total_st * 100, 1)
    pct_onnx = round(med_onnx / med_total_st * 100, 1)
    pct_obs = round(med_obs / med_total_st * 100, 1)

    # 1. Write results/step_breakdown.json
    step_breakdown_data = {
        "n_runs": 5,
        "n_iter_per_run": 500,
        "timing_method": "src/bench/timing.py: profile_rollout_step_and_breakdown",
        "definitions": {
            "physics_step": "Exactly 20 substeps of mujoco.mj_step(model, data). No rendering, no wrappers.",
            "wrapper_overhead": "Gymnasium env.step() duration outside raw physics (obs dict, action setup, reward).",
            "policy_inference": "Single-thread pinned ONNX INT8 actor session.run().",
            "obs_formatting": "Concatenation of ob, ag, dg into float32 (1, 16) array.",
            "closed_loop_step": "End-to-end execution of obs formatting + ONNX inference + env.step().",
        },
        "machine_state": SYSTEM_STATE,
        "gl_backend": os.environ.get("MUJOCO_GL", "glfw"),
        "total_closed_loop_step_ms": {
            "median": med_total_st,
            "p50": med_total_st,
            "mean": round(float(np.mean(total_st_list)), 3),
            "min": round(float(np.min(total_st_list)), 3),
            "max": round(float(np.max(total_st_list)), 3),
            "range": f"[{np.min(total_st_list):.3f}, {np.max(total_st_list):.3f}]",
        },
        "stages": {
            "mujoco_physics_20_substeps_ms": {
                "median": med_phys,
                "p50": med_phys,
                "range": f"[{np.min(phys_list):.3f}, {np.max(phys_list):.3f}]",
                "pct_of_total": pct_phys,
            },
            "gymnasium_env_wrapper_and_obs_building_ms": {
                "median": med_wrap,
                "p50": med_wrap,
                "range": f"[{np.min(wrap_list):.3f}, {np.max(wrap_list):.3f}]",
                "pct_of_total": pct_wrap,
            },
            "onnx_policy_inference_ms": {
                "median": med_onnx,
                "p50": med_onnx,
                "range": f"[{np.min(onnx_list):.3f}, {np.max(onnx_list):.3f}]",
                "pct_of_total": pct_onnx,
            },
            "obs_flattening_and_conversion_ms": {
                "median": med_obs,
                "p50": med_obs,
                "range": f"[{np.min(obs_list):.3f}, {np.max(obs_list):.3f}]",
                "pct_of_total": pct_obs,
            },
        },
        "closure": {
            "sum_of_medians_ms": sum_meds,
            "diff_from_closed_loop_pct": closure_diff_pct,
        },
        "individual_runs": runs_st,
    }
    with open("results/step_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(step_breakdown_data, f, indent=2)

    # 2. Write results/rollout_fps_audit.json
    audit_data = {
        "timing_method": "src/bench/timing.py: profile_rollout_step_and_breakdown with synchronized perf_counter_ns",
        "definitions": {
            "pure_mujoco_physics_20_substeps": "Exactly 20 calls to mj_step(model, data). No rendering, no wrappers.",
            "wrapper_overhead": "Gymnasium env.step() duration outside raw physics (obs dict, action setup, reward).",
            "closed_loop_rollout": "Sequential rollout of obs prep + ONNX inference + env.step() without reset overhead.",
        },
        "machine_state": SYSTEM_STATE,
        "gl_backend": os.environ.get("MUJOCO_GL", "glfw"),
        "raw_mujoco_physics_20_substeps_ms": runs_st[0]["pure_mujoco_physics_ms"],
        "pure_mujoco_step_alone_ms": med_phys,
        "closed_loop_single_thread_ort": {
            "total_steps": 500,
            "effective_fps": med_fps_st,
            "ms_per_step": med_total_st,
        },
        "closed_loop_default_thread_ort": {
            "total_steps": 500,
            "effective_fps": med_fps_mt,
            "ms_per_step": med_total_mt,
        },
        "system_execution_state": SYSTEM_STATE,
        "repeated_runs": {
            "n_runs": 5,
            "pure_mujoco_step_alone_ms": {
                "median": med_phys,
                "min": round(float(np.min(phys_list)), 3),
                "max": round(float(np.max(phys_list)), 3),
                "range": f"[{np.min(phys_list):.3f}, {np.max(phys_list):.3f}]",
                "values": phys_list,
            },
            "closed_loop_single_thread_ort_fps": {
                "median": med_fps_st,
                "min": round(float(np.min(fps_st_list)), 1),
                "max": round(float(np.max(fps_st_list)), 1),
                "range": f"[{np.min(fps_st_list):.1f}, {np.max(fps_st_list):.1f}]",
                "values": fps_st_list,
            },
            "closed_loop_single_thread_ort_step_ms": {
                "median": med_total_st,
                "min": round(float(np.min(total_st_list)), 3),
                "max": round(float(np.max(total_st_list)), 3),
                "range": f"[{np.min(total_st_list):.3f}, {np.max(total_st_list):.3f}]",
                "values": total_st_list,
            },
            "closed_loop_default_thread_ort_fps": {
                "median": med_fps_mt,
                "min": round(float(np.min(fps_mt_list)), 1),
                "max": round(float(np.max(fps_mt_list)), 1),
                "range": f"[{np.min(fps_mt_list):.1f}, {np.max(fps_mt_list):.1f}]",
                "values": fps_mt_list,
            },
            "closed_loop_default_thread_ort_step_ms": {
                "median": med_total_mt,
                "min": round(float(np.min(total_mt_list)), 3),
                "max": round(float(np.max(total_mt_list)), 3),
                "range": f"[{np.min(total_mt_list):.3f}, {np.max(total_mt_list):.3f}]",
                "values": total_mt_list,
            },
        },
    }
    with open("results/rollout_fps_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"Synchronized 5x benchmark complete. Median Physics: {med_phys} ms ({pct_phys}%), Median Step: {med_total_st} ms ({med_fps_st} FPS). Closure: {closure_diff_pct}%.", flush=True)


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
