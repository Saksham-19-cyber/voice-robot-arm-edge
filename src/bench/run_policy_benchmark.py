"""
Rigorous CPU Policy Microbenchmark:
- Fixed thread counts (intra_op=1, inter_op=1)
- 200 warm-up calls
- 20,000 timed iterations
- High-resolution timing (perf_counter_ns)
- 95% bootstrap confidence interval on p50 difference
"""

import os
import sys
import time
import json
import numpy as np
import onnxruntime as ort

def benchmark_policy_isolated(model_path: str, n_warmup: int = 200, n_iterations: int = 20000):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    input_name = session.get_inputs()[0].name
    # Fixed dummy input
    np.random.seed(42)
    dummy_input = np.random.randn(1, 16).astype(np.float32)
    
    # 200 warm-up iterations
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy_input})
        
    latencies_us = np.empty(n_iterations, dtype=np.float64)
    
    # Timed iterations
    for i in range(n_iterations):
        t0 = time.perf_counter_ns()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter_ns()
        latencies_us[i] = (t1 - t0) / 1000.0  # convert ns to us
        
    p50 = float(np.percentile(latencies_us, 50))
    p95 = float(np.percentile(latencies_us, 95))
    p99 = float(np.percentile(latencies_us, 99))
    mean = float(np.mean(latencies_us))
    std = float(np.std(latencies_us))
    
    return {
        "p50_us": round(p50, 2),
        "p95_us": round(p95, 2),
        "p99_us": round(p99, 2),
        "mean_us": round(mean, 2),
        "std_us": round(std, 2),
        "min_us": round(float(np.min(latencies_us)), 2),
        "max_us": round(float(np.max(latencies_us)), 2),
        "file_size_bytes": os.path.getsize(model_path),
        "file_size_kb": round(os.path.getsize(model_path) / 1024, 2),
        "latencies_us": latencies_us,
    }

def bootstrap_p50_diff(latencies_a, latencies_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(latencies_a)
    diffs = np.empty(n_boot, dtype=np.float64)
    
    for b in range(n_boot):
        sample_a = rng.choice(latencies_a, size=n, replace=True)
        sample_b = rng.choice(latencies_b, size=n, replace=True)
        diffs[b] = np.percentile(sample_a, 50) - np.percentile(sample_b, 50)
        
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    mean_diff = float(np.mean(diffs))
    
    return {
        "mean_p50_diff_us (fp32 - int8)": round(mean_diff, 3),
        "ci_95_low_us": round(ci_low, 3),
        "ci_95_high_us": round(ci_high, 3),
        "n_boot": n_boot
    }

def run_policy_benchmark():
    fp32_path = "models/fetch_reach_actor_fp32.onnx"
    int8_path = "models/fetch_reach_actor_int8.onnx"
    
    print("Benchmarking FP32 (20,000 iterations, 200 warmup, threads=1)...", flush=True)
    fp32_res = benchmark_policy_isolated(fp32_path, n_warmup=200, n_iterations=20000)
    print(f"FP32: p50={fp32_res['p50_us']} us, p95={fp32_res['p95_us']} us, p99={fp32_res['p99_us']} us", flush=True)
    
    print("Benchmarking INT8 (20,000 iterations, 200 warmup, threads=1)...", flush=True)
    int8_res = benchmark_policy_isolated(int8_path, n_warmup=200, n_iterations=20000)
    print(f"INT8: p50={int8_res['p50_us']} us, p95={int8_res['p95_us']} us, p99={int8_res['p99_us']} us", flush=True)
    
    print("Computing 95% bootstrap confidence interval on p50 difference...", flush=True)
    boot_res = bootstrap_p50_diff(fp32_res["latencies_us"], int8_res["latencies_us"])
    print(f"Bootstrap CI (FP32 p50 - INT8 p50): [{boot_res['ci_95_low_us']}, {boot_res['ci_95_high_us']}] us", flush=True)
    
    # Exclude raw array from json
    del fp32_res["latencies_us"]
    del int8_res["latencies_us"]
    
    # Check whether 0 is in the confidence interval
    zero_in_ci = (boot_res["ci_95_low_us"] <= 0.0 <= boot_res["ci_95_high_us"])
    p50_diff = fp32_res["p50_us"] - int8_res["p50_us"]
    
    if zero_in_ci:
        diff_conclusion = "No statistically measurable latency difference between FP32 and INT8 (95% CI on p50 difference spans 0)."
    else:
        diff_conclusion = f"Measurable difference of {p50_diff:.2f} us (95% CI: [{boot_res['ci_95_low_us']}, {boot_res['ci_95_high_us']}] us)."
        
    print(f"Conclusion: {diff_conclusion}", flush=True)
    
    output = {
        "benchmark_config": {
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
            "execution_mode": "ORT_SEQUENTIAL",
            "n_warmup": 200,
            "n_iterations": 20000,
            "input_dim": 16,
            "output_dim": 4
        },
        "fp32": fp32_res,
        "int8": int8_res,
        "bootstrap_p50_difference": boot_res,
        "zero_in_95_ci": zero_in_ci,
        "conclusion": diff_conclusion,
    }
    
    with open("results/policy_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved results to results/policy_benchmark.json", flush=True)
    return output

if __name__ == "__main__":
    run_policy_benchmark()

