# Detailed Empirical Benchmark Tables

This document records the exact, unrounded measurements produced by benchmark scripts in `src/bench/`. Every number is automatically generated from `results/*.json` via `tools/render_docs.py`.

---

## 1. Policy Convergence & Quantization Parity

*Environment: FetchReach-v4. Measured on Windows laptop, Intel Core i5-1235U, GLFW rendering.*
*Source: `results/reach_eval.json`, `results/quantization_eval.json`*

| Policy Engine | Precision | Model Size | 100-Ep Success Rate | Mean Steps to Goal | Platform / Rendering |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch SAC Actor | FP32 | 3.2 MB (.zip) | **100.0%** | 2.65 ± 0.93 | Windows / GLFW |
| ONNX Runtime | FP32 | 281.0 KB | **100.0%** | 2.65 ± 0.93 | Windows / GLFW |
| **ONNX Runtime (Edge)** | **INT8 Dynamic** | **78.5 KB** | **100.0%** | **2.65 ± 0.93** | Windows / GLFW |

- **Training Convergence:** Converges to 100.0% success at step 1,500; total training budget 4,000 steps completed in 363.27 s (~6.1 min on CPU).
- **Max Absolute Action Difference:** PyTorch vs ONNX FP32: `7.79e-07`; ONNX FP32 vs INT8: `0.1640` across 1,000 random inputs.

---

## 2. High-Precision Policy Latency Microbenchmark

*Single-thread pinned microbenchmark (`intra_op_num_threads=1, inter_op_num_threads=1`, 200 warm-up calls, 20,000 timed iterations).*
*Source: `results/policy_benchmark.json`*

| Model Precision | Model Size | Mean Latency | Median (p50) | 95th Percentile (p95) | 99th Percentile (p99) | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ONNX FP32** | 281.0 KB | 121.3 µs | 114.7 µs | 168.3 µs | 217.3 µs | 8,245.9 FPS |
| **ONNX INT8** | 78.5 KB | 88.6 µs | 85.9 µs | 108.9 µs | 156.4 µs | 11,289.4 FPS |

- **Bootstrap 95% Confidence Interval (FP32 p50 − INT8 p50):** `[28.2 µs, 29.4 µs]` (2,000 resamples).
- **Latency Difference Finding:** The single-thread INT8 speedup is outside the benchmark noise. INT8 p95 and p99 also show measurable improvements over FP32 under pinned thread allocation.

---

## 3. Rollout FPS & Physics Step Audit

*Measured on Windows laptop, Intel Core i5-1235U. Source: `results/rollout_fps_audit.json`*

| Component / Mode | Execution Details | Measured Throughput / Time |
| :--- | :--- | :--- |
| **Pure MuJoCo Physics Step** | `env.step()` without observation formatting or ONNX forward pass | 1.816 ms / step |
| **Observation Formatting** | Dict extraction and flat concatenate (16 dims) | 0.0043 ms / step |
| **Closed-Loop Rollout (1 Thread)** | Full simulation loop (`intra_op=1, inter_op=1`, no render) | **223.7 FPS** (4.469 ms/step) |
| **Closed-Loop Rollout (Default Threads)** | Full simulation loop (unconstrained threads, no render) | **356.0 FPS** (2.809 ms/step) |

---

## 4. End-to-End Request Latency & Pipeline Stages

*Measured on Windows laptop, Intel Core i5-1235U, GLFW rendering. Source: `results/pipeline_latency.json`*

| Pipeline Stage | Implementation | Median Latency (p50) | Mean Latency | 95th Percentile (p95) |
| :--- | :--- | :--- | :--- | :--- |
| **ASR Voice Decoding** | faster-whisper `tiny.en` INT8 (`cpu_threads=1`) | 1337.4 ms | 1392.8 ms | 1915.5 ms |
| **NLU Intent Parsing** | Dual-engine hybrid (Regex + Classifier fallback) | 2.76 ms | 2.66 ms | 4.45 ms |
| **Rollout + Visual Render** | Early termination (13.0 steps) @ 320x320 | 30.9 ms | 33.9 ms | 56.8 ms |
| **MP4 Video Encoding** | `imageio-ffmpeg` H.264 (`-movflags +faststart`) | 203.1 ms | 208.4 ms | 274.9 ms |
| **Total User Wait Time** | End-to-end request latency | **1562.7 ms** | 1639.4 ms | 2175.4 ms |

- **Environment State Reuse:** Reusing the environment in `st.session_state` yields **0.9 ms** warm reset vs **488.6 ms** cold recreation (**515x speedup**).

---

## 5. ASR Model & Thread Scaling Ablation

*Evaluated on 2.0s audio clip (`results/sample_2s.wav`) with production VAD enabled. Source: `results/asr_ablation.json`*

| Whisper Model | Quantization | CPU Threads | Median Latency (p50) | 95th Percentile (p95) | Peak RSS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tiny.en` | `int8` | 1 | 992.8 ms | 1241.9 ms | 198.1 MB |
| `tiny.en` | `int8` | 4 | 1730.4 ms | 2000.6 ms | 216.9 MB |
| `tiny.en` | `float32` | 1 | 1933.1 ms | 2715.2 ms | 412.6 MB |
| `tiny.en` | `float32` | 4 | 2023.8 ms | 2342.9 ms | 394.9 MB |
| `base.en` | `int8` | 1 | 3182.9 ms | 4281.1 ms | 229.7 MB |
| `base.en` | `int8` | 4 | 3000.1 ms | 4073.9 ms | 205.6 MB |
| `base.en` | `float32` | 1 | 6482.2 ms | 10679.3 ms | 376.2 MB |
| `base.en` | `float32` | 4 | 3961.4 ms | 4668.6 ms | 380.2 MB |

---

## 6. NLU Dual-Engine Accuracy Evaluation

*Wilson 95% confidence intervals reported. Source: `results/nlu_eval_detailed.json`*

| Evaluation Split | Sample Count | Rule Parser Intent Acc | Classifier Alone Intent Acc | Hybrid Parser Intent Acc | Hybrid End-to-End Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic Test Split** | N=80 | 97.5% | 100.0% | **100.0%** | 72.5% |
| **Agent-Written Held-Out** | N=50 | 60.0% | 70.0% | **76.0%** | 58.0% |
| **Human-Written Held-Out** | N=0 | not measured | not measured | not measured | not measured |

---

## 7. Goal Range Audit & Policy Success on NLU Goals

*Source: `results/goal_range.json`, `results/nlu_goal_eval.json`*

- **FetchReach Training Goal Bounds:**
  - x: `[1.1918, 1.4918]`
  - y: `[0.5991, 0.8991]`
  - z: `[0.3847, 0.6847]`
- **Out-of-Range NLU Goals Found:** 12 commands produce coordinates beyond the environment training bounds prior to clipping.
- **NLU Goal Closed-Loop Success Rate:** **100.0%** across all 30 goal categories (presets, metric offsets 5-30 cm, and edge clipped goals).

---

## 8. Action Error & Parity Evaluation

*Source: `results/action_difference_eval.json`*

| Evaluation Set | Number of States | Mean Absolute Action Error | Max Absolute Action Error |
| :--- | :--- | :--- | :--- |
| **Random Observation States** | 1000 | 0.00774 | 0.11362 |
| **Rollout Visited States** | 1000 | 0.00726 | 0.04207 |

---

## 9. Server Process Memory Footprint

*Source: `results/streamlit_memory_v2.json`, `results/streamlit_memory.json`*

| Platform / Environment | Benchmark Condition | Peak RSS | Peak Working Set | Headroom Under 1 GB |
| :--- | :--- | :--- | :--- | :--- |
| **Linux Ubuntu (CI Runner)** | 1 session (3 presets, OSMesa) | 103.92 MB | 103.92 MB | >89% |
| **Linux Ubuntu (CI Runner)** | 3 concurrent sessions (3 presets, OSMesa) | **193.65 MB** | **193.65 MB** | >80% |
| **Windows Laptop** | 1 session (full demo, Whisper + GLFW) | 243.6 MB | 291.7 MB | >70% |

---

## 10. Exploratory Stretch Goal: FetchPickAndPlace-v4

*Source: `results/stretch_eval.json`*

- **Environment:** `FetchPickAndPlace-v4`
- **Timesteps Budget:** `1000` steps on CPU (384.5 s)
- **Final Success Rate:** `0.0%`
- **Assessment:** Smoke test only, not a real attempt. Confirmed environment initialization and action piping under SAC on CPU.
