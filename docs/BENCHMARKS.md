# Detailed Empirical Benchmark Tables

This document records the exact, unrounded measurements produced by benchmark scripts in `src/bench/`. Every number is automatically generated from `results/*.json` via `tools/render_docs.py`.

---

## 1. Policy Convergence, Baseline Comparison & Quantization Parity

*Environment: FetchReach-v4. Measured on Windows laptop, Intel Core i5-1235U, GLFW rendering.*
*Source: `results/reach_eval.json`, `results/quantization_eval.json`, `results/baseline_controller_eval.json`*

| Controller / Policy Engine | Precision | Model Size | Success (<= 5 cm) | Success (<= 2 cm) | Success (<= 1 cm) | Mean Final Dist | Mean Steps to Goal | Platform / Rendering |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Proportional Baseline Controller** | N/A (Linear) | n/a (hand-coded) | **100.0%** | **98.0%** | **91.0%** | **0.18 cm** | 2.61 steps | Windows / GLFW |
| PyTorch SAC Actor | FP32 | 3.2 MB (SB3 .zip) | **100.0%** | **96.0%** | **54.0%** | 1.08 cm | 2.65 ± 0.93 | Windows / GLFW |
| ONNX Runtime | FP32 | 281.0 KB (.onnx) | **100.0%** | **96.0%** | **54.0%** | 1.08 cm | 2.65 ± 0.93 | Windows / GLFW |
| **ONNX Runtime (Edge)** | **INT8 Dynamic** | **78.5 KB (.onnx)** | **100.0%** | **96.0%** | **56.0%** | **1.08 cm** | **2.65 ± 0.93** | Windows / GLFW |

- **Success Threshold Findings:** At the default FetchReach tolerance (<= 5 cm), all policies achieve 100.0% success. At <= 2 cm, INT8 matches FP32 at 96.0%. At the tightest <= 1 cm threshold, INT8 achieves 56.0% (vs FP32 54.0%), while the analytical proportional baseline reaches 91.0% due to continuous linear velocity correction.
- **Baseline Controller Comparison:** FetchReach-v4 is an obstacle-free kinematic positioning task. The proportional controller achieves 100.0% success (<= 5 cm), 98.0% (<= 2 cm), and 91.0% (<= 1 cm) with mean final distance 0.18 cm. The learned SAC policy achieves 100.0% (<= 5 cm), 96.0% (<= 2 cm), and 56.0% (<= 1 cm) with mean final distance 1.08 cm. The proportional controller is more precise due to direct analytical error minimization; the SAC policy operates tabula rasa without manual gain tuning.
- **Training Convergence:** Converges to 100.0% success at step 1,500; total training budget 4,000 steps completed in 363.27 s (~6.1 min on CPU).
- **Max Absolute Action Difference:** PyTorch vs ONNX FP32: `7.79e-07`; ONNX FP32 vs INT8: `0.1640` across 1,000 random inputs.

---

## 2. High-Precision Policy Latency Microbenchmark

*Single-thread pinned microbenchmark (`intra_op_num_threads=1, inter_op_num_threads=1`, 200 warm-up calls, 20,000 timed iterations per run across 5 repeated runs under High Performance power plan).*
*Source: `results/policy_benchmark.json`*

| Run Number | ONNX FP32 p50 (µs) | ONNX INT8 p50 (µs) | Difference (FP32 minus INT8) (µs) |
| :--- | :--- | :--- | :--- |
| Run 1 | 61.9 µs | 45.4 µs | +16.5 µs |
| Run 2 | 44.4 µs | 50.4 µs | -6.0 µs |
| Run 3 | 34.7 µs | 38.6 µs | -3.9 µs |
| Run 4 | 25.4 µs | 42.6 µs | -17.2 µs |
| Run 5 | 26.6 µs | 38.4 µs | -11.8 µs |
| **5-Run Median** | **34.7 µs** | **42.6 µs** | **-6.0 µs** |

- **Latency Finding:** 3.58x smaller; no measurable latency difference on this CPU; run-to-run variation exceeds the difference; cause not profiled. (Direction: FP32 minus INT8, 5-run median difference: -6.0 µs, range [-17.2, 16.5] µs spanning zero).

---

## 3. Rollout FPS, Physics Step Audit & Step Time Breakdown

*Measured on Windows laptop, Intel Core i5-1235U. Source: `results/rollout_fps_audit.json`, `results/step_breakdown.json`*

| Component / Mode | Execution Details | 5-Run Median Value | 5-Run Min-Max Range |
| :--- | :--- | :--- | :--- |
| **Pure MuJoCo Physics Step** | Raw 20 substeps of `mj_step` alone (no wrappers, no obs formatting, no inference) | **0.907 ms/step** | [0.702, 0.973] ms/step |
| **Closed-Loop Rollout (1 Thread)** | Full simulation loop (`intra_op=1, inter_op=1`, no render) | **580.2 FPS** | [527.0, 760.2] FPS |
| **Closed-Loop Rollout (Default Threads)** | Full simulation loop (unconstrained threads, no render) | **609.6 FPS** | [478.4, 703.4] FPS |

### Detailed Step Time Breakdown (`results/step_breakdown.json`)
*Single-thread INT8 closed-loop step (1.724 ms total median):*
- **MuJoCo Physics (20 substeps):** 0.907 ms (52.6%)
- **Gymnasium Environment Wrapper & Obs Dict Building:** 0.656 ms (38.1%)
- **ONNX Policy Forward Pass:** 0.118 ms (6.8%)
- **Observation Array Formatting & Flattening:** 0.009 ms (0.5%)
- **Closure:** Sum of component medians is 1.690 ms (closure difference: 2.0% from closed-loop step, well within 5%).

---

## 4. End-to-End Request Latency & Pipeline Stages

*Measured on Windows laptop, Intel Core i5-1235U, GLFW rendering. Source: `results/pipeline_latency.json`*

| Pipeline Stage | Implementation | Median Latency (p50) | Mean Latency | 95th Percentile (p95) |
| :--- | :--- | :--- | :--- | :--- |
| **ASR stage in pipeline** | faster-whisper `tiny.en` INT8 (`cpu_threads=1`) | 1639.9 ms | 1645.2 ms | 2169.6 ms |
| **NLU Intent Parsing** | Dual-engine hybrid (Regex + Classifier fallback) | 3.27 ms | 4.19 ms | 9.25 ms |
| **Rollout + Visual Render** | Early termination (13.0 steps) @ 320x320 | 34.9 ms | 37.0 ms | 53.7 ms |
| **MP4 Video Encoding** | `imageio-ffmpeg` H.264 (`-movflags +faststart`) | 329.3 ms | 330.2 ms | 457.9 ms |
| **Total User Wait Time** | End-to-end request latency | **2008.5 ms** | 2017.8 ms | 2687.8 ms |

- **Repeated Runs User Wait Time:** Across 5 independent runs, warm early-termination wait time has median **1597.6 ms** (range: [1220.7, 2008.5] ms).
- **Environment State Reuse:** Reusing the environment in `st.session_state` yields **1.2 ms** warm reset (range: [0.5, 1.2] ms) vs **394.7 ms** cold recreation (range: [342.6, 817.6] ms), yielding a **672x speedup** (range: [289.1, 794.1]x).

---

## 5. ASR Model & Thread Scaling Ablation (`transcribe()` only)

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

## 6. NLU Dual-Engine Accuracy Evaluation & Ablation

*Wilson 95% confidence intervals reported. Source: `results/nlu_eval_detailed.json`*

| Evaluation Split | Sample Count | Rule Parser Intent Acc | Classifier Alone Intent Acc | Hybrid Parser Intent Acc | Hybrid End-to-End Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Template Split (tuned, not held-out)** | N=80 | 97.5% | 100.0% | **100.0%** | **100.0%** |
| **Development Set (Agent-Written; tuned on dev set; not independent)** | N=50 | 60.0% | 70.0% | **76.0%** | **68.0%** |
| **Human-Written Held-Out (`tests/nlu_heldout_human.json`)** | N=0 | not measured | not measured | not measured | not measured |

### Stop-List Expansion Ablation (Dev Set, N=50)
- **Without Stop-List Expansion:** Hybrid Intent: 66.0% (Wilson CI: [52.2%, 77.6%]), End-to-End: 58.0% (Wilson CI: [44.2%, 70.6%])
- **With Stop-List Expansion (25 phrases):** Hybrid Intent: 76.0% (Wilson CI: [62.6%, 85.7%]), End-to-End: 68.0% (Wilson CI: [54.2%, 79.2%])
- **Effect:** Effect of stop list on development set: contributed +10.0 pp absolute accuracy gain to both intent and end-to-end metrics on the development set.
- **Frozen Protocol:** Initial parser baseline frozen at tag `nlu-frozen-v1`. Following audit pass 3 fixes (resolving regression probes, unit conversions, stop priority, and goal invention avoidance), the updated parser is tagged as `nlu-frozen-v2`. The human held-out set (`tests/nlu_heldout_human.json`) must be evaluated once on tag `nlu-frozen-v2` with no tuning afterwards.

---

## 7. Goal Range Audit & Policy Success on NLU Goals

*Source: `results/goal_range.json`, `results/nlu_goal_eval.json`*

- **FetchReach Training Goal Bounds:**
  - x: `[1.1918, 1.4918]`
  - y: `[0.5991, 0.8991]`
  - z: `[0.3847, 0.6847]`
- **Out-of-Range NLU Goals Found:** 12 commands produce coordinates beyond the environment training bounds prior to clipping.
- **NLU Goal Closed-Loop Success Rate:** **100.0%** across all 33 goal categories (mean steps to goal: 2.97, mean final distance: 1.08 cm).

---

## 8. Action Error & Parity Evaluation

*Source: `results/action_difference_eval.json`*

| Evaluation Set | Number of States | Mean Absolute Action Error | Max Absolute Action Error |
| :--- | :--- | :--- | :--- |
| **Random Observation States** | 1000 | 0.00774 | 0.11362 |
| **Rollout Visited States** | 1000 | 0.00726 | 0.04207 |

---

## 9. Server Process Memory Footprint (By Operating System)

*Source: `results/streamlit_memory_v3.json`, `results/streamlit_memory_v4.json`*

### Linux Ubuntu (CI Runner, Headless OSMesa)

*Measured on Linux CI runner in headless server mode using OSMesa software rendering (`results/streamlit_memory_v4.json`). Evaluates the server Python process tree across 3 concurrent sessions each executing 3 presets plus 1 real audio transcription with faster-whisper INT8 loaded.*

| Benchmark Condition | Idle Server RSS | Peak RSS | Peak Working Set | Headroom Under 1 GB |
| :--- | :--- | :--- | :--- | :--- |
| Headless Server Baseline (OSMesa, no sessions) | 54.28 MB | 54.28 MB | 54.28 MB | >94% |
| 1 session (3 presets + 1 real transcription, faster-whisper INT8) | 54.28 MB | 88.88 MB | 88.88 MB | >90% |
| 3 concurrent sessions (3 presets + 1 real transcription, faster-whisper INT8) | 54.28 MB | **185.59 MB** | **185.59 MB** | >80% |

### Windows Laptop (Interactive Client, GLFW)

*Measured on Windows laptop, Intel Core i5-1235U, interactive browser session with hardware-accelerated GLFW rendering (`results/streamlit_memory_v3.json`).*

| Benchmark Condition | Idle Connected RSS | Peak RSS | Peak Working Set | Headroom Under 1 GB |
| :--- | :--- | :--- | :--- | :--- |
| Idle connected client session (GLFW, WebSocket connected) | 127.1 MB | 127.1 MB | 127.1 MB | >87% |
| Active browser session with faster-whisper tiny.en INT8 loaded | 127.1 MB | **205.1 MB** (+78.0 MB attributable ASR delta) | 291.7 MB | >70% |

- **What Each Measurement Means:**
  - **Linux Baseline (54.28 MB):** The raw headless Streamlit Python process on Linux before any client connects, using OSMesa software rendering.
  - **Linux 3 Concurrent Sessions + Real Transcription (185.59 MB peak):** Peak memory when 3 concurrent simulated sessions each execute 3 arm motion presets and 1 real audio transcription with `faster-whisper` INT8 loaded.
  - **Windows Idle Connected Baseline (127.1 MB):** Reflects a full interactive browser session with active WebSocket client connection, Streamlit frontend state, and initialized GLFW OpenGL context.
  - **Windows Active with ASR Cache (205.1 MB peak RSS, 291.7 MB working set):** Includes the connected browser session plus the dynamically cached `faster-whisper` `tiny.en` INT8 model (+78.0 MB attributable delta).

---

## 10. Exploratory Stretch Goal: FetchPickAndPlace-v4

*Source: `results/stretch_eval.json`*

- **Environment:** `FetchPickAndPlace-v4`
- **Timesteps Budget:** `1000` steps on CPU (384.5 s)
- **Final Success Rate:** `0.0%`
- **Assessment:** Smoke test only, not a real attempt. Confirmed environment initialization and action piping under SAC on CPU.
