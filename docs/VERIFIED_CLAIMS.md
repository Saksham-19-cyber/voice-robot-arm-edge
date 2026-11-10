# Verified Technical Claims & Empirical Benchmark Record

This document is the authoritative record of verified measurements for the Voice-Commanded Edge Robot Arm system. Every metric listed below is backed by reproducible test runs and recorded in a corresponding JSON artifact in `results/`.

---

## 1. Verified Technical Claims Table

| Technical Claim | Verified Metric / Number | Source File | Measurement Environment | Caveats & Scope | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Compression** | **3.58x compression** (281.0 KB FP32 &rarr; 78.5 KB INT8) | `results/quantization_eval.json` | Local Disk / Artifact Inspection | Post-training dynamic INT8 quantization via `onnxruntime.quantization`. | `reconciled` |
| **Quantization Fidelity** | **100.0% task success** (both first-reached <5cm and final step) | `results/quantization_eval.json` | Windows laptop, Intel Core i5-1235U, GLFW | Equal to PyTorch FP32 and ONNX FP32 baselines on default random goals. | `reconciled` |
| **Export Numerical Accuracy** | **7.79e-07 max absolute error** | `results/quantization_eval.json` | Windows laptop, Intel Core i5-1235U | Max absolute error between PyTorch FP32 and ONNX FP32 actors across 1,000 random test states (< 10⁻⁴ tolerance). | `reconciled` |
| **Proportional Baseline Comparison** | **100.0% task success** (mean 2.61 steps, 0.18 cm final dist) | `results/baseline_controller_eval.json` | Windows laptop, Intel Core i5-1235U, GLFW | Obstacle-free kinematic task is solvable by proportional control; learned policy matches 100% success. | `reconciled` |
| **Single-Thread Policy Latency** | **85.9 µs p50** (single run); **57.0 µs median** (5 runs, range [27.4, 66.0] µs) | `results/policy_benchmark.json` | Windows laptop, Intel Core i5-1235U (single thread pinned) | Single-thread pinned microbenchmark (intra_op=1, inter_op=1, 20,000 iterations, 200 warm-up). Bootstrap 95% CI: [28.2 µs, 29.4 µs]. | `reconciled` |
| **Policy Throughput** | **11,289 FPS** (INT8) vs **8,246 FPS** (FP32) | `results/policy_benchmark.json` | Windows laptop, Intel Core i5-1235U (single thread pinned) | Policy forward pass only, excluding physics step and graphics rendering. | `reconciled` |
| **Closed-Loop Rollout Throughput** | **428.5 FPS median** (range [400.7, 475.7] FPS) single-thread | `results/rollout_fps_audit.json` | Windows laptop, Intel Core i5-1235U | Full closed-loop physics simulation (obs prep + ONNX forward + MuJoCo physics step) without graphics rendering. | `reconciled` |
| **ASR Stage in Pipeline** | **1337.4 ms p50** (1392.8 ms mean, 1915.5 ms p95) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U (`cpu_threads=1`) | faster-whisper `tiny.en` INT8 on 2.0s audio clip. Includes WAV read, RMS energy calculation, PyAV decode, VAD, and text iteration. | `reconciled` |
| **ASR Transcribe() Only** | **992.8 ms p50** (`cpu_threads=1`) vs **1730.4 ms p50** (`cpu_threads=4`) | `results/asr_ablation.json` | Windows laptop, Intel Core i5-1235U | Isolated model transcription on pre-decoded audio. Single-thread is 737.6 ms faster p50 (cause not profiled; measured difference under production VAD). | `reconciled` |
| **NLU Parsing Latency** | **2.76 ms p50** (2.66 ms mean) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U | Dual-engine hybrid parser (regex/rule extractor + TF-IDF Logistic Regression fallback + workspace coordinate clipping). | `reconciled` |
| **End-to-End User Wait Time** | **1562.7 ms p50** (single run); **1958.5 ms median** (5 runs) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Total warm request time: ASR + NLU + Rollout & Render + H.264 Video Encode. | `reconciled` |
| **Early Termination Speedup** | **13.0 steps** vs **50.0 steps** (74% rollout step reduction) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Stops at first task success + 10 hold steps. Cuts total user wait time from 2,078.5 ms p50 down to 1,562.7 ms p50. | `reconciled` |
| **Environment State Reuse** | **1.0 ms median** warm reset vs **488.5 ms median** cold recreation (**584x speedup**) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Caching the simulation environment in `st.session_state` avoids repeated OpenGL context creation and MuJoCo XML parsing. Range: [333.8x, 874.0x]. | `reconciled` |
| **Application Memory (Linux OSMesa)** | **103.92 MB peak RSS** (1 session) / **187.06 MB peak RSS** (3 concurrent sessions) | `results/streamlit_memory_v2.json`, `results/streamlit_memory_v3.json` | Linux Ubuntu (CI runner, OSMesa rendering) | Measured on live Streamlit server process tree across 3 sequential presets (1-session) and 3 concurrent sessions with faster-whisper loaded. >80% headroom under 1 GB limit. | `reconciled` |
| **Application Memory (Windows Active)** | **205.1 MB peak RSS** (127.1 MB idle server baseline) | `results/streamlit_memory_v3.json` | Windows laptop, Intel Core i5-1235U, live browser | +78.0 MB attributable memory delta when loading faster-whisper tiny.en INT8 into memory cache. | `reconciled` |
| **NLU Accuracy (Synthetic Split)** | **100.0% intent accuracy**, **100.0% end-to-end accuracy** | `results/nlu_eval_detailed.json` | Local Evaluation (`n=80`) | Measured on synthetic template-generated test split (seed=42, test_size=0.2). | `reconciled` |
| **NLU Accuracy (Development Set)** | **76.0% intent accuracy**, **68.0% end-to-end accuracy** | `results/nlu_eval_detailed.json` | Local Evaluation (`n=50`, agent-written) | Development set (agent-written; parser was tuned while inspecting it; not independent). Wilson CI: intent [62.6%, 85.7%], e2e [54.2%, 79.2%]. | `dev set; tuned` |
| **Step Time Breakdown** | **51.8% physics**, **39.2% wrapper glue**, **8.4% ONNX policy** | `results/step_breakdown.json` | Windows laptop, Intel Core i5-1235U | Of 1.604 ms closed-loop step: 0.831 ms MuJoCo (20 substeps), 0.630 ms Gymnasium wrapper, 0.135 ms ONNX policy forward pass. | `reconciled` |
| **Action Error (Rollout States)** | **Mean error 0.00726**, **Max error 0.04207** | `results/action_difference_eval.json` | Windows laptop, Intel Core i5-1235U | Absolute action difference between FP32 and INT8 policies across 1,000 rollout-visited states. | `reconciled` |
| **Action Error (Random States)** | **Mean error 0.00774**, **Max error 0.11362** | `results/action_difference_eval.json` | Windows laptop, Intel Core i5-1235U | Absolute action difference between FP32 and INT8 policies across 1,000 random state draws. | `reconciled` |

---

## 2. Not Measured / Do Not Claim

The following claims are **NOT verified** and must **NOT** be claimed in interviews or portfolio documentation until concrete measurements are executed:

1. **GPU & NVIDIA Jetson Latency / Throughput:**
   * *Status:* Not measured. `notebooks/benchmark_gpu_colab.ipynb` is provided for user Colab execution, but no local GPU or Jetson hardware was used for repo metrics.
   * *Do not claim:* "Sub-millisecond latency on Jetson Nano", "10x faster on GPU", or any specific GPU FPS.
2. **ASR Word Error Rate (WER) on Real Human Speech:**
   * *Status:* Not measured ($N=0$ real recordings). `sample_speech.wav` is a synthetic test fixture.
   * *Do not claim:* "95% speech recognition accuracy", "<5% WER", or robustness to accents/background noise on real human voices.
3. **NLU Accuracy on Human-Collected Speech Transcripts:**
   * *Status:* Not measured ($N=0$ human samples). `tests/nlu_heldout_human.json` is an open template.
   * *Do not claim:* "Validated on real human users" or report metrics for `human_written_heldout`.
4. **Cloud Deployment Latency & Server Memory:**
   * *Status:* The app is deployed on Streamlit Community Cloud (`https://voice-robot-arm-edge-48q2t6rryznf97ypfkd59k.streamlit.app/`), but runtime latency and memory fields in `results/cloud_deploy.json` remain `null` until measured directly on the live app instance.
   * *Do not claim:* Specific Cloud FPS or memory numbers until verified.
5. **Pick-and-Place Manipulation:**
   * *Status:* Smoke test only (`results/stretch_eval.json`). 1000 steps explored on CPU, 0.0% success rate.
   * *Do not claim:* "Solves pick-and-place manipulation" or "Generalizable robotic manipulation across tasks". FetchReach-v4 is the only solved environment.
