# Verified Technical Claims & Empirical Benchmark Record

This document is the authoritative record of verified measurements for the Voice-Commanded Edge Robot Arm system. Every metric listed below is backed by reproducible test runs and recorded in a corresponding JSON artifact in `results/`. Use these exact figures in interviews, project presentations, and resumes.

---

## 1. Verified Resume & Interview Claims

| Technical Claim | Verified Metric / Number | Source File | Measurement Environment | Caveats & Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Model Compression** | **3.58x compression** (281.0 KB FP32 &rarr; 78.5 KB INT8) | `results/quantization_eval.json` | Local Disk / Artifact Inspection | Post-training dynamic INT8 quantization via `onnxruntime.quantization`. |
| **Quantization Fidelity** | **100.0% task success** (100/100 episodes, seed 42) | `results/quantization_eval.json` | Windows laptop, Intel Core i5-1235U, GLFW | Equal to PyTorch FP32 and ONNX FP32 baselines across 100 evaluation episodes. |
| **Export Numerical Accuracy** | **$7.79 \times 10^{-7}$ max absolute error** (up to $2.44 \times 10^{-5}$ across random draws) | `results/quantization_eval.json` | Windows laptop, Intel Core i5-1235U | Max absolute error between PyTorch FP32 and ONNX FP32 actors across 1,000 random test states ($< 10^{-4}$ tolerance verified). |
| **Single-Thread Policy Latency** | **85.9 µs p50** (INT8) vs **114.7 µs p50** (FP32) | `results/policy_benchmark.json` | Windows laptop, Intel Core i5-1235U (single thread pinned) | Single-thread pinned microbenchmark (intra_op=1, inter_op=1, 20,000 timed iterations, 200 warm-up). Bootstrap 95% CI on difference: [28.2 µs, 29.4 µs]. |
| **Policy Throughput** | **11,289 FPS** (INT8) vs **8,246 FPS** (FP32) | `results/policy_benchmark.json` | Windows laptop, Intel Core i5-1235U (single thread pinned) | Policy inference only (state forward pass), excluding physics step and graphics rendering. |
| **Closed-Loop Rollout Throughput** | **156.4 FPS** (no render, INT8) | `results/rollout_fps_audit.json` | Windows laptop, Intel Core i5-1235U (single thread pinned) | Full closed-loop physics simulation (obs prep + ONNX inference + MuJoCo physics step) without graphics rendering. |
| **ASR Inference Latency** | **1,337.4 ms p50** (1,392.8 ms mean, 1,915.5 ms p95) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U (`cpu_threads=1`) | faster-whisper `tiny.en` INT8 on 2.0s audio clip. Warm call (model pre-loaded into memory). |
| **ASR Thread Scaling Ablation** | **1,322.8 ms p50** (`cpu_threads=1`) vs **1,740.2 ms p50** (`cpu_threads=4`) | `results/asr_ablation.json` | Windows laptop, Intel Core i5-1235U | Single-thread execution is 417 ms faster p50 due to elimination of thread contention and synchronization on edge CPU. |
| **NLU Parsing Latency** | **2.76 ms p50** (2.66 ms mean) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U | Dual-engine hybrid parser (regex/rule extractor + TF-IDF Logistic Regression fallback + workspace coordinate clipping). |
| **End-to-End User Wait Time** | **1,562.7 ms p50** (1,639.4 ms mean, 2,175.4 ms p95) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Total warm request time: ASR (1,337 ms) + NLU (2.8 ms) + Early-Term Rollout & Render (30.9 ms) + H.264 Video Encode (203.1 ms). |
| **Early Termination Speedup** | **13.0 steps** vs **50.0 steps** (74% rollout step reduction) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Stops at first task success + 10 hold steps. Cuts total user wait time from 2,078.5 ms p50 down to 1,562.7 ms p50. |
| **Environment State Reuse** | **0.9 ms** warm reset vs **488.6 ms** cold recreation (**515x speedup**) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Caching the simulation environment in `st.session_state` eliminates repeated OpenGL context creation and MuJoCo XML parsing. |
| **Application Memory (Linux OSMesa)** | **103.92 MB peak RSS** (1 session) / **193.65 MB peak RSS** (3 concurrent sessions) | `results/streamlit_memory_v2.json` | Linux Ubuntu (CI runner, OSMesa rendering) | Measured on live Streamlit server process tree across 3 sequential presets (1-session) and 3 concurrent sessions each running 3 presets sequentially (simulation rollout + MP4 encoding; ASR not loaded into cache). >80% headroom under 1 GB limit. |
| **Application Memory (Windows)** | **291.7 MB peak working set** (243.6 MB peak RSS) | `results/streamlit_memory.json` | Windows laptop, Intel Core i5-1235U, single session | Measured on live Streamlit server process tree after complete demo run (Whisper loaded, rollout executed, MP4 encoded). |
| **NLU Accuracy (Synthetic Split)** | **97.0% intent accuracy**, **96.0% end-to-end accuracy** | `results/nlu_eval.json` | Local Evaluation (`n_test=100`) | Measured on synthetic template-generated test split. |
| **NLU Accuracy (Agent Held-Out)** | **54.0% intent accuracy**, **48.0% end-to-end accuracy** | `results/nlu_eval_detailed.json` | Local Evaluation (`n=50`, human-style hand-written) | 95% Wilson CI [40.4%, 67.0%]. Evaluates out-of-distribution phrasings not generated by grammar templates. Outperforms majority-class baseline (30.0%). |

---

## 2. Not Measured / Do Not Claim

The following claims are **NOT verified** and must **NOT** be claimed in interviews or portfolio documentation until concrete measurements are executed:

1. **GPU & NVIDIA Jetson Latency / Throughput:**
   * *Status:* Not measured. `notebooks/benchmark_gpu_colab.ipynb` is provided for user Colab execution, but no local GPU or Jetson hardware was used for repo metrics.
   * *Do not claim:* "Sub-millisecond latency on Jetson Nano", "10x faster on GPU", or any specific GPU FPS.
2. **ASR Word Error Rate (WER) on Real Human Speech:**
   * *Status:* Not measured ($N=0$ real recordings). `results/asr_eval.json` explicitly logs `not measured (N=0 real clips)`. `sample_speech.wav` is a synthetic test fixture.
   * *Do not claim:* "95% speech recognition accuracy", "<5% WER", or robustness to accents/background noise on real human voices.
3. **NLU Accuracy on Human-Collected Speech Transcripts:**
   * *Status:* Not measured ($N=0$ human samples). `tests/nlu_heldout_human.json` is an open template.
   * *Do not claim:* "Validated on real human users" or report metrics for `human_written_heldout`.
4. **Cloud Deployment Latency & Server Memory:**
   * *Status:* The app is deployed on Streamlit Community Cloud (Debian trixie, Python 3.11, OSMesa), but runtime latency and memory fields in `results/cloud_deploy.json` remain `null` until measured directly on the Cloud instance.
   * *Do not claim:* "Runs in 150 MB on Cloud" or present Windows memory/latency numbers as Streamlit Cloud performance.
5. **Pick-and-Place Manipulation:**
   * *Status:* Smoke test only (`results/stretch_eval.json`). 4,000 steps explored on CPU, 0.0% success rate.
   * *Do not claim:* "Solves pick-and-place manipulation" or "Generalizable robotic manipulation across tasks". FetchReach-v4 is the only solved environment.

