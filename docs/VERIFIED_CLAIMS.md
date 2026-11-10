# Verified Technical Claims & Empirical Benchmark Record

This document is the authoritative record of verified measurements for the Voice-Commanded Edge Robot Arm system. Every metric listed below is backed by reproducible test runs and recorded in a corresponding JSON artifact in `results/`.

---

## 1. Verified Technical Claims Table

| Technical Claim | Verified Metric / Number | Source File | Measurement Environment | Caveats & Scope | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Compression** | **3.58x compression** (281.0 KB FP32 &rarr; 78.5 KB INT8) | `results/quantization_eval.json` | Local Disk / Artifact Inspection | Post-training dynamic INT8 quantization via `onnxruntime.quantization`. | `reconciled` |
| **Quantization Fidelity** | **100.0% (<=5cm)**, **96.0% (<=2cm)**, **56.0% (<=1cm)** | `results/quantization_eval.json` | Windows laptop, Intel Core i5-1235U, GLFW | Equal to FP32 baselines (100% / 96% / 54%) within 1.08 cm mean distance. | `reconciled` |
| **Export Numerical Accuracy** | **7.79e-07 max absolute error** | `results/quantization_eval.json` | Windows laptop, Intel Core i5-1235U | Max absolute error between PyTorch FP32 and ONNX FP32 actors across 1,000 random test states (< 10⁻⁴ tolerance). | `reconciled` |
| **Proportional Baseline Comparison** | **100.0% (<=5cm)**, **98.0% (<=2cm)**, **91.0% (<=1cm)** | `results/baseline_controller_eval.json` | Windows laptop, Intel Core i5-1235U, GLFW | Obstacle-free kinematic task. Baseline is more precise (0.18 cm vs 1.08 cm); learned SAC policy achieves 100% success tabula rasa without manual gain tuning. | `reconciled` |
| **Single-Thread Policy Latency** | **3.58x smaller; no measurable latency difference on this CPU** | `results/policy_benchmark.json` | Windows laptop, Intel Core i5-1235U (single thread pinned) | Single-thread pinned microbenchmark (intra_op=1, inter_op=1, 20,000 iter/run). 5-run median difference: -6.0 µs (range [-17.2, 16.5] µs spanning zero; cause not profiled). | `reconciled` |
| **Closed-Loop Rollout Throughput** | **580.2 FPS median** (range [527.0, 760.3] FPS) single-thread | `results/rollout_fps_audit.json` | Windows laptop, Intel Core i5-1235U | Full closed-loop physics simulation (obs prep + ONNX forward + MuJoCo physics step) without graphics rendering. Default threads: 609.6 FPS median. | `reconciled` |
| **ASR Stage in Pipeline** | **1639.9 ms p50** (1645.2 ms mean, 2169.6 ms p95) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U (`cpu_threads=1`) | faster-whisper `tiny.en` INT8 on 2.0s audio clip. Includes WAV read, RMS energy calculation, PyAV decode, VAD, and text iteration. | `reconciled` |
| **ASR Transcribe() Only** | **992.8 ms p50** (`cpu_threads=1`) vs **1730.4 ms p50** (`cpu_threads=4`) | `results/asr_ablation.json` | Windows laptop, Intel Core i5-1235U | Isolated model transcription on pre-decoded audio. Single-thread is 737.6 ms faster p50 (cause not profiled; measured difference under production VAD). | `reconciled` |
| **NLU Parsing Latency** | **3.27 ms p50** (4.19 ms mean) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U | Dual-engine hybrid parser (regex/rule extractor + TF-IDF Logistic Regression fallback + workspace coordinate clipping). | `reconciled` |
| **End-to-End User Wait Time** | **2008.5 ms p50** (single run); **1597.6 ms median** (5 runs) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Total warm request time: ASR + NLU + Rollout & Render + H.264 Video Encode. Range across runs: [1220.7, 2008.5] ms. | `reconciled` |
| **Early Termination Speedup** | **13.0 steps** vs **50.0 steps** (74% rollout step reduction) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Stops at first task success + 10 hold steps. | `reconciled` |
| **Environment State Reuse** | **1.2 ms median** warm reset vs **394.7 ms median** cold recreation (**672x speedup**) | `results/pipeline_latency.json` | Windows laptop, Intel Core i5-1235U, GLFW rendering | Caching simulation environment in `st.session_state` avoids repeated OpenGL context creation and MuJoCo XML parsing. Range: [289.1x, 794.1x]. | `reconciled` |
| **Application Memory (Linux OSMesa)** | **88.88 MB peak RSS** (1 session) / **185.59 MB peak RSS** (3 concurrent sessions) | `results/streamlit_memory_v4.json` | Linux Ubuntu (CI runner, OSMesa rendering) | Measured on live Streamlit server process tree across 3 sequential presets + real transcription (1-session) and 3 concurrent sessions with faster-whisper loaded. >80% headroom under 1 GB limit. | `reconciled` |
| **Application Memory (Windows Active)** | **205.1 MB peak RSS** (127.1 MB idle server baseline) | `results/streamlit_memory_v3.json` | Windows laptop, Intel Core i5-1235U, live browser | +78.0 MB attributable memory delta when loading faster-whisper tiny.en INT8 into memory cache. | `reconciled` |
| **NLU Accuracy (Template Split)** | **100.0% intent accuracy**, **100.0% end-to-end accuracy** | `results/nlu_eval_detailed.json` | Local Evaluation (`n=80`) | Template split (tuned, not held-out). | `template split` |
| **NLU Accuracy (Development Set)** | **76.0% intent accuracy**, **68.0% end-to-end accuracy** | `results/nlu_eval_detailed.json` | Local Evaluation (`n=50`, agent-written) | Development set (agent-written; parser was tuned while inspecting it; not independent). Wilson CI: intent [62.6%, 85.7%], e2e [54.2%, 79.2%]. | `dev set; tuned` |
| **Step Time Breakdown** | **52.6% physics**, **38.1% wrapper glue**, **6.8% ONNX policy** | `results/step_breakdown.json` | Windows laptop, Intel Core i5-1235U | Of 1.724 ms closed-loop step: 0.907 ms MuJoCo (20 substeps), 0.656 ms Gymnasium wrapper, 0.118 ms ONNX policy forward pass, 0.009 ms obs format. Closure diff 2.0%. | `reconciled` |
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
