# Voice-Commanded Robot Arm with Quantized RL Policy on Edge

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://voice-robot-arm-edge-48q2t6rryznf97ypfkd59k.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Zero PyTorch in Serving](https://img.shields.io/badge/Serving-Zero_PyTorch_(ONNX_Only)-orange.svg)](https://onnxruntime.ai/)
[![Tests: 18 Passed](https://img.shields.io/badge/Tests-18%20Passed-brightgreen.svg)](tests/)

An end-to-end, edge-ready robotics portfolio system that translates natural voice commands into 3D spatial goals and executes them with a goal-conditioned reinforcement learning (SAC) policy inside MuJoCo physics simulation.

🔗 **Live Deployment:** [voice-robot-arm-edge-48q2t6rryznf97ypfkd59k.streamlit.app](https://voice-robot-arm-edge-48q2t6rryznf97ypfkd59k.streamlit.app/)

![Robot Arm Reach Demo](assets/demo_reach_left.gif)

---

## 🏗️ System Architecture & Pipeline

```mermaid
flowchart LR
    A["🎙️ Voice Input / Browser Mic / Text"] --> B["⚡ faster-whisper (tiny.en INT8)"]
    B -->|"Transcript"| C["🧠 NLU Intent Parser (Dual Engine)"]
    C -->|"3D Goal (XYZ)"| D["🤖 MuJoCo FetchReach-v4 Env"]
    D <-->|"State & Action Loop"| E["📉 Quantized ONNX Policy (INT8 Dynamic)"]
    D -->|"Frame Buffer"| F["🎬 Video Encoder (MP4 Faststart)"]
    C -->|"Confirmation Text"| G["🔊 Web Speech API TTS (Client-side)"]
    F --> H["💻 Streamlit Cloud UI"]
    G --> H
```

### 1. Voice Input (ASR)
- **Engine:** `faster-whisper` (`tiny.en` quantized to INT8 on CPU).
- **Cold-Start Model Load Time:** **5.11 s** (measured separately; cached in memory via `@st.cache_resource`).
- **Warm Inference Latency:** **1619.1 ms (p50)** on a 2.0s audio clip (`cpu_threads=1` / `4`).
- **Ablation & Thread Tuning:** Single-thread CPU execution (`cpu_threads=1`) avoids thread scheduling and context-switching overhead on short clips, yielding **1322.8 ms (p50)** vs **1740.2 ms (p50)** with 4 threads (see ASR Ablation Study below).
- **ASR Accuracy Note:** ASR accuracy is **not benchmarked** on real user speech because no labeled human voice dataset is included in this repository ($N=0$ real clips in `tests/asr/manifest.csv`; `sample_speech.wav` is a synthetic formant sine wave used for pipeline integration testing). When tested on 1.4s of silence or white noise, the unconstrained Whisper decoder hallucinates `"You"`.

### 2. Natural Language Understanding (NLU)
- **Dual-Engine Architecture:**
  - **Rule-based semantic parser:** Maps spatial orientations (`left`, `right`, `forward`, `up`, etc.) and metric offsets (`"move up by 10 cm"`) to 3D Cartesian coordinates, and clips coordinates strictly inside the robot's physical workspace bounds.
  - **Machine learning classifier:** Balanced TF-IDF + Logistic Regression model trained on synthetic intent templates with cross-validation.
  - **Unsupported Manipulation Guardrails:** Gripper manipulation commands (`"pick"`, `"grab"`, `"snatch"`, `"release"`) explicitly return `goal_xyz = None` with a clear user notice that the reach policy does not perform gripper manipulation. **No simulation rollout is executed.**
  - **Emergency Stop:** Halts immediately with zero physics motion.

### 3. Policy Execution (Goal-Conditioned SAC on the Edge)
- **Environment:** `gymnasium-robotics` `FetchReach-v4` (16-dimensional observation vector: 10 state + 3 achieved goal + 3 desired goal).
- **Training Algorithm:** Soft Actor-Critic (SAC) + Hindsight Experience Replay (`HerReplayBuffer`) with observation running normalization (`VecNormalize`).
- **Edge Quantization:** Deterministic SAC actor extracted with running normalization parameters baked in directly, exported to ONNX FP32 (281.0 KB), and dynamically quantized to **INT8** (78.5 KB, **3.58x compression**).
- **Zero PyTorch in Serving:** Web serving path runs purely on `onnxruntime` + `gymnasium-robotics`, completely omitting `torch`.

### 4. Headless Visual Rendering & Interactive UI Tabs
- **Rendering:** Headless OSMesa software rasterization on Linux / GLFW on Windows, rendered at 320x320 and encoded via `imageio-ffmpeg` to browser-playable MP4 with `-movflags +faststart`.
- **Environment Reuse:** Reusing the simulation environment in `st.session_state` eliminates the ~690 ms context re-creation overhead, reducing warm reset to **4.3 ms** (163x speedup).
- **Early Termination:** Rollout automatically terminates 10 steps after first reaching the goal (`HOLD_STEPS = 10`), reducing total user wait time from **2357.5 ms** down to **1990.1 ms (p50)**. A toggle in the UI allows users to request the full 50-step rollout if desired.
- **Three UI Visualization Tabs:**
  - 🎬 **Replay:** Streamable MP4 video with autoplay and animated GIF fallback.
  - 🖼️ **Rollout Frames:** Interactive frame scrubber allowing step-by-step frame inspection.
  - 📈 **Trajectory & Metrics:** Real-time plot of Euclidean distance to target over steps, 5 cm threshold line, and 3D end-effector coordinates.

---

## 📊 Measured Evaluation & Benchmark Results

All metrics below are strictly derived from benchmark output files in `results/`.

### 1. Policy Convergence & Steps-to-Goal

> **Definition of Steps-to-Goal:** Mean steps-to-goal is strictly defined as the first step index within an episode where `info['is_success'] == 1.0` is achieved (i.e. Euclidean distance between the gripper end-effector and the target goal is $\le 5$ cm).

Across two separate 100-episode evaluations:
1. **Initial SB3 Training Evaluation (`results/reach_eval.json`):**
   - Environment: `VecNormalize(DummyVecEnv)` with initial environment seed 42 over 100 consecutive episodes.
   - **Success Rate:** **100.0%** (100/100).
   - **Mean Steps to Goal:** **2.85 ± 0.83** steps (min: 1, max: 4).
2. **Post-Export Parity & Quantization Comparison (`results/quantization_eval.json`):**
   - Environment: 100 discrete episode seeds (`seed = 42 + ep` for `ep` $\in [0, 99]$) evaluated identically across PyTorch, ONNX FP32, and ONNX INT8.
   - **Success Rate:** **100.0%** across all three engines.
   - **Mean Steps to Goal:** **2.65 ± 0.93** steps across all three engines.

| Policy Engine | Precision | Model Size | 100-Ep Success Rate | Mean Steps to Goal |
|---|---|---|---|---|
| PyTorch SAC Actor | FP32 | 3.2 MB (.zip) | **100.0%** | 2.65 ± 0.93 |
| ONNX Runtime | FP32 | 281.0 KB | **100.0%** | 2.65 ± 0.93 |
| **ONNX Runtime (Edge)** | **INT8 Dynamic** | **78.5 KB** | **100.0%** | **2.65 ± 0.93** |

Numerical consistency check on 1,000 random inputs:
- PyTorch vs ONNX FP32 max absolute error: **$7.79 \times 10^{-7}$** (tolerance $< 10^{-4}$ verified ✅).

---

### 2. High-Precision CPU Policy Microbenchmark (20,000 Iterations)

To eliminate scheduling jitter, thread contention was controlled by pinning `intra_op_num_threads=1` and `inter_op_num_threads=1` under 200 warm-up calls and 20,000 timed iterations (`results/policy_benchmark.json`):

| Model Precision | Model Size | Mean Latency | Median (p50) | 95th Percentile (p95) | 99th Percentile (p99) | Throughput |
|---|---|---|---|---|---|---|
| **ONNX FP32** | 280.97 KB | 325.06 µs | 255.70 µs | 920.73 µs | 2,693.44 µs | 3,076.4 FPS |
| **ONNX INT8** | 78.46 KB | 127.55 µs | 90.00 µs | 325.00 µs | 732.11 µs | 7,839.9 FPS |

- **Bootstrap 95% Confidence Interval on Median Difference (FP32 p50 − INT8 p50):** `[162.90, 167.75] µs` (2,000 bootstrap resamples).
- **Hardware Context:** Measured on **12th Gen Intel(R) Core(TM) i5-1235U**, single-thread pinned (`intra_op_num_threads=1, inter_op_num_threads=1`).
- **Reconciliation of Per-Step Figures:**
  - *Single-thread pinned microbenchmark:* INT8 p50 is **90.0 µs** vs FP32 p50 **255.7 µs**.
  - *Multi-threaded unconstrained session (`results/bench_cpu.json`):* INT8 p50 was **160.1 µs** vs FP32 p50 **173.9 µs** (difference within noise due to thread pool management overhead on small MLPs).
  - *File Size Compression:* INT8 provides a **3.58x compression** (78.5 KB vs 281.0 KB), fitting comfortably in edge L2/L3 cache.

---

### 3. Closed-Loop Rollout FPS vs Full Request Latency

It is critical to distinguish between raw closed-loop physics stepping and full web application serving:

1. **Closed-Loop Physics + Policy Step (No Rendering, `results/rollout_fps_audit.json`):**
   - Mean Step Time: **4.58 ms / step** (**218.3 FPS**) under default multi-threading.
   - Single-threaded ORT: **7.93 ms / step** (**126.1 FPS**).
   - Physics-only stepping (`mj_step`): **3.53 ms / step**.

2. **Full End-to-End Pipeline Latency (`results/pipeline_latency.json`):**
   Measured over 20 runs each with audio transcription, NLU, physics rollout, 320x320 rendering, and MP4 video encoding:

| Execution Mode | Mean Steps | ASR (p50) | NLU (p50) | Rollout+Render (p50) | MP4 Encode (p50) | **Total User Wait (p50)** | Total Mean ± Std |
|---|---|---|---|---|---|---|---|
| **Cold Start (1st run)** | 11.0 steps | 3119.6 ms | 44.8 ms | 2063.5 ms | 2627.4 ms | **13363.9 ms** | 13.36 s (incl. model loads) |
| **Warm Early-Term (`HOLD=10`)** | **13.1 steps** | **1619.1 ms** | **9.0 ms** | **59.3 ms** | **322.8 ms** | **1990.1 ms** | **2128.6 ± 395.2 ms** |
| Warm Full Trajectory (50 steps) | 50.0 steps | 1529.9 ms | 9.7 ms | 229.5 ms | 637.3 ms | **2357.5 ms** | 2478.4 ± 341.1 ms |

- **Environment Reuse Speedup:** Recreating the MuJoCo OpenGL renderer context takes **692.7 ms** per request. Reusing the environment in `st.session_state` drops warm reset + render to **4.3 ms** (**163.0x speedup**).
- **Early Termination Gain:** Stopping at `first_success_step + 10` saves **367.4 ms** per interaction.

---

### 4. ASR Model & Thread Ablation Study

Evaluated on a fixed 2.0s audio clip across 8 configurations (5 warmup passes, 20 timed iterations each, `results/asr_ablation.json`):

| Model | Precision | CPU Threads | Latency p50 | Latency p95 | Latency Mean ± Std | Process Peak RSS |
|---|---|---|---|---|---|---|
| `tiny.en` | **INT8** | **1** | **1322.8 ms** | **1735.8 ms** | **1380.1 ± 199.5 ms** | **166.5 MB** |
| `tiny.en` | INT8 | 4 | 1740.2 ms | 2499.4 ms | 1862.1 ± 406.1 ms | 140.2 MB |
| `tiny.en` | FP32 | 1 | 1866.4 ms | 2328.3 ms | 1987.5 ± 419.4 ms | 355.1 MB |
| `tiny.en` | FP32 | 4 | 2060.5 ms | 2390.6 ms | 2121.8 ± 189.5 ms | 309.5 MB |
| `base.en` | INT8 | 1 | 2990.0 ms | 4664.6 ms | 3375.2 ± 775.1 ms | 150.9 MB |
| `base.en` | INT8 | 4 | 2930.3 ms | 3402.2 ms | 2883.5 ± 405.8 ms | 161.7 MB |
| `base.en` | FP32 | 1 | 3689.8 ms | 6878.1 ms | 4504.2 ± 1388.9 ms | 379.5 MB |
| `base.en` | FP32 | 4 | 4117.2 ms | 4704.7 ms | 4118.8 ± 419.0 ms | 376.2 MB |

- **Key Insight:** `tiny.en` with `compute_type="int8"` and `cpu_threads=1` yields the lowest latency (1322.8 ms p50) while cutting memory by more than half compared to FP32 (166.5 MB vs 355.1 MB). Multi-threading on `tiny.en` increases latency due to thread barrier overhead.

---

### 5. NLU Evaluation with Wilson Confidence Intervals

Evaluated on synthetic template test split and agent-written out-of-distribution phrasings (`results/nlu_eval_detailed.json`):

| Dataset Split | Samples | Majority Baseline | Rule Parser Intent Acc | Classifier Intent Acc | Hybrid Parser End-to-End Acc |
|---|---|---|---|---|---|
| **Synthetic Test Split** | 90 | 44.4% [34.6%, 54.7%] | **100.0%** [95.9%, 100.0%] | **100.0%** [95.9%, 100.0%] | **100.0%** [95.9%, 100.0%] |
| **Agent-Written Held-Out** (`tests/nlu_heldout_handwritten.json`) | 50 | 30.0% [19.1%, 43.8%] | 52.0% [38.5%, 65.2%] | 70.0% [56.3%, 80.9%] | 48.0% [34.8%, 61.5%] |

> **Evaluation Honesty:** The synthetic test split was generated by the same template pipeline as the training data and achieves 100.0% accuracy. On the agent-written held-out split with unseen phrasings, classifier intent accuracy is **70.0%** [56.3%, 80.9%] and end-to-end coordinate accuracy is **48.0%** [34.8%, 61.5%]. Real-world deployments require collection of natural user recordings (`tests/nlu_heldout_human.json`).

---

### 6. Memory Footprint & Live Process Audit

Measured directly on the live Streamlit server process tree (`results/streamlit_memory.json`):

- **Streamlit Community Cloud Budget:** ~1024 MB (1.0 GB).
- **Measured Peak RSS:** **291.70 MB** (measured via `psutil` on the live Streamlit server process tree after a complete demo execution: faster-whisper INT8 loaded, MuJoCo rollout executed, MP4 video encoded).
- **Peak Working Set:** **291.70 MB**.
- **Available Cloud Headroom:** **~732.3 MB** (~71.5% headroom remaining).
- **Zero PyTorch in Serving:** PyTorch is completely omitted from production requirements.

---

### 7. Stretch Task: `FetchPickAndPlace-v4`

- **Status:** Smoke test only, not a real attempt.
- **Details:** A 1,000-step smoke test on CPU confirmed correct environment initialization, goal conditioning, and action piping under SAC, but yielded **0.0% success rate**. Achieving non-prehensile contact grasping and multi-stage pick-and-place subgoals requires substantial training beyond our CPU budget. FetchReach-v4 is retained as the verified production policy.

---

## ⚙️ Compatibility & Linux Runtime Diagnosis

### MuJoCo PyBind11 Enum Comparison on Linux
Under Linux GCC environments (such as Streamlit Community Cloud running Debian Trixie / Ubuntu), evaluating `np.int32(joint_type) in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)` inside `gymnasium_robotics/utils/mujoco_utils.py` causes an `AssertionError` because pybind11's enum equality comparison returns `False` when comparing pybind11 enums against `numpy.int32` scalars from model arrays under NumPy >= 2.0.

- **Strategy A (Stock Testing):** Multi-version evaluation across `mujoco` 3.x candidates (`3.17.0`, `3.12.0`, `3.2.7`, `3.1.6`) is tracked via the Linux CI workflow (`.github/workflows/linux-smoke.yml`).
- **Strategy B (Module Patch):** `src/rl/env_utils.py` applies a non-invasive import-time compatibility patch to `gymnasium_robotics.utils.mujoco_utils` (`get_joint_qpos`, `set_joint_qpos`, `get_joint_qvel`, `set_joint_qvel`, `reset_mocap_welds`, `reset_mocap2body_xpos`), coercing array values and enum variants to Python `int()`.
- **Reproducibility:** A dedicated test suite (`tests/test_enum_patch.py`) validates that accessor operations execute cleanly without assertion failures.

---

## 🚀 Quickstart & Local Installation

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Saksham-19-cyber/voice-robot-arm-edge.git
cd voice-robot-arm-edge

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Optional) For offline SAC training and ONNX export:*
```bash
pip install -r requirements-train.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Unit Tests

Run the full 16-test suite covering environment smoke lifecycles, preset rollout visual generation, video encoding (`h264`, `yuv420p`), ASR silence handling, NLU bounds, unsupported manipulation intents, handwritten held-out phrasings, and closed-loop ONNX rollouts:

```bash
pytest tests/ -v
```
*(All 16 unit tests pass cleanly in ~40s).*

---

## 🔬 Reproducing Benchmarks

All evaluation figures can be re-generated with the scripts below:
- **CPU Policy Latency (20k iterations, single-thread):** `python src/bench/run_policy_benchmark.py`
- **Rollout FPS Audit:** `python src/bench/audit_rollout_fps.py`
- **Pipeline Latency & Early Termination:** `python src/bench/measure_pipeline_latency.py`
- **ASR Ablation Study (8 configs):** `python src/bench/asr_ablation.py`
- **ASR Evaluation (WER & Provenance Check):** `python src/bench/asr_eval.py`
- **NLU Detailed Evaluation (Wilson CIs):** `python src/bench/nlu_eval.py`
- **Streamlit Server Process Peak RSS:** `python src/bench/measure_streamlit_rss.py`

---

## ☁️ Deployment on Streamlit Community Cloud

Complete deployment instructions are detailed in [`DEPLOY.md`](DEPLOY.md).

> ⚠️ **IMPORTANT:** When deploying on Streamlit Community Cloud, **you must select Python 3.11** under **Advanced Settings**. PyAV (`av==13.1.0`) provides prebuilt binary wheels for Python 3.11 on Debian Trixie (`manylinux2014_x86_64`), avoiding source compilation failures.

---

## 📜 License
MIT License. Created for open-source robotics and edge reinforcement learning portfolio deployment.
