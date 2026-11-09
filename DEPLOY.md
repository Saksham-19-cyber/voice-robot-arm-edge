# Streamlit Community Cloud Deployment Guide

This guide details the exact steps to deploy the **Voice-Commanded Robot Arm with Quantized RL Policy on the Edge** to Streamlit Community Cloud.

---

## 1. Prerequisites
- A **Streamlit Community Cloud account** linked to GitHub ([share.streamlit.io](https://share.streamlit.io)).
- Access to the GitHub repository: `https://github.com/Saksham-19-cyber/voice-robot-arm-edge`.

---

## 2. Setup Instructions
When deploying the app via the Streamlit Community Cloud console:
1. **Repository**: Select `Saksham-19-cyber/voice-robot-arm-edge`.
2. **Branch**: `main`.
3. **Main file path**: `app/app.py`.
4. **App URL**: Choose a custom subdomain or use the default.
5. **Advanced Settings (CRITICAL)**:
   - Click **Advanced Settings** before hitting Deploy.
   - Set **Python version** to **`3.11`**.
   - *Note*: Streamlit Community Cloud does not allow changing Python versions on an existing deployment; if previously deployed on 3.12/3.13, delete the app and redeploy it.

---

## 3. Pinned Packages Explanation
- **`packages.txt` (Debian Trixie apt packages)**:
  ```text
  libgl1
  libglu1-mesa
  libosmesa6
  libosmesa6-dev
  libglfw3
  ffmpeg
  ```
  *Why*: Debian Trixie removed `libgl1-mesa-glx`. The software rasterizer for headless MuJoCo rendering is provided by `libosmesa6` and `libglu1-mesa`. `ffmpeg` is required for video encoding.
- **`av==13.1.0`**:
  *Why*: PyAV is required by `faster-whisper`. Pinned to `13.1.0` because PyPI provides prebuilt `manylinux2014_x86_64` wheels for Python 3.11, avoiding C compiler and `pkg-config` source build failures.
- **`mujoco==4.0.0` & `gymnasium-robotics==1.4.2`**:
  *Why*: Strict exact pins ensure ABI and pybind11 compatibility with headless OSMesa rendering and FetchReach-v4 environment dynamics. All dependencies in `requirements.txt` are pinned with `==` to prevent resolver variance.

---

## 4. Post-Deployment Verification Steps
Once the build completes and the app is running:
1. **Inspect System Info Expander**: Expand "ℹ️ System Info" in the sidebar to verify:
   - Python version is `3.11.x`.
   - `mujoco==4.0.0` and `gymnasium-robotics==1.4.2` are loaded.
   - `MUJOCO_GL` is `osmesa`.
   - RSS memory is well below the 1.0 GB container limit (~104 MB peak single-session, ~194 MB 3 concurrent sessions on Linux CI; ~292 MB peak on Windows).
2. **Preset Command**: Click "⬅️ Reach Left" and click "🚀 Execute Command". Confirm:
   - Physics simulation runs and early-terminates at target.
   - 🎬 Replay tab renders the MP4 video with browser playback.
   - 🖼️ Rollout Frames tab allows scrubbing through recorded steps.
   - 📈 Trajectory & Metrics tab displays distance-to-goal curve.
3. **Safe Refusal Guardrail**: Click "📦 Pick Block (unsupported: safe refusal)". Confirm the warning banner appears stating manipulation is unsupported, and **no rollout simulation is run**.
4. **Emergency Stop**: Click "🛑 Emergency Stop". Confirm arm halt banner appears with **zero motion**.

---

## 5. Known Limitations
- **CPU Software Rasterization (OSMesa)**: Rendering on cloud CPU takes ~19 ms per frame. Cold start MuJoCo context construction takes ~500-700 ms on first request; subsequent requests reuse the cached simulation environment in `st.session_state` (<5 ms reset).
- **First-Request Cold Start**: Loading faster-whisper and initializing the first MuJoCo GL context takes ~13-17 seconds on the very first cold request. All subsequent requests run in ~2.0-2.5 seconds end-to-end.
