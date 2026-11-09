# Streamlit Community Cloud Deployment Guide

This guide details the exact steps to deploy the **Voice-Commanded Robot Arm with Quantized RL Policy on the Edge** to Streamlit Community Cloud.

---

## ⚠️ Critical Prerequisite: Python 3.11 Selection

Streamlit Community Cloud running on Debian Trixie requires prebuilt binary wheels for PyAV (`av==13.1.0`), which is used by `faster-whisper`.
- On **Python 3.11**, PyPI provides 100% prebuilt binary wheels for Linux x86_64 (`manylinux2014_x86_64`). No C compiler or `pkg-config` is invoked.
- On Python 3.12, `av==13.1.0` has no binary wheels and pip attempts compilation from source, failing with `pkg-config is required for building PyAV`.
- **CRITICAL NOTE:** Streamlit Community Cloud does **not** allow modifying Python versions on existing apps. If your app was deployed using Python 3.12, you **must delete the existing app and redeploy it** to select Python 3.11.

---

## Deployment Parameters

When creating the new app on [share.streamlit.io](https://share.streamlit.io):

| Setting | Value |
| :--- | :--- |
| **Repository** | `Saksham-19-cyber/voice-robot-arm-edge` |
| **Branch** | `main` |
| **Main file path** | `app/app.py` |
| **App URL** | *Default or custom subdomain* |
| **Advanced settings $\to$ Python version** | **`3.11`** *(Mandatory)* |

---

## Build Architecture & Configuration Files

1. **`packages.txt` (Debian Trixie apt packages):**
   ```text
   libgl1
   libglu1-mesa
   libosmesa6
   libosmesa6-dev
   libglfw3
   ffmpeg
   ```
2. **`requirements.txt`:**
   Pinned to verified binary wheels for `cp311-manylinux2014_x86_64` (including `av==13.1.0` and `faster-whisper==1.2.1`).
3. **`.python-version` & `runtime.txt`:**
   Both set to `3.11` / `python-3.11`.
4. **Headless MuJoCo Rendering:**
   `app/app.py` initializes headless OSMesa software rasterization on Linux via:
   ```python
   import os
   os.environ.setdefault("MUJOCO_GL", "osmesa")
   ```

---

## Post-Deployment Verification Checklist

After the cloud build completes, verify the following in the live application:
1. **System Info**: Expand the "ℹ️ System Info" accordion in the sidebar. Verify Python is `3.11.x` and RSS memory is well below the 1.0 GB ceiling (~290 MB peak).
2. **Preset Command**: Click "⬅️ Reach Left" and click "🚀 Execute Command". Confirm that the MuJoCo simulation executes, early terminates within ~11 steps, and displays the rendered rollout video.
3. **Voice Input (Optional)**: If testing on a device with a microphone, test recording or allow faster-whisper `tiny.en` to initialize.
4. **Update `results/cloud_deploy.json`**: Populate the verification fields once deployed.
