# Linux pybind11 Enum Diagnostics & Compatibility Report

This document records the empirical evidence and root-cause analysis for the MuJoCo / Gymnasium-Robotics enum compatibility bug observed on Linux platforms (Ubuntu 24.04 and Debian Trixie on Streamlit Community Cloud) versus Windows.

---

## 1. Problem Summary

When initializing Gymnasium-Robotics robot environments (such as `FetchReach-v4`) on Linux, environment construction failed during model parsing in `gymnasium_robotics.utils.mujoco_utils.get_joint_type` with:
```python
ValueError: 'np.int32(3)' is not a valid joint type
```
The exact same code succeeded on Windows without errors.

---

## 2. Empirical Evidence from Linux CI (Run 42805685275)

The diagnostic job `Diagnose Linux pybind11 Enum Equality vs Windows` in GitHub Actions workflow `linux-smoke.yml` produced the following log output across MuJoCo versions (3.1.6, 3.2.7, 3.12.0, 3.17.0, and 4.0.0):

```text
=== LINUX ENUM DIAGNOSTICS ===
type(raw_type): <class 'numpy.int32'>
repr(raw_type): np.int32(3)
(raw_type == HINGE): True
(HINGE == raw_type): False
(raw_type in (HINGE, SLIDE)): False
type(int(raw_type)): <class 'int'>
repr(int(raw_type)): 3
(int(raw_type) == HINGE): True
(HINGE == int(raw_type)): True
(int(raw_type) in (HINGE, SLIDE)): True
```

---

## 3. Root Cause Analysis

In MuJoCo's Linux wheels, `model.jnt_type` returns an array of NumPy scalar integers (`numpy.int32`).
Pybind11 exposes C++ enums such as `mujoco.mjtJoint.mjJNT_HINGE` with custom equality operators (`__eq__`):
1. **Asymmetric Equality:**
   - `numpy.int32(3) == mujoco.mjtJoint.mjJNT_HINGE` dispatches to NumPy's `__eq__`, which compares the underlying integer value (3 == 3) and evaluates to `True`.
   - `mujoco.mjtJoint.mjJNT_HINGE == numpy.int32(3)` dispatches to pybind11's enum `__eq__`, which performs strict C++ type checking against the enum type. Because `numpy.int32` is not an instance of `mujoco.mjtJoint`, pybind11 returns `False`.
2. **Tuple Membership Failure:**
   In Python, `raw_type in (HINGE, SLIDE)` iterates through the container and evaluates `item == raw_type` (i.e. `HINGE == raw_type`). Because of the pybind11 equality asymmetry, `HINGE == raw_type` evaluates to `False`, causing membership tests to fail!

---

## 4. Monkey-Patch Solution (`src/rl/env_utils.py`)

Coercing the NumPy scalar integer to a standard Python `int` (`int(raw_type)`) restores symmetric equality:
- `int(raw_type) == HINGE` &rarr; `True`
- `HINGE == int(raw_type)` &rarr; `True`
- `int(raw_type) in (HINGE, SLIDE)` &rarr; `True`

The patch intercepts `model.jnt_type` and `model.eq_type` in `src/rl/env_utils.py` and returns standard Python integers, ensuring full cross-platform compatibility across Windows, Linux CI, and Streamlit Community Cloud.

