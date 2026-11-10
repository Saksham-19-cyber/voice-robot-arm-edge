# Truth-Reconciliation Audit Summary

This document records the 18 contradictions identified and reconciled during the comprehensive audit pass of the repository. Every fix was driven strictly by reproducibility artifacts in `results/*.json` — no numbers were hand-edited. Full provenance and environment details are documented in [`docs/VERIFIED_CLAIMS.md`](VERIFIED_CLAIMS.md) and [`docs/BENCHMARKS.md`](BENCHMARKS.md).

---

## Reconciliation Table

| # | Topic | Previous (Wrong) Claim | Correct Value | Source File |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Training steps | "~12,000 steps (~8 min)" | **4,000 steps / 363.3 s** | `results/reach_eval.json` |
| 2 | 100% success first reached | Not stated | **Step 1,500** | `results/training_reach.json` |
| 3 | Rollout FPS | 156.9 FPS | **683.5 FPS** (1-thread) / **851.6 FPS** (default) | `results/rollout_fps_audit.json` |
| 4 | Agent held-out NLU intent | 54.0% (old hybrid) | **76.0%** (new hybrid, dev set) | `results/nlu_eval_detailed.json` |
| 5 | Agent held-out NLU e2e | 48.0% (old hybrid) | **68.0%** (new hybrid, dev set) | `results/nlu_eval_detailed.json` |
| 6 | Stretch-v1 steps | 2,000 steps | **1,000 steps / 384.5 s / 0.0% success** | `results/stretch_eval.json` |
| 7 | Cloud deployment status | "not_attempted" | **deployed** (last commit 2026-11-09) | `results/cloud_deploy.json` |
| 8 | ASR memory | "~80-100 MB subtraction" | **+78.0 MB delta** (127.1 MB idle → 205.1 MB ASR loaded) | `results/streamlit_memory_v3.json` |
| 9 | Enum fix mechanism | "cause not established" | **Empirically confirmed**: pybind11 `HINGE == np.int32(3)` → False; `int()` coercion fixes it | Linux CI diagnostic logs |
| 10 | Test count | 16, 18, 27, 28, or 29 | **31 tests** | `pytest tests/ -v` |
| 11 | NLU keyword collisions | `"lift"` refused as manipulation | **Fixed**: `"lift the arm up"` now parsed as reach +z | `src/nlu/rule_parser.py` |
| 12 | Stop coverage | Missing common phrases | **Fixed**: 25 new stop phrases added (+10.0 pp gain on dev set) | `src/nlu/rule_parser.py` |
| 13 | ONNX external data file | Stale `fetch_reach_actor_fp32.onnx.data` present | **Deleted**: model is self-contained | `git rm models/*.data` |
| 14 | Action error (rollout states) | Not reported | **Mean: 0.00726, Max: 0.04207** | `results/action_difference_eval.json` |
| 15 | Action error (random states) | Omitted | **Mean: 0.00774, Max: 0.11362** | `results/action_difference_eval.json` |
| 16 | Env reuse speedup | 692.7 ms / 4.3 ms (163x) | **394.7 ms / 1.2 ms (672.1x median speedup)** | `results/pipeline_latency.json` |
| 17 | Baseline controller | Not compared | **100.0% success, 2.61 mean steps, 0.18 cm final dist** | `results/baseline_controller_eval.json` |
| 18 | Unprofiled causal language | "thread contention", "barrier overhead", "fits in L2 cache" | **Replaced** with "cause not profiled" | `README.md`, `docs/` |

