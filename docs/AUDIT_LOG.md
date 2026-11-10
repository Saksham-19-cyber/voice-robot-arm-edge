# Truth-Reconciliation Audit Summary

This document records the 18 contradictions identified and reconciled during the comprehensive audit pass of the repository. Every fix was driven strictly by reproducibility artifacts in `results/*.json` — no numbers were hand-edited. Full provenance and environment details are documented in [`docs/VERIFIED_CLAIMS.md`](VERIFIED_CLAIMS.md) and [`docs/BENCHMARKS.md`](BENCHMARKS.md).

---

## Reconciliation Table

| # | Topic | Previous (Wrong) Claim | Correct Value | Source File |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Training steps | "~12,000 steps (~8 min)" | **4,000 steps / 363.3 s** | `results/reach_eval.json` |
| 2 | 100% success first reached | Not stated | **Step 1,500** | `results/training_reach.json` |
| 3 | Rollout FPS | 156.9 FPS | **580.2 FPS** (1-thread) / **609.6 FPS** (default) | `results/rollout_fps_audit.json` |
| 4 | Agent held-out NLU intent | 54.0% (old hybrid) | **76.0%** (new hybrid, dev set) | `results/nlu_eval_detailed.json` |
| 5 | Agent held-out NLU e2e | 48.0% (old hybrid) | **68.0%** (new hybrid, dev set) | `results/nlu_eval_detailed.json` |
| 6 | Stretch-v1 steps | 2,000 steps | **1,000 steps / 384.5 s / 0.0% success** | `results/stretch_eval.json` |
| 7 | Cloud deployment status | "not_attempted" | **deployed** (last commit 2026-11-09) | `results/cloud_deploy.json` |
| 8 | ASR memory | "~80-100 MB subtraction" | **+78.0 MB delta** (127.1 MB idle → 205.1 MB ASR loaded) | `results/streamlit_memory_v3.json` |
| 9 | Enum fix mechanism | "cause not established" | **Empirically confirmed**: pybind11 `HINGE == np.int32(3)` → False; `int()` coercion fixes it | Linux CI diagnostic logs |
| 10 | Test count | 16, 18, 27, 28, 29, or 31 | **32 tests** | `pytest tests/ -v` |
| 11 | NLU keyword collisions | `"lift"` refused as manipulation | **Fixed**: `"lift the arm up"` now parsed as reach +z | `src/nlu/rule_parser.py` |
| 12 | Stop coverage | Missing common phrases | **Fixed**: 25 new stop phrases added (+10.0 pp gain on dev set) | `src/nlu/rule_parser.py` |
| 13 | ONNX external data file | Stale `fetch_reach_actor_fp32.onnx.data` present | **Deleted**: model is self-contained | `git rm models/*.data` |
| 14 | Action error (rollout states) | Not reported | **Mean: 0.00726, Max: 0.04207** | `results/action_difference_eval.json` |
| 15 | Action error (random states) | Omitted | **Mean: 0.00774, Max: 0.11362** | `results/action_difference_eval.json` |
| 16 | Env reuse speedup | 692.7 ms / 4.3 ms (163x) | **394.7 ms / 1.2 ms (672.1x median speedup)** | `results/pipeline_latency.json` |
| 17 | Baseline controller | Not compared | **100.0% success, 2.61 mean steps, 0.18 cm final dist** | `results/baseline_controller_eval.json` |
| 18 | Unprofiled causal language | "thread contention", "barrier overhead", "fits in L2 cache" | **Replaced** with "cause not profiled" | `README.md`, `docs/` |

---

## Audit Pass 3 Reconciliation (fix/audit-pass-3)

| # | Topic | Previous Issue / State | Reconciled Value / Resolution | Source File |
| :--- | :--- | :--- | :--- | :--- |
| 19 | NLU parser regressions & probes | 6 regression probes failed (center goal invention, pick refusal on "can", missing inches/feet, stop precedence) | **Resolved all 6 probes + 24 edge cases** in `tests/nlu_regression_probes.json`; Stop priority enforced; goal invention avoided; tagged `nlu-frozen-v2` | `src/nlu/rule_parser.py`, `src/nlu/parser.py`, `tests/nlu_regression_probes.json` |
| 20 | Baseline controller finding & counts | Static stale text (claimed 98.3% / 5.86 steps) and stale key `nlu_goals_eval_600_episodes` | **Dynamically generated finding** matching variables; key renamed to `nlu_goals_eval_660_episodes` (33 categories * 20 episodes) | `src/bench/evaluate_baseline_controller.py`, `results/baseline_controller_eval.json` |
| 21 | Environment reuse measurement | 20 warm resets with standard timer | **High-precision `perf_counter_ns` over 100 warm resets**, reporting unrounded numbers and ratio of medians | `src/bench/measure_pipeline_latency.py` |
| 22 | Policy benchmark structure | Top-level conclusion cited single-run 28.8 µs diff | **Single run moved to `superseded_single_run`**; conclusion derives strictly from `repeated_runs` (-6.0 µs median difference spanning zero) | `results/policy_benchmark.json` |
| 23 | Timing definitions synchronization | Throughput cited as ~580 FPS in one section vs ~230 FPS in another without distinction | **"Measurement definitions" subsection added** to `docs/BENCHMARKS.md` distinguishing closed-loop policy rollout step vs evaluation harness step | `docs/BENCHMARKS.md`, `tools/render_docs.py` |
| 24 | Allowlist integrity & scope enforcement | Unused number `1337.4 ms` present; no scope checks | **`1337.4 ms` removed**; allowlist split into active numbers and historical quotes scoped strictly to `docs/AUDIT_LOG.md` | `tests/doc_number_allowlist.txt`, `tests/test_docs_match_results.py` |
| 25 | Cloud deployment record tool | Static commit SHA and timestamp | **Tool `tools/update_cloud_deploy_record.py`** syncs `commit_sha` and `last_commit_at` from git metadata | `tools/update_cloud_deploy_record.py`, `results/cloud_deploy.json` |
| 26 | CI enum diagnostics evidence | Diagnostic logs not uploaded as artifacts | **Workflow step added to dump & upload enum diagnostics**; documented pybind11 equality asymmetry in `docs/ENUM_DIAGNOSTICS.md` | `.github/workflows/linux-smoke.yml`, `docs/ENUM_DIAGNOSTICS.md` |
| 27 | Multi-seed training harness | Single-seed training only (seed 42) | **Multi-seed harness `src/rl/train_multiseed.py`** with bootstrap confidence intervals and smoke test suite | `src/rl/train_multiseed.py`, `tests/test_multiseed_smoke.py` |
| 28 | Test suite count | 32 tests passed | **36 tests passed** (added probes test, allowlist integrity test, and 2 multiseed smoke tests) | `pytest tests/ -v` |

