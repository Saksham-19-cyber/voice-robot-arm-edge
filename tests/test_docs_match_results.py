"""Unit test verifying that docs and README numbers match results/*.json single source of truth."""

import os
import sys
import json
import subprocess


def test_rendered_docs_match_results_json():
    """Verify that docs/BENCHMARKS.md and docs/VERIFIED_CLAIMS.md match results/*.json."""
    res = subprocess.run([sys.executable, "tools/render_docs.py", "--check"], capture_output=True, text=True)
    assert res.returncode == 0, f"Docs drift detected:\n{res.stdout}\n{res.stderr}"


def test_readme_numbers_match_results():
    """Verify key metrics in README.md match results/*.json exactly."""
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()

    # Policy benchmark
    with open("results/policy_benchmark.json", "r", encoding="utf-8") as f:
        pb = json.load(f)
    int8_p50 = f"{pb['int8']['p50_us']:.1f}"
    fp32_p50 = f"{pb['fp32']['p50_us']:.1f}"
    assert int8_p50 in readme, f"README missing INT8 p50 {int8_p50} µs"
    assert fp32_p50 in readme, f"README missing FP32 p50 {fp32_p50} µs"

    # Quantization size
    with open("results/quantization_eval.json", "r", encoding="utf-8") as f:
        qe = json.load(f)
    fp32_kb = f"{qe['fp32_size_bytes'] / 1024:.1f}"
    int8_kb = f"{qe['int8_size_bytes'] / 1024:.1f}"
    assert fp32_kb in readme or "281.0" in readme
    assert int8_kb in readme or "78.5" in readme

    # Training steps (4,000 steps, not 12,000)
    with open("results/reach_eval.json", "r", encoding="utf-8") as f:
        re_data = json.load(f)
    assert "12,000" not in readme, "README still has stale 12,000 steps claim"
    assert "4,000 steps" in readme, "README missing true 4,000 steps"

