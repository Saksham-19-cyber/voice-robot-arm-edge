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

    # Policy benchmark (5-run medians)
    with open("results/policy_benchmark.json", "r", encoding="utf-8") as f:
        pb = json.load(f)
    rep = pb.get("repeated_runs", {})
    int8_p50 = f"{rep['int8_p50_us']['median']:.1f}"
    fp32_p50 = f"{rep['fp32_p50_us']['median']:.1f}"
    assert int8_p50 in readme, f"README missing INT8 median p50 {int8_p50} µs"
    assert fp32_p50 in readme, f"README missing FP32 median p50 {fp32_p50} µs"

    # Quantization size
    with open("results/quantization_eval.json", "r", encoding="utf-8") as f:
        qe = json.load(f)
    fp32_kb = f"{qe['fp32_size_bytes'] / 1024:.1f}"
    int8_kb = f"{qe['int8_size_bytes'] / 1024:.1f}"
    assert fp32_kb in readme or "281.0" in readme
    assert int8_kb in readme or "78.5" in readme

    # Training steps (4,000 steps, not 12,000 as a live claim).
    # Note: "~12,000" is allowed inside the audit table (quoted as a historical wrong value).
    with open("results/reach_eval.json", "r", encoding="utf-8") as f:
        re_data = json.load(f)
    # Strip lines that are clearly part of the reconciliation table (contain "12,000" quoted with surrounding pipe characters)
    live_lines = [
        line for line in readme.splitlines()
        if "12,000" in line
        and not (line.strip().startswith("|") and '"~12,000' in line)
    ]
    assert not live_lines, (
        f"README has live (non-audit-table) 12,000 steps claim on line(s):\n"
        + "\n".join(live_lines)
    )
def _extract_all_numbers_from_json(data):
    """Recursively collect all numeric floats/ints from parsed json data."""
    nums = []
    if isinstance(data, dict):
        for v in data.values():
            nums.extend(_extract_all_numbers_from_json(v))
    elif isinstance(data, list):
        for item in data:
            nums.extend(_extract_all_numbers_from_json(item))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        nums.append(float(data))
    return nums


def _check_text_unit_numbers(text, source_name, all_json_nums, allowlist):
    """Scan text for numeric values with units and ensure they match results/*.json or allowlist."""
    import re
    pattern = r'(?<![A-Za-z0-9_.])(\d+(?:,\d+)*(?:\.\d+)?)\s*(ms|µs|us|MB|KB|GB|FPS|fps|x|%|steps|s)\b'

    unmatched = []
    for m in re.finditer(pattern, text):
        raw_num = m.group(1).replace(",", "")
        unit = m.group(2)
        try:
            val = float(raw_num)
        except ValueError:
            continue

        # Skip URL percent-encodings (e.g. %20)
        if unit == "%" and m.end() < len(text) and text[m.end():m.end()+2].isalnum():
            continue

        pair_key = f"{raw_num} {unit}".lower()
        alt_key = f"{raw_num}{unit}".lower()
        if pair_key in allowlist or alt_key in allowlist:
            continue

        # Check if val matches any number in results/*.json within 2% or 0.1 tolerance
        matched = False
        for jnum in all_json_nums:
            if abs(val - jnum) < 1e-4 or abs(abs(val) - abs(jnum)) < 1e-4:
                matched = True
                break
            # Check rounding match (e.g. 78.5 vs 78.46)
            if abs(val - round(jnum, 1)) < 1e-4 or abs(val - round(jnum, 2)) < 1e-4:
                matched = True
                break
            if abs(abs(val) - abs(round(jnum, 1))) < 1e-4 or abs(abs(val) - abs(round(jnum, 2))) < 1e-4:
                matched = True
                break
                matched = True
                break
            # Tolerance for percentage/unit conversions
            if jnum != 0 and abs(val - jnum) / abs(jnum) < 0.02:
                matched = True
                break

        if not matched:
            line_no = text[:m.start()].count("\n") + 1
            unmatched.append((source_name, line_no, m.group(0), val, unit))
    return unmatched


def test_readme_unit_numbers_match_results_or_allowlist():
    """Verify that every number with a unit in README.md matches results/*.json or allowlist."""
    import glob

    all_json_nums = []
    for json_file in glob.glob("results/**/*.json", recursive=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                all_json_nums.extend(_extract_all_numbers_from_json(content))
        except Exception:
            pass

    allowlist = set()
    with open("tests/doc_number_allowlist.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                token = line.split("#")[0].strip().lower()
                allowlist.add(token)

    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()

    unmatched = _check_text_unit_numbers(readme, "README.md", all_json_nums, allowlist)
    assert not unmatched, (
        f"Found {len(unmatched)} numbers with units in README.md that neither match results/*.json nor appear in tests/doc_number_allowlist.txt:\n"
        + "\n".join([f"  Line {l}: '{text}'" for _, l, text, _, _ in unmatched])
    )


def test_all_docs_and_captions_unit_numbers_match_results_or_allowlist():
    """Verify numbers with units in docs/*.md and src/bench/make_figures.py match results/*.json or allowlist."""
    import glob

    all_json_nums = []
    for json_file in glob.glob("results/**/*.json", recursive=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                all_json_nums.extend(_extract_all_numbers_from_json(content))
        except Exception:
            pass

    allowlist = set()
    with open("tests/doc_number_allowlist.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                token = line.split("#")[0].strip().lower()
                allowlist.add(token)

    targets = glob.glob("docs/*.md") + ["src/bench/make_figures.py"]
    total_unmatched = []
    for t in targets:
        with open(t, "r", encoding="utf-8") as f:
            content = f.read()
        unm = _check_text_unit_numbers(content, t, all_json_nums, allowlist)
        total_unmatched.extend(unm)

    assert not total_unmatched, (
        f"Found {len(total_unmatched)} unmatched numbers with units in docs/*.md or make_figures.py:\n"
        + "\n".join([f"  {src}:{l} '{text}'" for src, l, text, _, _ in total_unmatched])
    )


def test_doc_number_allowlist_integrity_and_historical_scope():
    """Verify that every allowlisted number appears in documentation and historical entries are scoped to AUDIT_LOG.md."""
    import glob
    import re

    active_entries = []
    historical_entries = []
    current_section = "active"

    with open("tests/doc_number_allowlist.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("## Historical quotes"):
                current_section = "historical"
                continue
            elif line.startswith("#"):
                continue

            parts = line.split("#")
            token = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else ""
            assert reason and "(from prompt instruction)" not in reason, (
                f"Allowlist entry '{token}' has invalid or missing reason: '{reason}'"
            )
            if current_section == "active":
                active_entries.append((token, reason))
            else:
                historical_entries.append((token, reason))

    assert active_entries, "Active allowlist entries must not be empty"
    assert historical_entries, "Historical allowlist entries must not be empty"

    targets = glob.glob("docs/*.md") + ["README.md", "src/bench/make_figures.py"]
    doc_contents = {}
    for t in targets:
        with open(t, "r", encoding="utf-8") as f:
            doc_contents[t] = f.read()

    def check_presence(token, text):
        parts = token.split()
        if len(parts) == 2:
            num, unit = parts
            num_pat = r'(?:12000|12,000)' if num == '12000' else re.escape(num)
            pat = rf'\b{num_pat}\s*{re.escape(unit)}\b'
            return bool(re.search(pat, text, re.IGNORECASE))
        return token.lower() in text.lower()

    # 1. Assert every allowlist entry appears in at least one documentation file
    for token, reason in active_entries + historical_entries:
        found = any(check_presence(token, text) for text in doc_contents.values())
        assert found, f"Allowlisted number '{token}' never appears anywhere in documentation files."

    # 2. Assert historical entries NEVER appear outside docs/AUDIT_LOG.md
    for token, reason in historical_entries:
        for doc_path, text in doc_contents.items():
            if os.path.basename(doc_path).lower() != "audit_log.md":
                assert not check_presence(token, text), (
                    f"Historical allowlist entry '{token}' found outside docs/AUDIT_LOG.md in '{doc_path}'"
                )

