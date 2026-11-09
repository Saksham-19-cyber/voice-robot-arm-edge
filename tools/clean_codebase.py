"""Codebase cleaner and AST validator."""
import ast
import io
import os
import re
import tokenize

TARGET_DIRS = ["app", "src", "tests", "tools"]

MODULE_DOCS = {
    "src/asr/__init__.py": '"""ASR module."""',
    "src/asr/transcriber.py": '"""faster-whisper ASR transcriber."""',
    "src/bench/__init__.py": '"""Benchmarking suite."""',
    "src/bench/asr_ablation.py": '"""ASR latency ablation benchmark."""',
    "src/bench/asr_eval.py": '"""ASR Word Error Rate evaluator."""',
    "src/bench/audit_benchmarks.py": '"""Benchmark consistency auditor."""',
    "src/bench/audit_rollout_fps.py": '"""Closed-loop rollout throughput benchmark."""',
    "src/bench/bench_runner.py": '"""Multi-device benchmark runner."""',
    "src/bench/benchmark.py": '"""Policy latency benchmark."""',
    "src/bench/make_figures.py": '"""Generate PNG figures from benchmark results."""',
    "src/bench/measure_pipeline_latency.py": '"""Pipeline latency and early termination benchmark."""',
    "src/bench/measure_streamlit_linux.py": '"""Streamlit server memory auditor on Linux."""',
    "src/bench/measure_streamlit_rss.py": '"""Streamlit server memory auditor on Windows."""',
    "src/bench/nlu_eval.py": '"""NLU accuracy evaluator with Wilson intervals."""',
    "src/bench/profile_request.py": '"""Per-stage request latency profiler."""',
    "src/bench/run_policy_benchmark.py": '"""Single-thread CPU policy microbenchmark."""',
    "src/nlu/__init__.py": '"""Natural Language Understanding module."""',
    "src/nlu/classifier.py": '"""TF-IDF Logistic Regression intent classifier."""',
    "src/nlu/parser.py": '"""Dual-engine hybrid NLU parser."""',
    "src/nlu/rule_parser.py": '"""Rule-based spatial parser and workspace clipper."""',
    "src/nlu/schema.py": '"""NLU data schemas and result types."""',
    "src/rl/__init__.py": '"""Reinforcement learning and simulation module."""',
    "src/rl/env_utils.py": '"""FetchReach environment utilities and compatibility patch."""',
    "src/rl/export_onnx.py": '"""PyTorch actor export and INT8 quantization."""',
    "src/rl/rollout.py": '"""Closed-loop ONNX policy rollout."""',
    "src/rl/train.py": '"""SAC training on FetchReach-v4 with HER."""',
    "src/rl/train_stretch.py": '"""Exploratory SAC training on FetchPickAndPlace-v4."""',
    "src/tts/__init__.py": '"""Text-to-speech module."""',
    "src/tts/speaker.py": '"""Client-side Web Speech API generator."""',
    "src/__init__.py": '"""Root package."""',
    "tests/__init__.py": '"""Test package."""',
    "tests/test_asr.py": '"""Unit tests for ASR module."""',
    "tests/test_enum_patch.py": '"""Unit tests for MuJoCo enum patch."""',
    "tests/test_env_reuse.py": '"""Unit tests for simulation environment reuse."""',
    "tests/test_env_smoke.py": '"""Smoke tests for FetchReach-v4 environment."""',
    "tests/test_nlu.py": '"""Unit tests for NLU parser and boundary clipping."""',
    "tests/test_presets_rollout.py": '"""Integration tests for preset rollouts and video encoding."""',
    "tests/test_readme_assets.py": '"""Tests verifying all README asset links exist on disk."""',
    "tests/test_rollout.py": '"""Unit tests for closed-loop ONNX policy rollouts."""',
    "tools/__init__.py": '"""Tools package."""',
    "tools/clean_codebase.py": '"""Codebase cleaner and AST validator."""',
}


def normalize_ast(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
                if not node.body:
                    node.body = [ast.Pass()]
    elif isinstance(node, ast.Module):
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]

    for child in ast.iter_child_nodes(node):
        normalize_ast(child)
    return node


def get_normalized_ast_dump(source_code):
    tree = ast.parse(source_code)
    norm_tree = normalize_ast(tree)
    return ast.dump(norm_tree, include_attributes=False)


def preserve_comment(comment_text):
    c = comment_text.strip()
    if c.startswith("#!"):
        return True
    if re.match(r"^#\s*-\*-\s*coding:", c, re.IGNORECASE) or re.match(r"^#\s*coding[:=]", c, re.IGNORECASE):
        return True
    if "noqa" in c or "type: ignore" in c or "pragma" in c:
        return True
    return False


def find_docstring_spans(tree):
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                expr = node.body[0]
                if isinstance(expr.value, ast.Constant) and isinstance(expr.value.value, str):
                    is_only = (len(node.body) == 1)
                    spans.append((expr.lineno, expr.end_lineno, is_only, expr.col_offset))
    return spans


def clean_source(rel_path, source):
    tree = ast.parse(source)


    mod_doc_span = None
    if rel_path != "app/app.py" and tree.body and isinstance(tree.body[0], ast.Expr):
        first = tree.body[0]
        if isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            mod_doc_span = (first.lineno, first.end_lineno)


    doc_spans = find_docstring_spans(tree)


    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    comment_spans = []
    for tok in tokens:
        tok_type, tok_str, start, end, line = tok
        if tok_type == tokenize.COMMENT and not preserve_comment(tok_str):
            comment_spans.append((start[0], start[1], end[0], end[1]))

    lines = source.splitlines()


    for line_idx, start_col, end_line_idx, end_col in sorted(comment_spans, reverse=True):
        l_num = line_idx - 1
        line = lines[l_num]
        prefix = line[:start_col]
        if prefix.strip() == "":
            lines[l_num] = ""
        else:
            lines[l_num] = prefix.rstrip()


    for start_l, end_l, is_only, col in sorted(doc_spans, reverse=True):
        if is_only:
            indent = " " * col
            lines[start_l - 1] = f"{indent}pass"
            for idx in range(start_l, end_l):
                lines[idx] = ""
        else:
            for idx in range(start_l - 1, end_l):
                lines[idx] = ""


    if rel_path != "app/app.py":
        target_mod_doc = MODULE_DOCS.get(rel_path, '"""Module."""')
        if mod_doc_span:
            s_l, e_l = mod_doc_span
            lines[s_l - 1] = target_mod_doc
            for idx in range(s_l, e_l):
                lines[idx] = ""
        else:
            lines.insert(0, target_mod_doc)

    result = "\n".join(lines).rstrip() + "\n"
    return result


def main(dry_run=True):
    py_files = []
    for root_dir in TARGET_DIRS:
        if not os.path.exists(root_dir):
            continue
        for r, _, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(r, f)
                    py_files.append(full)

    py_files = sorted(py_files)
    print(f"Found {len(py_files)} Python files to inspect.")

    ast_diffs = []
    for path in py_files:
        rel = path.replace("\\", "/")
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()

        orig_ast = get_normalized_ast_dump(src)

        try:
            cleaned = clean_source(rel, src)
            cleaned_ast = get_normalized_ast_dump(cleaned)
        except Exception as e:
            print(f"Error processing {rel}: {e}")
            ast_diffs.append((rel, str(e)))
            continue

        if orig_ast != cleaned_ast:
            print(f"AST DIFF in {rel}!")
            ast_diffs.append((rel, "AST mismatch"))
        else:
            if not dry_run:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                print(f"Cleaned {rel} (AST verified identical)")
            else:
                print(f"OK (dry-run): {rel}")

    if not ast_diffs:
        print("\nAll files verified! Zero AST differences found.")
    else:
        print(f"\nFound {len(ast_diffs)} issues: {ast_diffs}")
    return len(ast_diffs) == 0


if __name__ == "__main__":
    import sys
    apply_mode = "--apply" in sys.argv
    main(dry_run=not apply_mode)
