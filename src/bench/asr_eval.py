import csv
import json
import os
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

from src.asr.transcriber import WhisperASR
from src.nlu.parser import VoiceCommandParser


def normalize_text(text: str) -> list[str]:
    """Tokenize and normalize text for WER calculation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.strip().split()
    return words


def compute_word_error_rate(ref_words: list[str], hyp_words: list[str]) -> float:
    """Compute standard Levenshtein-based Word Error Rate."""
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    r = len(ref_words)
    h = len(hyp_words)
    dp = [[0] * (h + 1) for _ in range(r + 1)]

    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j

    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # substitution
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j] + 1       # deletion
                )

    return float(dp[r][h]) / float(r)


def evaluate_asr(manifest_path: str = "tests/asr/manifest.csv") -> dict:
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Evaluating ASR on {len(rows)} samples from {manifest_path}...")
    transcriber = WhisperASR(model_size="tiny.en", compute_type="int8")
    parser = VoiceCommandParser()

    results = []
    total_wer = 0.0
    command_parse_matches = 0

    for idx, row in enumerate(rows):
        file_path = row["file"]
        # Try resolving relative to manifest or repo root
        if not os.path.exists(file_path):
            alt_path = manifest_p.parent / file_path
            if alt_path.exists():
                file_path = str(alt_path)
            else:
                raise FileNotFoundError(f"Audio file not found: {file_path}")

        ref_text = row["reference_text"]
        provenance = row.get("provenance", "unknown")

        res = transcriber.transcribe(file_path)
        hyp_text = res.get("text", "")

        ref_tokens = normalize_text(ref_text)
        hyp_tokens = normalize_text(hyp_text)
        wer = compute_word_error_rate(ref_tokens, hyp_tokens)

        # Parse reference and hypothesis with NLU
        ref_nlu = parser.parse(ref_text)
        hyp_nlu = parser.parse(hyp_text)

        command_matched = (
            ref_nlu.intent == hyp_nlu.intent
            and (ref_nlu.goal_xyz is None and hyp_nlu.goal_xyz is None or (
                ref_nlu.goal_xyz is not None and hyp_nlu.goal_xyz is not None
            ))
        )
        if command_matched:
            command_parse_matches += 1

        total_wer += wer

        results.append({
            "file": file_path,
            "provenance": provenance,
            "reference_text": ref_text,
            "hypothesis_text": hyp_text,
            "wer": round(wer, 4),
            "ref_intent": ref_nlu.intent,
            "hyp_intent": hyp_nlu.intent,
            "command_parse_matched": command_matched,
            "hyp_goal_xyz": [round(x, 4) for x in hyp_nlu.goal_xyz] if hyp_nlu.goal_xyz else None
        })

    n = len(rows)
    mean_wer = total_wer / n if n > 0 else 0.0
    command_accuracy = command_parse_matches / n if n > 0 else 0.0

    summary = {
        "num_samples": n,
        "mean_wer": round(mean_wer, 4),
        "command_level_parse_accuracy": round(command_accuracy, 4),
        "samples": results
    }

    os.makedirs("results", exist_ok=True)
    out_path = Path("results/asr_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"ASR Evaluation Summary:")
    print(f"  Samples: {n}")
    print(f"  Mean WER: {mean_wer:.2%}")
    print(f"  Command-Level Parse Accuracy: {command_accuracy:.2%}")
    print(f"Saved results to {out_path}")
    return summary


if __name__ == "__main__":
    evaluate_asr()
