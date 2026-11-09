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

    # Check whether real human voice recordings exist
    human_samples = [r for r in rows if r.get("provenance", "").lower() == "human"]
    has_real_benchmark = len(human_samples) >= 10

    if not has_real_benchmark:
        print(f"Notice: N={len(human_samples)} real human audio clips found in {manifest_path}.")
        print("Reporting WER as not measured (N=0 real clips; sample_speech.wav is synthetic/test-only).")
        summary = {
            "status": "not measured (N=0 real clips; sample_speech.wav is synthetic/test-only)",
            "wer_measured": False,
            "num_human_samples": len(human_samples),
            "num_total_samples": len(rows),
            "mean_wer": None,
            "command_level_parse_accuracy": None,
            "synthetic_test_run": {
                "file": rows[0]["file"] if rows else None,
                "hypothesis_text": transcriber.transcribe(rows[0]["file"]).get("text", "") if rows else "",
                "note": "Synthetic formant sine wave only, not a benchmark of real human speech."
            }
        }
        os.makedirs("results", exist_ok=True)
        out_path = Path("results/asr_eval.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved results to {out_path}")
        return summary


if __name__ == "__main__":
    evaluate_asr()
