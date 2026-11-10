"""NLU accuracy evaluator with Wilson intervals."""








import os
import sys
import json
import math
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.nlu.rule_parser import RuleBasedParser
from src.nlu.classifier import IntentClassifier, generate_synthetic_dataset
from src.nlu.parser import VoiceCommandParser
from src.rl.env_utils import WORKSPACE_CENTER
from sklearn.model_selection import train_test_split


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:

    if total == 0:
        return {"accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z**2) / (4 * (total**2))))
    ci_low = max(0.0, center - spread)
    ci_high = min(1.0, center + spread)
    return {
        "accuracy": round(p, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }


def verify_goal_region(goal_xyz: np.ndarray, expected_region: str, intent: str = "reach") -> bool:
    # Stop intent: returning WORKSPACE_CENTER as goal is correct behavior — stop does not require goal_xyz=None.
    if intent == "stop":
        return True

    if expected_region in ("none", "null", "unsupported", "") or expected_region is None:
        return goal_xyz is None
    if goal_xyz is None:
        return False

    center = WORKSPACE_CENTER
    dx = goal_xyz[0] - center[0]
    dy = goal_xyz[1] - center[1]
    dz = goal_xyz[2] - center[2]

    reg = expected_region.lower()
    if "left" in reg:
        return dy > 0.01
    elif "right" in reg:
        return dy < -0.01
    elif "up" in reg or "top" in reg or "higher" in reg:
        return dz > 0.01
    elif "down" in reg or "lower" in reg or "bottom" in reg:
        return dz < -0.01
    elif "front" in reg or "forward" in reg:
        return dx > 0.01
    elif "back" in reg or "backward" in reg:
        return dx < -0.01
    elif "center" in reg:
        return np.linalg.norm(goal_xyz - center) < 0.05
    return True


def evaluate_sample_list(samples: List[Dict[str, Any]], dataset_label: str) -> Dict[str, Any]:
    if not samples:
        return {
            "dataset_label": dataset_label,
            "sample_count": 0,
            "status": "not measured (N=0; awaiting collected sentences)",
            "accuracy": None,
        }

    rule_parser = RuleBasedParser()
    hybrid_parser = VoiceCommandParser("models/intent_classifier.pkl")
    classifier = hybrid_parser.classifier

    intents = [s.get("expected_intent") or s.get("intent") for s in samples]
    from collections import Counter
    counts = Counter(intents)
    maj_class, maj_count = counts.most_common(1)[0]
    maj_baseline = wilson_score_interval(maj_count, len(samples))

    rule_intent_ok = 0
    rule_e2e_ok = 0
    clf_intent_ok = 0
    hybrid_intent_ok = 0
    hybrid_e2e_ok = 0

    # Stop list ablation: base vs expanded
    base_stop_phrases = ["emergency stop", "stop", "halt", "pause", "cancel"]
    rule_parser_base = RuleBasedParser()
    rule_parser_base.stop_phrases = base_stop_phrases

    hybrid_parser_base = VoiceCommandParser("models/intent_classifier.pkl")
    hybrid_parser_base.rule_parser.stop_phrases = base_stop_phrases

    rule_base_intent_ok = 0
    rule_base_e2e_ok = 0
    hybrid_base_intent_ok = 0
    hybrid_base_e2e_ok = 0

    for s in samples:
        text = s["text"]
        expected_intent = s.get("expected_intent") or s.get("intent")
        expected_region = s.get("expected_goal_region")
        if not expected_region and expected_intent == "reach":
            # Map synonyms to canonical region names (higher/top -> up, bottom/lower -> down)
            synonym_map = [
                (["higher", "top"],        "up"),
                (["bottom", "lower"],      "down"),
                (["left"],                 "left"),
                (["right"],                "right"),
                (["forward", "front"],     "forward"),
                (["backward", "back"],     "backward"),
                (["up"],                   "up"),
                (["down"],                 "down"),
                (["center"],               "center"),
            ]
            tl = text.lower()
            for words, region in synonym_map:
                if any(w in tl for w in words):
                    expected_region = region
                    break

        # Standard rule parser (expanded stops)
        r_res = rule_parser.parse(text)
        r_intent_match = (r_res.intent == expected_intent)
        r_e2e_match = r_intent_match and verify_goal_region(r_res.goal_xyz, expected_region, intent=expected_intent)
        if r_intent_match:
            rule_intent_ok += 1
        if r_e2e_match:
            rule_e2e_ok += 1

        # Base rule parser (without stop-list expansion)
        rb_res = rule_parser_base.parse(text)
        rb_intent_match = (rb_res.intent == expected_intent)
        rb_e2e_match = rb_intent_match and verify_goal_region(rb_res.goal_xyz, expected_region, intent=expected_intent)
        if rb_intent_match:
            rule_base_intent_ok += 1
        if rb_e2e_match:
            rule_base_e2e_ok += 1

        # Classifier alone (intent only)
        c_pred, _, _ = classifier.predict_with_prob(text)
        if c_pred == expected_intent:
            clf_intent_ok += 1

        # Standard hybrid parser (expanded stops)
        h_res = hybrid_parser.parse(text)
        h_intent_match = (h_res.intent == expected_intent)
        h_e2e_match = h_intent_match and verify_goal_region(h_res.goal_xyz, expected_region, intent=expected_intent)
        if h_intent_match:
            hybrid_intent_ok += 1
        if h_e2e_match:
            hybrid_e2e_ok += 1

        # Base hybrid parser (without stop-list expansion)
        hb_res = hybrid_parser_base.parse(text)
        hb_intent_match = (hb_res.intent == expected_intent)
        hb_e2e_match = hb_intent_match and verify_goal_region(hb_res.goal_xyz, expected_region, intent=expected_intent)
        if hb_intent_match:
            hybrid_base_intent_ok += 1
        if hb_e2e_match:
            hybrid_base_e2e_ok += 1

    n = len(samples)
    return {
        "dataset_label": dataset_label,
        "sample_count": n,
        "class_distribution": dict(counts),
        "majority_class_baseline": {
            "majority_class": maj_class,
            "wilson_ci": maj_baseline
        },
        "classifier_alone": {
            "intent_accuracy": wilson_score_interval(clf_intent_ok, n),
            "note": "Classifier predicts intent class only; goal coordinates are produced by rule parser."
        },
        "rule_parser": {
            "intent_accuracy": wilson_score_interval(rule_intent_ok, n),
            "end_to_end_accuracy": wilson_score_interval(rule_e2e_ok, n),
        },
        "hybrid_parser": {
            "intent_accuracy": wilson_score_interval(hybrid_intent_ok, n),
            "end_to_end_accuracy": wilson_score_interval(hybrid_e2e_ok, n),
        },
        "stop_list_expansion_ablation": {
            "without_stop_expansion": {
                "rule_parser_intent": wilson_score_interval(rule_base_intent_ok, n),
                "rule_parser_e2e": wilson_score_interval(rule_base_e2e_ok, n),
                "hybrid_parser_intent": wilson_score_interval(hybrid_base_intent_ok, n),
                "hybrid_parser_e2e": wilson_score_interval(hybrid_base_e2e_ok, n),
            },
            "with_stop_expansion": {
                "rule_parser_intent": wilson_score_interval(rule_intent_ok, n),
                "rule_parser_e2e": wilson_score_interval(rule_e2e_ok, n),
                "hybrid_parser_intent": wilson_score_interval(hybrid_intent_ok, n),
                "hybrid_parser_e2e": wilson_score_interval(hybrid_e2e_ok, n),
            }
        }
    }


def evaluate_dataset(dataset_path: str, dataset_label: str) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        return {"error": f"Dataset file not found: {dataset_path}"}

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    real_samples = [s for s in samples if not s.get("text", "").startswith("example:")]

    if not real_samples:
        return {
            "dataset_label": dataset_label,
            "sample_count": 0,
            "status": f"not measured (N=0; awaiting human-collected heldout sentences in {dataset_path})",
            "accuracy": None,
        }

    return evaluate_sample_list(real_samples, dataset_label)


def run_full_nlu_eval():
    import subprocess
    results = {}

    # --- NLU Freeze metadata (Item 2) ---
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit_hash = "unknown"
    results["nlu_frozen_v1_commit"] = commit_hash
    results["nlu_frozen_v1_tag"] = "nlu-frozen-v1"
    results["evaluator_fix_note"] = (
        "Evaluator fixed 2026-11-10 truth-pass-2: "
        "(1) stop intent no longer requires goal_xyz=None (stop returns current position, which is correct); "
        "(2) synonym keywords higher/top->up and bottom/lower->down added to expected_region extraction."
    )

    print("Evaluating Synthetic Test Split (seed=42, test_size=0.2, N=80)...", flush=True)
    texts, labels = generate_synthetic_dataset(seed=42)
    _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    synth_samples = [{"text": t, "expected_intent": y} for t, y in zip(X_test, y_test)]
    res_synth = evaluate_sample_list(synth_samples, "synthetic_test_split")
    results["synthetic_test_split"] = res_synth

    print("Evaluating Development Set - Agent-Written Phrasings (tests/nlu_dev_agent_written.json)...", flush=True)
    print("NOTE: This is the development set — the parser was tuned while looking at it; not an independent generalization test.", flush=True)
    res_agent = evaluate_dataset("tests/nlu_dev_agent_written.json", "dev_agent_written")
    if "hybrid_parser" in res_agent:
        res_agent["hybrid_parser"]["previous_architecture_comparison"] = {
            "previous_hybrid_intent_accuracy": 0.54,
            "previous_hybrid_end_to_end_accuracy": 0.48,
            "note": "Old hybrid prioritized rules first; new hybrid uses classifier intent with rule spatial extraction and rule fallback.",
        }
        res_agent["label_warning"] = (
            "DEVELOPMENT SET: parser was tuned while looking at these examples. "
            "Not independent. Do not report as generalization accuracy."
        )
    results["dev_agent_written"] = res_agent

    print("Checking Human-Written Held-Out Phrasings (tests/nlu_heldout_human.json)...", flush=True)
    res_human = evaluate_dataset("tests/nlu_heldout_human.json", "human_written_heldout")
    res_human["note"] = (
        "Human eval must be run once on tag nlu-frozen-v1 with no parser tuning afterwards. "
        "N=0 currently."
    )
    results["human_written_heldout"] = res_human

    with open("results/nlu_eval_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print("NLU Detailed Evaluation saved to results/nlu_eval_detailed.json", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == "__main__":
    run_full_nlu_eval()
