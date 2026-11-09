"""NLU accuracy evaluator with Wilson intervals."""








import os
import sys
import json
import math
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.nlu.rule_parser import RuleBasedParser
from src.nlu.classifier import IntentClassifier
from src.nlu.parser import VoiceCommandParser
from src.rl.env_utils import WORKSPACE_CENTER


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


def verify_goal_region(goal_xyz: np.ndarray, expected_region: str) -> bool:

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

    samples = real_samples

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

    for s in samples:
        text = s["text"]
        expected_intent = s.get("expected_intent") or s.get("intent")
        expected_region = s.get("expected_goal_region")
        if not expected_region and expected_intent == "reach":

            for kw in ["left", "right", "up", "down", "forward", "front", "back", "backward", "center"]:
                if kw in text.lower():
                    expected_region = kw
                    break


        r_res = rule_parser.parse(text)
        r_intent_match = (r_res.intent == expected_intent)
        r_e2e_match = r_intent_match and (verify_goal_region(r_res.goal_xyz, expected_region) if expected_intent == "reach" else (r_res.goal_xyz is None))
        if r_intent_match:
            rule_intent_ok += 1
        if r_e2e_match:
            rule_e2e_ok += 1


        c_pred, _, _ = classifier.predict_with_prob(text)
        if c_pred == expected_intent:
            clf_intent_ok += 1


        h_res = hybrid_parser.parse(text)
        h_intent_match = (h_res.intent == expected_intent)
        h_e2e_match = h_intent_match and (verify_goal_region(h_res.goal_xyz, expected_region) if expected_intent == "reach" else (h_res.goal_xyz is None))
        if h_intent_match:
            hybrid_intent_ok += 1
        if h_e2e_match:
            hybrid_e2e_ok += 1

    n = len(samples)
    return {
        "dataset_label": dataset_label,
        "sample_count": n,
        "class_distribution": dict(counts),
        "majority_class_baseline": {
            "majority_class": maj_class,
            "wilson_ci": maj_baseline
        },
        "rule_parser": {
            "intent_accuracy": wilson_score_interval(rule_intent_ok, n),
            "end_to_end_accuracy": wilson_score_interval(rule_e2e_ok, n),
        },
        "classifier_alone": {
            "intent_accuracy": wilson_score_interval(clf_intent_ok, n),
            "note": "Classifier predicts intent class only; goal coordinates are produced by rule parser."
        },
        "hybrid_parser": {
            "intent_accuracy": wilson_score_interval(hybrid_intent_ok, n),
            "end_to_end_accuracy": wilson_score_interval(hybrid_e2e_ok, n),
        }
    }


def run_full_nlu_eval():
    results = {}
    print("Evaluating Agent-Written Held-Out Phrasings (tests/nlu_heldout_handwritten.json)...", flush=True)
    res_agent = evaluate_dataset("tests/nlu_heldout_handwritten.json", "agent_written_heldout")
    results["agent_written_heldout"] = res_agent

    print("Checking Human-Written Held-Out Phrasings (tests/nlu_heldout_human.json)...", flush=True)
    res_human = evaluate_dataset("tests/nlu_heldout_human.json", "human_written_heldout")
    results["human_written_heldout"] = res_human

    with open("results/nlu_eval_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print("NLU Detailed Evaluation saved to results/nlu_eval_detailed.json", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return results


if __name__ == "__main__":
    run_full_nlu_eval()
