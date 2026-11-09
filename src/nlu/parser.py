"""
Unified NLU Parser integrating Rule-Based Parser (v1) and Statistical Intent Classifier (v2).
Provides fallback routing, clarification triggers, and target workspace mapping.
"""

import os
from typing import Optional
import numpy as np

from src.nlu.schema import IntentResult
from src.nlu.rule_parser import RuleBasedParser
from src.nlu.classifier import IntentClassifier


class VoiceCommandParser:
    """
    Dual-engine parser:
    1. Statistical Classifier for intent classification with confidence scores.
    2. Rule-Based Parser for entity extraction (directions, distances, objects, coordinates)
       and workspace bounds clipping.
    """

    def __init__(self, classifier_path: Optional[str] = "models/intent_classifier.pkl"):
        self.rule_parser = RuleBasedParser()
        self.classifier = None
        if classifier_path and os.path.exists(classifier_path):
            try:
                self.classifier = IntentClassifier(classifier_path)
            except Exception as e:
                print(f"Warning: Failed to load intent classifier from {classifier_path}: {e}")

    def parse(self, text: str, current_gripper_pos: Optional[np.ndarray] = None) -> IntentResult:
        # First, run rule-based parsing to extract entities, directions, coordinates
        rule_result = self.rule_parser.parse(text, current_gripper_pos=current_gripper_pos)

        # If statistical classifier is available, evaluate intent prediction
        if self.classifier is not None and text.strip():
            try:
                clf_intent, clf_prob, all_probs = self.classifier.predict_with_prob(text)
                rule_result.metadata["classifier_intent"] = clf_intent
                rule_result.metadata["classifier_confidence"] = clf_prob
                rule_result.metadata["all_probabilities"] = all_probs

                # If rule parser was uncertain (unknown) or classifier is highly confident on stop/unknown
                if rule_result.intent == "unknown" and clf_intent != "unknown" and clf_prob > 0.70:
                    rule_result.intent = clf_intent
                    rule_result.confidence = clf_prob
                    rule_result.clarification_needed = False
                elif clf_intent == "unknown" and clf_prob > 0.85:
                    rule_result.intent = "unknown"
                    rule_result.clarification_needed = True
                    rule_result.message = f"I'm not sure what you mean by '{text.strip()}'. Please specify a command like 'reach left' or 'stop'."
            except Exception as e:
                rule_result.metadata["classifier_error"] = str(e)

        return rule_result

