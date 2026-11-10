"""Dual-engine hybrid NLU parser."""




import os
from typing import Optional
import numpy as np

from src.nlu.schema import IntentResult
from src.nlu.rule_parser import RuleBasedParser
from src.nlu.classifier import IntentClassifier


class VoiceCommandParser:







    def __init__(self, classifier_path: Optional[str] = "models/intent_classifier.pkl"):
        self.rule_parser = RuleBasedParser()
        self.classifier = None
        if classifier_path and os.path.exists(classifier_path):
            try:
                self.classifier = IntentClassifier(classifier_path)
            except Exception as e:
                print(f"Warning: Failed to load intent classifier from {classifier_path}: {e}")

    def parse(self, text: str, current_gripper_pos: Optional[np.ndarray] = None) -> IntentResult:
        rule_result = self.rule_parser.parse(text, current_gripper_pos=current_gripper_pos)

        # 1. Conflicting / ambiguous commands: preserve clarification needed
        if rule_result.metadata.get("conflicting", False):
            return rule_result

        # 2. Rule stop always wins (safety critical: statistical classifier cannot override explicit stop)
        if rule_result.intent == "stop":
            return rule_result

        # If classifier is available, classifier refines intent
        if self.classifier is not None and text.strip():
            try:
                clf_intent, clf_prob, all_probs = self.classifier.predict_with_prob(text)
                rule_result.metadata["classifier_intent"] = clf_intent
                rule_result.metadata["classifier_confidence"] = clf_prob
                rule_result.metadata["all_probabilities"] = all_probs

                # Classifier cannot override explicit rule-based reach to pick unless high confidence and no directional vector
                if rule_result.intent == "reach" and clf_intent == "pick":
                    has_direction = any(d in rule_result.target for d in ["left", "right", "up", "down", "forward", "back"])
                    if clf_prob >= 0.70 and not has_direction:
                        final_intent = "pick"
                    else:
                        final_intent = "reach"
                elif rule_result.intent == "pick" and clf_intent == "reach":
                    if clf_prob >= 0.85:
                        final_intent = "reach"
                    else:
                        final_intent = "pick"
                elif clf_prob >= 0.50:
                    final_intent = clf_intent
                else:
                    final_intent = rule_result.intent

                if final_intent == "stop":
                    rule_result.intent = "stop"
                    rule_result.clarification_needed = False
                    rule_result.message = "Emergency stop acknowledged. No rollout is started."
                    rule_result.confidence = max(rule_result.confidence, clf_prob)
                elif final_intent == "pick":
                    rule_result.intent = "pick"
                    rule_result.goal_xyz = None
                    rule_result.clarification_needed = True
                    rule_result.message = "I can only move the arm to a position. Gripper manipulation (pick and place) is not supported by this policy."
                    rule_result.metadata["supported"] = False
                    rule_result.confidence = max(rule_result.confidence, clf_prob)
                elif final_intent == "unknown":
                    rule_result.intent = "unknown"
                    rule_result.clarification_needed = True
                    rule_result.message = f"I did not understand '{text.strip()}'. Try saying 'reach left', 'move up 10cm', or 'emergency stop'."
                    rule_result.confidence = clf_prob
                elif final_intent == "reach":
                    rule_result.intent = "reach"
                    rule_result.confidence = max(rule_result.confidence, clf_prob)
                    if rule_result.goal_xyz is None:
                        rule_result.clarification_needed = True
                        rule_result.message = f"Reach intent recognized for '{text.strip()}', but no target or direction was specified. Please clarify where to move."
                    else:
                        rule_result.clarification_needed = False
            except Exception as e:
                rule_result.metadata["classifier_error"] = str(e)

        return rule_result
