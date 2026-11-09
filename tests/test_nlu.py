"""Unit tests for NLU parser and boundary clipping."""




import os
import pytest
import numpy as np

from src.nlu.schema import IntentResult
from src.nlu.rule_parser import RuleBasedParser
from src.nlu.parser import VoiceCommandParser
from src.rl.env_utils import WORKSPACE_BOUNDS, WORKSPACE_CENTER


@pytest.fixture
def parser():
    return VoiceCommandParser(classifier_path="models/intent_classifier.pkl")


@pytest.fixture
def rule_parser():
    return RuleBasedParser()


def test_normal_reach_commands(parser):
    commands = [
        ("reach to the left", "reach", "left"),
        ("move the arm forward", "reach", "forward"),
        ("go up", "reach", "up"),
        ("point arm towards table", "reach", "table"),
    ]
    for text, expected_intent, expected_target in commands:
        res = parser.parse(text)
        assert res.intent == expected_intent
        assert expected_target in res.target or res.target in expected_target
        assert res.goal_xyz is not None
        assert len(res.goal_xyz) == 3
        assert not res.clarification_needed


def test_distance_extraction(parser):
    res = parser.parse("move up by 10 cm")
    assert res.intent == "reach"
    assert "up" in res.target

    assert res.goal_xyz[2] > WORKSPACE_CENTER[2]


def test_unsupported_manipulation_commands(parser):
    unsupported_commands = [
        "pick up the red block",
        "grab the cube",
        "snatch that plastic piece",
        "release the object",
        "grasp the cylinder",
    ]
    for text in unsupported_commands:
        res = parser.parse(text)
        assert res.intent == "pick"

        assert res.goal_xyz is None

        assert "I can only move the arm to a position" in res.message
        assert res.clarification_needed is True


def test_stop_commands(parser):
    commands = [
        "stop the robot",
        "halt immediately",
        "emergency stop",
        "freeze",
    ]
    for text in commands:
        res = parser.parse(text)
        assert res.intent == "stop"
        assert not res.clarification_needed


def test_workspace_clipping_bounds(parser):

    res = parser.parse("move forward by 10 m")
    assert res.intent == "reach"

    assert res.goal_xyz[0] <= WORKSPACE_BOUNDS["x"][1]
    assert res.goal_xyz[0] >= WORKSPACE_BOUNDS["x"][0]
    assert res.metadata.get("was_clipped", False) is True


def test_ambiguous_and_adversarial_inputs(parser):
    adversarial_inputs = [
        "what is the weather in Paris",
        "tell me a funny joke",
        "play despacito on spotify",
        "xyzabc 12345 !@#$",
    ]
    for text in adversarial_inputs:
        res = parser.parse(text)
        assert res.intent == "unknown" or res.clarification_needed is True
        assert res.clarification_needed is True


def test_empty_input(parser):
    res = parser.parse("   ")
    assert res.intent == "unknown"
    assert res.clarification_needed is True


def test_handwritten_heldout_dataset_evaluation(parser):
    import json
    heldout_path = "tests/nlu_heldout_handwritten.json"
    assert os.path.exists(heldout_path)
    with open(heldout_path, "r") as f:
        samples = json.load(f)
    assert len(samples) == 50
    correct = 0
    for s in samples:
        res = parser.parse(s["text"])
        if res.intent == s["intent"]:
            correct += 1
    accuracy = correct / len(samples)

    assert accuracy >= 0.50
