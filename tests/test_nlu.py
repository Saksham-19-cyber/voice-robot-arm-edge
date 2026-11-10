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
    heldout_path = "tests/nlu_dev_agent_written.json"
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


def test_preset_commands_accuracy(parser):
    """Verify each of the six app presets produces the expected goal within 1 cm."""
    # 1. reach to the left
    res = parser.parse("reach to the left")
    assert res.intent == "reach"
    expected_left = np.array([1.34, 0.87, 0.53])
    assert np.allclose(res.goal_xyz, expected_left, atol=0.01), f"Left goal {res.goal_xyz} != {expected_left}"

    # 2. move up by 10 cm
    res = parser.parse("move up by 10 cm")
    assert res.intent == "reach"
    expected_up = np.array([1.34, 0.75, 0.63])
    assert np.allclose(res.goal_xyz, expected_up, atol=0.01), f"Up goal {res.goal_xyz} != {expected_up}"

    # 3. reach to the right
    res = parser.parse("reach to the right")
    assert res.intent == "reach"
    expected_right = np.array([1.34, 0.63, 0.53])
    assert np.allclose(res.goal_xyz, expected_right, atol=0.01), f"Right goal {res.goal_xyz} != {expected_right}"

    # 4. move down towards table
    res = parser.parse("move down towards table")
    assert res.intent == "reach"
    expected_table = np.array([1.34, 0.75, 0.42])
    assert np.allclose(res.goal_xyz, expected_table, atol=0.01), f"Table goal {res.goal_xyz} != {expected_table}"

    # 5. pick up the red block (unsupported manipulation)
    res = parser.parse("pick up the red block")
    assert res.intent == "pick"
    assert res.goal_xyz is None
    assert res.clarification_needed

    # 6. emergency stop
    res = parser.parse("emergency stop")
    assert res.intent == "stop"
    expected_stop = WORKSPACE_CENTER
    assert np.allclose(res.goal_xyz, expected_stop, atol=0.01), f"Stop goal {res.goal_xyz} != {expected_stop}"


def test_nlu_regression_known_failures(parser):
    """Regression tests for known failure cases (NOT part of accuracy benchmarks)."""
    # 1. 'lift the arm up 10 cm' -> reach upwards, not refused as manipulation
    res = parser.parse("lift the arm up 10 cm")
    assert res.intent == "reach", f"Expected reach but got {res.intent}"
    assert res.goal_xyz is not None
    assert res.goal_xyz[2] > WORKSPACE_CENTER[2]
    assert not res.clarification_needed

    # 2. 'hold your position' -> stop, not refused as manipulation
    res = parser.parse("hold your position")
    assert res.intent == "stop", f"Expected stop but got {res.intent}"
    assert not res.clarification_needed
    assert "No rollout is started" in res.message or "stop" in res.message.lower()

    # 3. 'kill all motors' -> stop
    res = parser.parse("kill all motors")
    assert res.intent == "stop", f"Expected stop but got {res.intent}"
    assert not res.clarification_needed

    # 4. 'cut power' -> stop
    res = parser.parse("cut power")
    assert res.intent == "stop", f"Expected stop but got {res.intent}"
    assert not res.clarification_needed


def test_nlu_regression_probes(parser):
    """Test the regression probes suite (tests/nlu_regression_probes.json)."""
    import json
    probes_path = "tests/nlu_regression_probes.json"
    assert os.path.exists(probes_path)
    with open(probes_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    assert len(probes) >= 26  # 6 original probes + at least 20 similar ones

    for p in probes:
        text = p["text"]
        res = parser.parse(text)
        assert res.intent == p["expected_intent"], (
            f"Probe '{text}' failed: expected intent {p['expected_intent']}, got {res.intent} ({p.get('description')})"
        )
        assert res.clarification_needed == p["expected_clarification"], (
            f"Probe '{text}' failed: expected clarification {p['expected_clarification']}, got {res.clarification_needed}"
        )
        if p.get("expected_goal_null"):
            assert res.goal_xyz is None, f"Probe '{text}' expected goal_xyz=None, got {res.goal_xyz}"
        if "expected_goal_xyz" in p:
            assert np.allclose(res.goal_xyz, p["expected_goal_xyz"], atol=0.01), (
                f"Probe '{text}' expected goal_xyz {p['expected_goal_xyz']}, got {res.goal_xyz}"
            )
        if "expected_was_clipped" in p:
            assert res.metadata.get("was_clipped") == p["expected_was_clipped"], (
                f"Probe '{text}' expected was_clipped={p['expected_was_clipped']}, got {res.metadata.get('was_clipped')}"
            )
        if "expected_goal_delta" in p:
            expected_delta = np.array(p["expected_goal_delta"])
            actual_delta = np.array(res.goal_xyz) - WORKSPACE_CENTER
            assert np.allclose(actual_delta, expected_delta, atol=0.01), (
                f"Probe '{text}' expected delta {expected_delta}, got {actual_delta}"
            )
        if "expected_direction" in p:
            assert p["expected_direction"] in res.target, (
                f"Probe '{text}' expected direction {p['expected_direction']} in target {res.target}"
            )
