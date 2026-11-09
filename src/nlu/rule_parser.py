"""Rule-based spatial parser and workspace clipper."""




import re
from typing import Optional, List, Tuple
import numpy as np

from src.rl.env_utils import WORKSPACE_BOUNDS, WORKSPACE_CENTER, clip_to_workspace
from src.nlu.schema import IntentResult



ANCHORS = {
    "center": np.array([1.34, 0.75, 0.53], dtype=np.float32),
    "table": np.array([1.34, 0.75, 0.42], dtype=np.float32),
    "block": np.array([1.30, 0.75, 0.43], dtype=np.float32),
    "cube": np.array([1.30, 0.75, 0.43], dtype=np.float32),
    "target": np.array([1.34, 0.75, 0.53], dtype=np.float32),
}





DIRECTION_VECTORS = {
    "front": np.array([0.10, 0.0, 0.0], dtype=np.float32),
    "forward": np.array([0.10, 0.0, 0.0], dtype=np.float32),
    "back": np.array([-0.10, 0.0, 0.0], dtype=np.float32),
    "backward": np.array([-0.10, 0.0, 0.0], dtype=np.float32),
    "left": np.array([0.0, 0.12, 0.0], dtype=np.float32),
    "right": np.array([0.0, -0.12, 0.0], dtype=np.float32),
    "up": np.array([0.0, 0.0, 0.10], dtype=np.float32),
    "higher": np.array([0.0, 0.0, 0.10], dtype=np.float32),
    "top": np.array([0.0, 0.0, 0.10], dtype=np.float32),
    "down": np.array([0.0, 0.0, -0.10], dtype=np.float32),
    "lower": np.array([0.0, 0.0, -0.10], dtype=np.float32),
    "bottom": np.array([0.0, 0.0, -0.10], dtype=np.float32),
}


def extract_distance(text: str) -> Optional[float]:





    cm_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)\b', text, re.IGNORECASE)
    if cm_match:
        return float(cm_match.group(1)) / 100.0

    m_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meters?)\b', text, re.IGNORECASE)
    if m_match:
        return float(m_match.group(1))

    mm_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:mm|millimeters?)\b', text, re.IGNORECASE)
    if mm_match:
        return float(mm_match.group(1)) / 1000.0

    return None


class RuleBasedParser:


    def __init__(self):
        self.stop_words = {"stop", "halt", "pause", "freeze", "abort", "cancel", "stand by"}
        self.pick_words = {"pick", "grasp", "grab", "lift", "take", "hold", "snatch", "release", "clasp"}
        self.reach_words = {"reach", "touch", "go", "move", "navigate", "extend", "point"}

    def parse(self, text: str, current_gripper_pos: Optional[np.ndarray] = None) -> IntentResult:
        raw_text = text.strip()
        cleaned = raw_text.lower()

        if not cleaned:
            return IntentResult(
                intent="unknown",
                target="none",
                confidence=0.0,
                clarification_needed=True,
                message="No command detected. Please speak or type a robot command.",
                raw_text=raw_text,
            )


        for sw in self.stop_words:
            if re.search(rf"\b{sw}\b", cleaned):
                goal = current_gripper_pos.tolist() if current_gripper_pos is not None else WORKSPACE_CENTER.tolist()
                return IntentResult(
                    intent="stop",
                    target="current_position",
                    goal_xyz=goal,
                    confidence=0.98,
                    clarification_needed=False,
                    message="Stopping arm movement immediately.",
                    raw_text=raw_text,
                )


        is_pick = any(re.search(rf"\b{pw}\b", cleaned) for pw in self.pick_words)


        is_reach = any(re.search(rf"\b{rw}\b", cleaned) for rw in self.reach_words)


        detected_directions = []
        for word, vec in DIRECTION_VECTORS.items():
            if re.search(rf"\b{word}\b", cleaned):
                detected_directions.append((word, vec))

        detected_anchor = None
        for anchor_name in ANCHORS:
            if re.search(rf"\b{anchor_name}\b", cleaned):
                detected_anchor = anchor_name
                break


        if not is_pick and not is_reach and not detected_directions and not detected_anchor:
            return IntentResult(
                intent="unknown",
                target="unrecognized",
                confidence=0.1,
                clarification_needed=True,
                message=f"I did not understand '{raw_text}'. Try saying 'reach left', 'move up 10cm', or 'pick up the block'.",
                raw_text=raw_text,
            )



        if is_pick or any(re.search(rf"\b{w}\b", cleaned) for w in ["pick", "grab", "snatch", "release", "clasp", "grasp"]):
            return IntentResult(
                intent="pick",
                target="unsupported",
                confidence=0.95,
                goal_xyz=None,
                clarification_needed=True,
                message="I can only move the arm to a position. Gripper manipulation (pick and place) is not supported by this policy.",
                raw_text=raw_text,
                metadata={"supported": False, "requires_gripper": True},
            )

        if is_reach or detected_directions or detected_anchor:
            intent = "reach"
        else:
            intent = "move"


        base_origin = current_gripper_pos.copy() if current_gripper_pos is not None else WORKSPACE_CENTER.copy()
        explicit_dist = extract_distance(cleaned)

        if detected_anchor:
            target_name = detected_anchor
            goal_pos = ANCHORS[detected_anchor].copy()
        elif detected_directions:
            target_name = " ".join([d[0] for d in detected_directions])
            offset = np.zeros(3, dtype=np.float32)
            for word, vec in detected_directions:
                if explicit_dist is not None:

                    unit_v = vec / (np.linalg.norm(vec) + 1e-8)
                    offset += unit_v * explicit_dist
                else:
                    offset += vec
            goal_pos = base_origin + offset
        else:
            target_name = "center"
            goal_pos = WORKSPACE_CENTER.copy()


        clipped_goal = clip_to_workspace(goal_pos)
        was_clipped = not np.allclose(goal_pos, clipped_goal, atol=1e-3)

        relative_offset = (clipped_goal - base_origin).tolist()

        if intent == "pick":
            confirm_msg = f"Acknowledged. Executing pick sequence for {target_name} at [{clipped_goal[0]:.2f}, {clipped_goal[1]:.2f}, {clipped_goal[2]:.2f}]."
        else:
            dir_str = target_name
            if explicit_dist:
                dir_str += f" by {int(explicit_dist * 100)}cm"
            clip_note = " (bounded to table limits)" if was_clipped else ""
            confirm_msg = f"Moving arm towards {dir_str}{clip_note}."

        return IntentResult(
            intent=intent,
            target=target_name,
            goal_xyz=[round(float(c), 3) for c in clipped_goal],
            relative_offset=[round(float(o), 3) for o in relative_offset],
            confidence=0.92 if detected_directions or detected_anchor else 0.75,
            clarification_needed=False,
            message=confirm_msg,
            raw_text=raw_text,
            metadata={"was_clipped": was_clipped},
        )
