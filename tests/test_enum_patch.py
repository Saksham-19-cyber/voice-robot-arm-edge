"""
Unit test verifying MuJoCo pybind11 enum comparison handling in gymnasium_robotics.utils.mujoco_utils.
On Linux GCC with NumPy >= 2.0, model.jnt_type[i] is a numpy.int32 which causes
`assert joint_type in (mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)` to raise AssertionError
if uncoerced. This test verifies that the patched functions execute without assertion error.
"""

import sys
import pytest
import numpy as np
import mujoco

import gymnasium_robotics.utils.mujoco_utils as mu
from src.rl.env_utils import apply_mujoco_utils_patch


def test_enum_coercion_membership():
    """Validates integer coercion logic used by the compatibility patch."""
    xml = """<mujoco><worldbody><body name="b1"><joint type="hinge" name="j1"/><geom size="0.1" type="sphere"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j1")
    raw_type = model.jnt_type[joint_id]

    assert isinstance(raw_type, (np.integer, int))
    assert int(raw_type) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def test_patched_mujoco_utils_joint_accessors():
    """Verifies that patched joint accessors (get/set qpos and qvel) succeed without AssertionError."""
    xml = """<mujoco><worldbody><body name="b1"><joint type="hinge" name="j1"/><geom size="0.1" type="sphere"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Ensure patch is applied
    apply_mujoco_utils_patch()

    # Test get_joint_qpos
    qpos = mu.get_joint_qpos(model, data, "j1")
    assert qpos.shape == (1,)

    # Test set_joint_qpos
    mu.set_joint_qpos(model, data, "j1", [0.75])
    assert np.isclose(data.qpos[0], 0.75)

    # Test get_joint_qvel
    qvel = mu.get_joint_qvel(model, data, "j1")
    assert qvel.shape == (1,)

    # Test set_joint_qvel
    mu.set_joint_qvel(model, data, "j1", [1.25])
    assert np.isclose(data.qvel[0], 1.25)
