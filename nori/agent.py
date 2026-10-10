import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
@register_agent()
class NoriLeftArmAgent(BaseAgent):
    uid = "nori_a3"
    urdf_path = str("nori/nori_act.urdf")
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_gripper_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            left_gripper_idler_link=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    fix_root_link = True
    disable_self_collisions = True


    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    arm_joint_names = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_bicep_yaw_joint",
    "left_elbow_pitch_joint",
    "left_forearm_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    )
    gripper_joint_names = (
    "left_gripper_joint",
    "left_gripper_idler_joint",
)

    left_gripper_mimic_name = "left_gripper_idler_joint"

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            use_target=True,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            upper=0.8,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={
                self.left_gripper_mimic_name: {
                    "joint": "left_gripper_joint",
                    "multiplier": -1.0,
                    "offset": 0.0,
                }
            },
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(
              arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
    
    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="front_camera_mount_frame",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=720,
                height=720,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["front_camera_mount_frame"],
            )
        ]