import numpy as np

from .agent import NoriLeftArmAgent
import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv, CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building import actors
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building.ground import build_ground

import torch

@register_env("Nori-Pick-Place-v1", max_episode_steps=1000)
class NoriPickPlaceEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["nori_a3"]
    goal_radius = 0.06
    cube_half_size = 0.02
    placement_height_tolerance = 0.01
    floor_z = 0
    x = -.5
    agent: NoriLeftArmAgent
    # TCP above the table, pointing +X with the open jaws level along Y.
    robot_start_qpos = np.array(
        [-1.20465, -1.47397, 0.81563, 0.90344, 0.91489, -0.53418, -0.30929, 0.8, -0.8],
        dtype=np.float32,
    )

    def __init__(self, *args, robot_uids="nori_a3", **kwargs):
        # robot_uids="fetch" is possible, or even multi-robot 
        # setups via robot_uids=("fetch", "panda")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[self.x, 0, self.floor_z]))

   
    def _load_scene(self, options: dict):
        self.table_length = 0.8
        self.table_width = 0.6
        self.table_height = 0.42
        self.table_top_z = self.floor_z + self.table_height
        half_size = (
            self.table_length / 2,
            self.table_width / 2,
            self.table_height / 2,
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size)
        builder.initial_pose = sapien.Pose(p=[0, 0, self.floor_z +  self.table_height / 2])
        self.table = builder.build_kinematic(name="table")
        self.ground = build_ground(self.scene, altitude=self.floor_z)

        self.obj = actors.build_cube(
        self.scene,
        half_size=self.cube_half_size,
        color=np.array([12, 42, 160, 255]) / 255,
        name="cube",
        body_type="dynamic",
        initial_pose=sapien.Pose(p=[0, 0, self.table_top_z  + self.cube_half_size]),
        )
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, self.table_top_z+1e-3]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # using torch.device context manager to auto create tensors 
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.reset(torch.as_tensor(self.robot_start_qpos).expand(b, -1))

            # here is randomization code that randomizes the x, y position 
            # of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
            p = torch.zeros((b, 3))
            p[..., 0] =  torch.rand((b)) * 0.2 - 0.3
            p[..., 1] = torch.rand((b)) * 0.2 + .1
            p[..., 2] = self.table_top_z + self.cube_half_size
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=p, q=q)
            self.obj.set_pose(obj_pose)

            goal_p = p.clone()


            min_distance = 0.11
            max_distance = 0.15
            margin = self.goal_radius

            low = torch.tensor([
                -self.table_length / 2 + margin,
                -self.table_width / 2 + margin,
            ])
            high = -low

            goal_p = p.clone()
            remaining = torch.arange(b)

            while remaining.numel() > 0:
                n = remaining.numel()
                angle = 2 * np.pi * torch.rand(n)
                radius = torch.sqrt(
                    min_distance**2
                    + torch.rand(n) * (max_distance**2 - min_distance**2)
                )
                offset = torch.stack(
                    (radius * torch.cos(angle), radius * torch.sin(angle)), dim=1
                )
                candidate = p[remaining, :2] + offset
                valid = ((candidate >= low) & (candidate <= high)).all(dim=1)

                goal_p[remaining[valid], :2] = candidate[valid]
                remaining = remaining[~valid]


            goal_p[..., 2] = self.table_top_z + 1e-3

            self.goal_region.set_pose(
                Pose.create_from_pq(p=goal_p, q=euler2quat(0, np.pi / 2, 0))
)

    def _get_obs_extra(self, info: dict):
        if self.obs_mode_struct.use_state:
            return {
                "obj_pos": self.obj.pose.p,
                "goal_pos": self.goal_region.pose.p,
            }
        return {}

    def evaluate(self):
        # The cube must be inside the goal in XY and resting at table height.
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        cube_bottom_z = self.obj.pose.p[..., 2] - self.cube_half_size
        is_obj_on_table = torch.abs(cube_bottom_z - self.table_top_z) < self.placement_height_tolerance

        return {
            "success": is_obj_placed & is_obj_on_table,
        }

