"""Collect absolute-position Nori demonstrations with SAPIEN's click-and-drag gizmo.

Run this script from the ACT baseline directory with a graphical display.
The viewer's IK moves only the seven controlled joints of Nori's left arm.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import sapien.utils.viewer

import nori.env  # Register the custom robot and task before gym.make.
from mani_skill.utils.wrappers import RecordEpisode


ENV_ID = "Nori-Pick-Place-v1"
CONTROL_MODE = "pd_joint_pos"
ARM_STEP = 0.05  # Maximum commanded joint change per viewer frame, in radians.
GRIPPER_RAMP_STEPS = 20  # Close or open over one simulated second.
GRIPPER_HOLD_STEPS = 8  # Hold the final command for 0.4 simulated seconds.
SETTLE_STEPS = 30
IK_ERROR_TOLERANCE = 0.2
EEF_POSITION_TOLERANCE = 0.01  # Keep the gripper TCP within 1 cm of the gizmo.
EEF_LINK = "left_gripper_tcp_link"


def arm_positions(robot, joint_names):
    names = [joint.name for joint in robot.active_joints]
    qpos = robot.get_qpos()[0].cpu().numpy()
    return np.array([qpos[names.index(name)] for name in joint_names])


def check_arm_state(env):
    qpos = arm_positions(env.unwrapped.agent.robot, env.unwrapped.agent.arm_joint_names)
    if not np.isfinite(qpos).all() or np.max(np.abs(qpos)) > 2 * np.pi:
        raise RuntimeError("Nori arm state is unstable; discard this attempt and tune its controller")


def solve_ik_target(env, transform):
    """Return an arm joint target for the current TCP gizmo pose, if reachable."""
    robot = env.unwrapped.agent.robot
    eef = robot.links_map[EEF_LINK]._objs[0]
    if transform.selected_entity is None or transform.selected_entity.name != eef.entity.name or not transform.ik_enabled:
        return None, "Gripper TCP selection or IK was lost; press n to recenter the gizmo"
    result = transform.compute_ik()
    if result is None:
        return None, "Gripper TCP selection or IK was lost; press n to recenter the gizmo"
    target_qpos, success, error = result
    error_norm = np.linalg.norm(error)
    if (
        not np.isfinite(target_qpos).all()
        or not np.isfinite(error_norm)
        or (not success and error_norm > IK_ERROR_TOLERANCE)
    ):
        return None, "IK cannot reach this pose; move or rotate the gizmo closer"

    link_index = transform.ik_articulation.get_links().index(eef)
    transform.pinocchio_model.compute_forward_kinematics(target_qpos)
    predicted_pose = (
        transform.ik_articulation.pose
        * transform.pinocchio_model.get_link_pose(link_index)
    )
    position_error = np.linalg.norm(predicted_pose.p - transform._gizmo_pose.p)
    if position_error > EEF_POSITION_TOLERANCE:
        return None, "Gripper TCP cannot reach the gizmo within 1 cm"

    joint_indices = {joint.name: i for i, joint in enumerate(robot.active_joints)}
    target_arm = np.array([
        target_qpos[joint_indices[name]] for name in env.unwrapped.agent.arm_joint_names
    ])
    return target_arm, None


def execute_gripper(env, gripper_action, steps=8):
    arm_target = env.unwrapped.agent.controller.controllers["arm"].get_state()["target_qpos"][0].cpu().numpy()
    action = np.r_[arm_target, gripper_action].astype(np.float32)
    last_info = None
    done = False
    for _ in range(steps):
        _, _, terminated, truncated, last_info = env.step(action)
        check_arm_state(env)
        done = bool(terminated.item() or truncated.item())
        if done:
            break
    return last_info, done


def select_eef(env, viewer):
    eef = env.unwrapped.agent.robot.links_map[EEF_LINK]
    viewer.select_entity(eef._objs[0].entity)
    transform = next(
        plugin for plugin in viewer.plugins
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow)
    )
    transform.enabled = True
    transform.ik_enabled = True
    robot = env.unwrapped.agent.robot
    arm_names = {
        robot.joints_map[name]._objs[0].name
        for name in env.unwrapped.agent.arm_joint_names
    }
    missing = arm_names.difference(transform.move_group_joints)
    if missing:
        raise RuntimeError(f"SAPIEN IK is missing arm joints: {sorted(missing)}")
    transform.move_group_selection[:] = [
        name in arm_names for name in transform.move_group_joints
    ]
    transform.gizmo_matrix = eef._objs[0].entity.pose.to_transformation_matrix()
    return transform


def validate_dataset(path, observation_dim, action_dim):
    with h5py.File(path, "r") as data, path.with_suffix(".json").open() as metadata_file:
        metadata = json.load(metadata_file)
        if not data:
            raise ValueError("No successful demonstrations were recorded")
        if len(data) != len(metadata["episodes"]):
            raise ValueError("Trajectory and metadata episode counts differ")
        for name, trajectory in data.items():
            actions = trajectory["actions"]
            observations = trajectory["obs"]
            if actions.ndim != 2 or actions.shape[1] != action_dim:
                raise ValueError(f"{name}: expected {action_dim} action values per step")
            if observations.ndim != 2 or observations.shape != (len(actions) + 1, observation_dim):
                raise ValueError(f"{name}: expected one more state observation than actions")
            if not np.isfinite(actions[:]).all() or not np.isfinite(observations[:]).all():
                raise ValueError(f"{name}: non-finite observations or actions")
        for episode in metadata["episodes"]:
            if episode["control_mode"] != CONTROL_MODE or not episode["success"]:
                raise ValueError("Dataset contains a failed episode or a different control mode")
    return len(metadata["episodes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demos/Nori-Pick-Place-v1/teleop"))
    parser.add_argument("--trajectory-name", default="trajectory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-successes", type=int, default=20)
    args = parser.parse_args()

    trajectory_path = args.output_dir / f"{args.trajectory_name}.h5"
    if trajectory_path.exists() or trajectory_path.with_suffix(".json").exists():
        parser.error(f"{trajectory_path} already exists; choose another --trajectory-name")

    env = gym.make(
        ENV_ID,
        obs_mode="state",
        control_mode=CONTROL_MODE,
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
    )
    observation_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[0]
    if action_dim != len(env.unwrapped.agent.arm_joint_names) + 1:
        raise ValueError("Nori's action space must be seven arm positions plus one gripper value")
    env = RecordEpisode(
        env,
        output_dir=str(args.output_dir),
        trajectory_name=args.trajectory_name,
        save_trajectory=True,
        save_video=False,
        save_on_reset=False,
        source_type="teleoperation",
        source_desc="Nori click-and-drag using SAPIEN IK and absolute joint-position control",
    )

    seed = args.seed
    saved = 0
    last_info = None
    done = False
    gripper_action = 1.0  # The mimic controller maps +1 to open, -1 to closed.
    gripper_target = 1.0
    try:
        env.reset(seed=seed)
        last_info, done = execute_gripper(env, gripper_action)
        viewer = env.unwrapped.render_human()
        transform = select_eef(env, viewer)
        last_gizmo_matrix = np.array(transform.gizmo_matrix, copy=True)
        target_arm = None
        settle_steps = 0
        gripper_steps = 0
        last_warning = None
        arm_controller = env.unwrapped.agent.controller.controllers["arm"]
        print("Drag the left gripper TCP gizmo to move live. g: gripper, c: finish, r: retry, b: new scene, n: recenter, q: quit")
        while saved < args.num_successes:
            env.unwrapped.render_human()
            if viewer.window.key_press("h"):
                print("Drag the gizmo to move live; g: gripper; c: finish; r: retry; b: new scene; n: recenter; q: quit")
            elif viewer.window.key_press("n"):
                transform = select_eef(env, viewer)
                last_gizmo_matrix = np.array(transform.gizmo_matrix, copy=True)
                target_arm = None
                settle_steps = 0
                last_warning = None
                print("Gripper TCP gizmo recentered")
                continue
            elif viewer.window.key_press("g") and not done:
                gripper_target = -gripper_target
                gripper_steps = GRIPPER_RAMP_STEPS + GRIPPER_HOLD_STEPS
                print(f"Gripper {'opening' if gripper_target > 0 else 'closing'}")
            elif viewer.window.key_press("c"):
                success = last_info is not None and bool(last_info["success"].item())
                env.flush_trajectory(save=success)
                saved += int(success)
                print(f"{'Saved' if success else 'Discarded'} attempt; {saved}/{args.num_successes} successes")
                if saved == args.num_successes:
                    break
                seed += 1
                env.reset(seed=seed)
                gripper_action = gripper_target = 1.0
                last_info, done = execute_gripper(env, gripper_action)
                transform = select_eef(env, viewer)
                last_gizmo_matrix = np.array(transform.gizmo_matrix, copy=True)
                target_arm = None
                settle_steps = 0
                gripper_steps = 0
                last_warning = None
                continue
            elif viewer.window.key_press("b"):
                env.flush_trajectory(save=False)
                seed += 1
                env.reset(seed=seed)
                gripper_action = gripper_target = 1.0
                last_info, done = execute_gripper(env, gripper_action)
                transform = select_eef(env, viewer)
                last_gizmo_matrix = np.array(transform.gizmo_matrix, copy=True)
                target_arm = None
                settle_steps = 0
                gripper_steps = 0
                last_warning = None
                print(f"New box/goal layout (seed {seed})")
                continue
            elif viewer.window.key_press("r"):
                env.flush_trajectory(save=False)
                env.reset(seed=seed)
                gripper_action = gripper_target = 1.0
                last_info, done = execute_gripper(env, gripper_action)
                transform = select_eef(env, viewer)
                last_gizmo_matrix = np.array(transform.gizmo_matrix, copy=True)
                target_arm = None
                settle_steps = 0
                gripper_steps = 0
                last_warning = None
                print(f"Retrying seed {seed}")
                continue
            elif viewer.window.key_press("q"):
                success = last_info is not None and bool(last_info["success"].item())
                env.flush_trajectory(save=success)
                saved += int(success)
                break

            if done:
                continue
            eef_entity = env.unwrapped.agent.robot.links_map[EEF_LINK]._objs[0].entity
            if transform.selected_entity is None or transform.selected_entity.name != eef_entity.name or not transform.ik_enabled:
                warning = "Gripper TCP selection or IK was lost; press n to recenter the gizmo"
                if warning != last_warning:
                    print(warning)
                    settle_steps = 10
                last_warning = warning
                target_arm = arm_controller.get_state()["target_qpos"][0].cpu().numpy().copy()
            else:
                gizmo_matrix = np.array(transform.gizmo_matrix, copy=True)
                if not np.allclose(gizmo_matrix, last_gizmo_matrix, rtol=0, atol=1e-5):
                    last_gizmo_matrix = gizmo_matrix
                    new_target, warning = solve_ik_target(env, transform)
                    if warning is not None:
                        if warning != last_warning:
                            print(warning)
                        target_arm = arm_controller.get_state()["target_qpos"][0].cpu().numpy().copy()
                        settle_steps = 10
                    else:
                        target_arm = new_target
                        settle_steps = SETTLE_STEPS
                    last_warning = warning

            arm_delta = np.zeros(len(env.unwrapped.agent.arm_joint_names))
            tracking = False
            held_target = arm_controller.get_state()["target_qpos"][0].cpu().numpy()
            if target_arm is not None:
                arm_delta = np.clip(target_arm - held_target, -ARM_STEP, ARM_STEP)
                tracking = np.max(np.abs(arm_positions(env.unwrapped.agent.robot, env.unwrapped.agent.arm_joint_names) - target_arm)) > 0.01
            moving = np.max(np.abs(arm_delta)) > 1e-4
            if moving or gripper_steps > 0 or (tracking and settle_steps > 0):
                if gripper_steps > GRIPPER_HOLD_STEPS:
                    remaining = gripper_steps - GRIPPER_HOLD_STEPS
                    gripper_action += (gripper_target - gripper_action) / remaining
                elif gripper_steps > 0:
                    gripper_action = gripper_target
                action = np.r_[held_target + arm_delta, gripper_action].astype(np.float32)
                _, _, terminated, truncated, last_info = env.step(action)
                check_arm_state(env)
                if moving:
                    settle_steps = SETTLE_STEPS
                elif settle_steps > 0:
                    settle_steps -= 1
                gripper_steps = max(0, gripper_steps - 1)
                done = bool(terminated.item() or truncated.item())
                if done:
                    print(f"Task success: {bool(last_info['success'].item())}; press c to finish, r to retry, or b for a new scene")
    finally:
        env.close()

    if saved:
        count = validate_dataset(trajectory_path, observation_dim, action_dim)
        print(f"Absolute-position dataset: {trajectory_path} ({count} successful episodes)")
    else:
        print("No successful episodes saved")


if __name__ == "__main__":
    main()
