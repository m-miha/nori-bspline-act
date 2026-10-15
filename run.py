"""Run a saved state-based Nori spline policy on seeded episodes."""

import argparse
import math
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils import sapien_utils

import nori.env  # Register Nori-Pick-Place-v1.
from act.spline import decode_spline_action, sample_spline
from train import Agent, Args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Checkpoint written by train.py")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the first episode")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="Control steps per episode; defaults to 50 simulated seconds")
    parser.add_argument("--speedup", "--spline-speedup", type=float, default=1.0,
                        help="Curve-time multiplier: 2 runs at 2x speed, 0.5 at half speed")
    parser.add_argument("--control-freq", type=int, choices=(20, 100), default=100,
                        help="Spline command sampling rate in Hz (paper: 100; legacy: 20)")
    parser.add_argument("--blend-seconds", type=float, default=0.0,
                        help="Experimental arm-only handoff duration in simulated seconds (0 disables)")
    parser.add_argument("--video", type=Path, default=None,
                        help="Export MP4 matched to standard ACT live-viewer pace")
    parser.add_argument("--video-speed", type=float, default=1.0,
                        help="Additional playback multiplier after automatic 20 Hz pacing (default: 1)")
    parser.add_argument("--act-checkpoint", type=Path, default=None,
                        help="Standard ACT checkpoint for video pacing; defaults to newest ACT best checkpoint")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True,
                        help="Open the SAPIEN viewer (use --no-render for headless runs)")
    args = parser.parse_args()
    if args.num_episodes < 1:
        parser.error("--num-episodes must be positive")
    if args.max_episode_steps is None:
        args.max_episode_steps = 50 * args.control_freq
    if not math.isfinite(args.speedup) or args.speedup <= 0:
        parser.error("--speedup must be finite and positive")
    if not math.isfinite(args.blend_seconds) or args.blend_seconds < 0:
        parser.error("--blend-seconds must be finite and nonnegative")
    if args.max_episode_steps < 1:
        parser.error("--max-episode-steps must be positive")

    if not math.isfinite(args.video_speed) or args.video_speed <= 0:
        parser.error("--video-speed must be finite and positive")
    if args.video is None and args.video_speed != 1:
        parser.error("--video-speed requires --video")
    video_paths = []
    if args.video is not None:
        if args.video.suffix.lower() != ".mp4":
            parser.error("--video must end in .mp4")
        video_paths = [args.video if args.num_episodes == 1 else args.video.with_name(
            f"{args.video.stem}_seed{args.seed + i}{args.video.suffix}")
            for i in range(args.num_episodes)]
        from act.video import SimulationVideo, match_reference_pacing  # Registers a front-facing render camera.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["spline_config"]
    if config["degree"] != 3 or config["control_mode"] != "pd_joint_pos":
        parser.error("checkpoint must use cubic splines and pd_joint_pos control")

    env = gym.make(
        "Nori-Pick-Place-Video-v1" if args.video else "Nori-Pick-Place-v1", obs_mode="state", control_mode=config["control_mode"],
        reward_mode="sparse", sim_backend="physx_cpu",
        render_mode="human" if args.render else "rgb_array",
        max_episode_steps=args.max_episode_steps,
        sim_config=dict(sim_freq=100, control_freq=args.control_freq),
    )
    video = None
    try:
        # t22 knots are in 20 Hz demonstration ticks, not seconds. The paper's
        # S(speedup * t) therefore advances speedup * 20 / control_freq per tick.
        phase_step = args.speedup * 20.0 / env.unwrapped.control_freq
        blend_steps_requested = round(args.blend_seconds * env.unwrapped.control_freq)
        model_args = Args(num_queries=config["num_queries"])
        agent = Agent(env.unwrapped, model_args).to(device)
        agent.load_state_dict(checkpoint["ema_agent"])
        agent.eval()
        stats = {key: value.to(device) for key, value in checkpoint["norm_stats"].items()}
        low = torch.as_tensor(env.action_space.low, device=device)
        high = torch.as_tensor(env.action_space.high, device=device)

        with torch.inference_mode():
            for episode in range(args.num_episodes):
                seed = args.seed + episode
                obs, _ = env.reset(seed=seed)
                if args.video:
                    video = SimulationVideo(env, video_paths[episode])
                if args.render:
                    viewer = env.unwrapped.render_human()
                    if episode == 0:
                        # Look across the table toward the robot's front (+X side).
                        viewer.set_camera_pose(sapien_utils.look_at(
                            eye=[1.0, 0.1, 1.1], target=[-0.2, 0.1, 0.45]).sp)
                        env.unwrapped.render_human()
                segment = None
                phase = torch.zeros(1, device=device)
                last_action = None
                blend_segment = None
                blend_index = 0
                blend_steps = 0
                episode_return = 0.0
                if device.type == "cuda":
                    torch.cuda.synchronize()
                rollout_start = time.perf_counter()
                for step in range(1, args.max_episode_steps + 1):
                    state = (torch.as_tensor(obs, device=device) - stats["state_mean"]) / stats["state_std"]
                    # Request an overlapping segment early enough to traverse both curves.
                    lead = (blend_steps_requested + 1) * phase_step
                    if segment is None or (blend_segment is None and torch.any(
                            phase + lead >= segment["knots"][:, segment["control_points"].shape[1]])):
                        old_segment = segment
                        old_phase = phase.clone()
                        new_segment = agent.get_action(state)
                        if last_action is not None:
                            count = new_segment["control_points"].shape[1]
                            start = new_segment["knots"][:, 3].clamp_min(0)
                            end = start + 0.5 * (new_segment["knots"][:, count] - start)
                            candidates = start[:, None] + (end - start)[:, None] * torch.linspace(0, 1, 129, device=device)[None]
                            sampled = sample_spline(new_segment["knots"], new_segment["control_points"], candidates)
                            candidate_actions = decode_spline_action(
                                sampled, stats["action_mean"], stats["action_std"], low, high)
                            error = ((candidate_actions - last_action[:, None]) ** 2).mean(dim=-1)
                            phase = candidates.gather(1, error.argmin(dim=1, keepdim=True)).squeeze(1)
                        else:
                            phase.zero_()
                        if old_segment is not None and blend_steps_requested > 0:
                            old_end = old_segment["knots"][:, old_segment["control_points"].shape[1]]
                            new_end = new_segment["knots"][:, new_segment["control_points"].shape[1]]
                            # Short segments may not support the full requested overlap.
                            available = torch.minimum(old_end - old_phase, new_end - phase)
                            blend_steps = min(blend_steps_requested, max(
                                0, int(torch.floor((available.min() - 1e-5) / phase_step).item())))
                            if blend_steps > 0:
                                blend_segment = old_segment
                                blend_phase = old_phase
                                blend_index = 0
                        segment = new_segment

                    normalized_action = sample_spline(segment["knots"], segment["control_points"], phase[:, None])[:, 0]
                    action = decode_spline_action(
                        normalized_action, stats["action_mean"], stats["action_std"], low, high)
                    if blend_segment is not None:
                        old_sample = sample_spline(
                            blend_segment["knots"], blend_segment["control_points"], blend_phase[:, None])[:, 0]
                        old_action = decode_spline_action(
                            old_sample, stats["action_mean"], stats["action_std"], low, high)
                        fraction = blend_index / blend_steps
                        # Quintic smoothstep has zero first/second derivatives at both ends.
                        weight = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
                        action = torch.cat([
                            (1 - weight) * old_action[..., :-1] + weight * action[..., :-1],
                            action[..., -1:],  # Preserve the new segment's gripper command.
                        ], dim=-1)
                        blend_phase += phase_step
                        blend_index += 1
                        if blend_index > blend_steps:
                            blend_segment = None
                    last_action = action
                    obs, reward, terminated, truncated, info = env.step(action[0].cpu().numpy())
                    if video is not None:
                        video.step()
                    episode_return += float(np.asarray(reward).item())
                    if args.render:
                        env.unwrapped.render_human()
                    phase += phase_step
                    if bool(np.asarray(terminated).item()) or bool(np.asarray(truncated).item()):
                        break
                success = bool(np.asarray(info["success"]).item())
                print(f"seed={seed} success={success} steps={step} sim_seconds={step / env.unwrapped.control_freq:.2f} speedup={args.speedup:g} return={episode_return:.1f}", flush=True)
                if (args.render or video is not None) and success:
                    # Let physics settle for one second while holding the last command.
                    control_freq = env.unwrapped.control_freq
                    hold_action = action[0].cpu().numpy()
                    next_frame = time.monotonic()
                    for _ in range(control_freq):
                        if args.render and viewer.closed:
                            break
                        env.step(hold_action)
                        if video is not None:
                            video.step()
                        if args.render:
                            env.unwrapped.render_human()
                        next_frame += 1 / control_freq
                        if args.render:
                            time.sleep(max(0.0, next_frame - time.monotonic()))
                if video is not None:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    wall_seconds = time.perf_counter() - rollout_start
                    video.close()
                    match_reference_pacing(video, args, seed, wall_seconds)
                    video = None
    finally:
        try:
            if video is not None:
                video.close()
        finally:
            env.close()


if __name__ == "__main__":
    main()
