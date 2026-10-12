"""Roll out absolute-position B-spline policies in ManiSkill environments."""

from collections import defaultdict

import numpy as np
import torch

from mani_skill.utils import common

from act.spline import sample_spline


def evaluate(n: int, agent, eval_envs, eval_kwargs):
    stats = eval_kwargs['stats']
    device = eval_kwargs['device']
    sim_backend = eval_kwargs['sim_backend']
    speedup = eval_kwargs.get('spline_speedup', 1.0)
    if speedup <= 0:
        raise ValueError('spline_speedup must be positive')
    state_mean = stats['state_mean'].to(device)
    state_std = stats['state_std'].to(device)
    action_mean = stats['action_mean'].to(device)
    action_std = stats['action_std'].to(device)
    use_visual_obs = isinstance(eval_envs.single_observation_space.sample(), dict)
    num_envs = eval_envs.num_envs
    action_space = eval_envs.single_action_space
    low = torch.as_tensor(action_space.low, device=device)
    high = torch.as_tensor(action_space.high, device=device)

    agent.eval()
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        segment = None
        phase = torch.zeros(num_envs, device=device)
        last_action = None
        eps_count = 0
        while eps_count < n:
            if use_visual_obs:
                obs = {key: common.to_tensor(value, device) for key, value in obs.items()}
                obs['state'] = (obs['state'] - state_mean) / state_std
            else:
                obs = (common.to_tensor(obs, device) - state_mean) / state_std

            if segment is None or torch.any(phase + speedup >= segment['knots'][:, segment['control_points'].shape[1]]):
                new_segment = agent.get_action(obs)
                if last_action is not None:
                    # Match the previous command within the beginning of the new curve.
                    count = new_segment['control_points'].shape[1]
                    start = new_segment['knots'][:, 3].clamp_min(0)
                    end = torch.minimum(new_segment['knots'][:, count] - 1e-4,
                                        start + 2 * speedup)
                    candidates = start[:, None] + (end - start)[:, None] * torch.linspace(
                        0, 1, 17, device=device)[None]
                    sampled = sample_spline(new_segment['knots'], new_segment['control_points'], candidates)
                    error = (((sampled - last_action[:, None]) * action_std) ** 2).mean(dim=-1)
                    phase = candidates.gather(1, error.argmin(dim=1, keepdim=True)).squeeze(1)
                else:
                    phase.zero_()
                segment = new_segment

            normalized_action = sample_spline(segment['knots'], segment['control_points'], phase[:, None])[:, 0]
            last_action = normalized_action
            action = (normalized_action * action_std + action_mean).clamp(low, high)
            if sim_backend == 'physx_cpu':
                action = action.cpu().numpy()
            obs, rew, terminated, truncated, info = eval_envs.step(action)
            phase = phase + speedup

            if truncated.any():
                assert truncated.all() == truncated.any(), 'All episodes must truncate together'
                if isinstance(info['final_info'], dict):
                    for key, value in info['final_info']['episode'].items():
                        eval_metrics[key].append(value.float().cpu().numpy())
                else:
                    for final_info in info['final_info']:
                        for key, value in final_info['episode'].items():
                            eval_metrics[key].append(value)
                eps_count += num_envs
                segment = None
                phase.zero_()
                last_action = None

    agent.train()
    return {key: np.stack(values) for key, values in eval_metrics.items()}
