"""Cubic B-spline targets and differentiable action sampling."""

import numpy as np
import torch
from scipy.interpolate import BSpline, splprep


DEGREE = 3


def fit_trajectory(actions, tolerance=0.002):
    """Fit one shared knot vector to every action channel with FITPACK.

    The final command is extended by one control step so a segment starting at
    the last recorded observation still has a future interval to execute.
    """
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Expected a nonempty (time, action_dim) trajectory")
    values = np.concatenate([values, values[-1:]], axis=0)
    if len(values) < DEGREE + 1:
        values = np.concatenate([values, np.repeat(values[-1:], DEGREE + 1 - len(values), axis=0)])
    times = np.arange(len(values), dtype=np.float64)
    smoothing = len(values) * tolerance**2
    for _ in range(12):
        (knots, coefficients, degree), _ = splprep(
            values.T, u=times, k=DEGREE, s=smoothing, nest=len(values) + DEGREE + 1)
        reconstructed = BSpline(knots, np.asarray(coefficients).T, degree)(times)
        if np.max(np.linalg.norm(reconstructed - values, axis=-1)) <= tolerance or smoothing == 0:
            break
        smoothing *= 0.25
    else:
        (knots, coefficients, degree), _ = splprep(
            values.T, u=times, k=DEGREE, s=0, nest=len(values) + DEGREE + 1)
    return np.asarray(knots), np.asarray(coefficients).T


def future_segment(knots, control_points, start_time, num_points):
    """Select fixed-size local support around an observation time."""
    if num_points < DEGREE + 2:
        raise ValueError("num_points must be at least 5 for a future cubic segment")
    span = int(np.searchsorted(knots, start_time, side="right") - 1)
    span = max(DEGREE, min(span, len(control_points) - 1))
    first = span - DEGREE
    local_points = control_points[first:first + num_points]
    if len(local_points) < num_points:
        local_points = np.concatenate([
            local_points,
            np.repeat(control_points[-1:], num_points - len(local_points), axis=0),
        ])
    local_knots = knots[first:first + num_points + DEGREE + 1]
    if len(local_knots) < num_points + DEGREE + 1:
        extension = knots[-1] + np.arange(1, num_points + DEGREE + 2 - len(local_knots))
        local_knots = np.concatenate([local_knots, extension])
    return local_knots.astype(np.float32) - start_time, local_points.astype(np.float32)


def knot_range_scale(splines, trajectory_lengths, num_points):
    """Half-range of the relative knot channel used in training targets."""
    minimum, maximum = np.inf, -np.inf
    for spline, length in zip(splines, trajectory_lengths):
        for step in range(length):
            knots, _ = future_segment(*spline, step, num_points)
            minimum = min(minimum, float(knots.min()))
            maximum = max(maximum, float(knots.max()))
    return max((maximum - minimum) / 2, 1e-2)


def sample_spline(knots, control_points, times):
    """Evaluate batched cubic splines in torch; knots may be learned tensors.

    knots: (B, P+4), control_points: (B, P, D), times: (B, T).
    Times outside the local support are held at its first or last value.
    """
    count = control_points.shape[1]
    left = knots[:, DEGREE:DEGREE + 1]
    right = knots[:, count:count + 1]
    times = torch.maximum(torch.minimum(times, right - 1e-5), left)
    basis = [((times >= knots[:, i:i + 1]) & (times < knots[:, i + 1:i + 2])).to(control_points.dtype)
             for i in range(count + DEGREE)]
    for degree in range(1, DEGREE + 1):
        next_basis = []
        for i in range(count + DEGREE - degree):
            left_width = knots[:, i + degree:i + degree + 1] - knots[:, i:i + 1]
            right_width = knots[:, i + degree + 1:i + degree + 2] - knots[:, i + 1:i + 2]
            left_term = (times - knots[:, i:i + 1]) / left_width.clamp_min(1e-6) * basis[i]
            right_term = (knots[:, i + degree + 1:i + degree + 2] - times) / right_width.clamp_min(1e-6) * basis[i + 1]
            next_basis.append(left_term + right_term)
        basis = next_basis
    return torch.bmm(torch.stack(basis, dim=-1), control_points)


def decode_spline_action(normalized_action, action_mean, action_std, low, high):
    """Decode absolute arm targets and the final normalized gripper channel.

    Absolute PD arm targets are not clipped by the controller or demonstrations.
    The gripper controller clips its normalized input to the action bounds.
    """
    action = normalized_action * action_std + action_mean
    return torch.cat([
        action[..., :-1],
        action[..., -1:].clamp(low[-1], high[-1]),
    ], dim=-1)


def spline_training_loss(prediction, target, knot_scale):
    """ACT L1 regression over range-scaled spline parameters."""
    point_error = (prediction["control_points"] - target["control_points"]).abs()
    knot_error = (prediction["knots"] - target["knots"]).abs() / knot_scale
    parameters_per_sample = point_error[0].numel() + knot_error[0].numel()
    loss = (point_error.sum() + knot_error.sum()) / (point_error.shape[0] * parameters_per_sample)
    return loss, point_error.mean(), knot_error.mean()
