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


def spline_training_loss(prediction, target, actions, reconstruction_weight=1.0):
    points_loss = torch.nn.functional.l1_loss(prediction["control_points"], target["control_points"])
    knots_loss = torch.nn.functional.l1_loss(prediction["knots"], target["knots"])
    times = torch.arange(actions.shape[1], device=actions.device, dtype=actions.dtype)[None].expand(actions.shape[0], -1)
    reconstructed = sample_spline(prediction["knots"], prediction["control_points"], times)
    # Local segments have variable duration; do not score later actions against
    # an extrapolated constant endpoint.
    valid = (times < target["knots"][:, target["control_points"].shape[1]:target["control_points"].shape[1] + 1]).to(actions.dtype)
    action_loss = ((reconstructed - actions).abs() * valid[..., None]).sum() / (valid.sum() * actions.shape[-1]).clamp_min(1)
    return points_loss + knots_loss + reconstruction_weight * action_loss, points_loss, knots_loss, action_loss
