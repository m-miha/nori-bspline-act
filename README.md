# B-spline ACT for ManiSkill with Blending

This project adapts ACT to predict cubic B-spline control points and knots and adds a novel blending technique for smooth handoffs during spline alignment.

## Installation

Using conda/mamba:

```bash
conda create -n act-ms python=3.9
conda activate act-ms
pip install -e .
```

## Demonstration Collection

This repository collected demonstrations with the [Nori A3](https://www.norirobotics.com/) as is a realistic choice for home use. Demonstrations were collected in SAPIEN/maniskill using TCP gizmo handling that positioned the end effector, leading to primitive and often overly linear demonstrations. This was done to demonstrate the robustness of this architecture under low quality low sample demonstrations that would potentially reflect home use.

```bash
python collect_nori_demos.py --num-successes 20 --trajectory-name [name]
```

Collection happens as follows:
Drag the **left gripper** TCP gizmo and the end effector follows as you move it.
Press `g` to toggle the gripper, `n` to recenter the gizmo, `c` to finish an attempt,
`r` to discard it and retry the same seed, `b` to discard it and load a new cube/goal
layout, or `q` to quit. Once you have
successful episodes, start state-based training with:

```bash
python train.py --env-id Nori-Pick-Place-v1 \
  --demo-path demos/Nori-Pick-Place-v1/teleop/[demo].h5 \
  --control-mode pd_joint_pos --sim-backend physx_cpu \
  --num-demos 20 --batch-size 16 --max-episode-steps 500 \
  --num-eval-envs 1 --total-iters 30000 --no-capture-video
```

Each demonstration is fitted to a cubic B-spline. At each observation, the
model predicts a local segment of control points and knots. Training compares
those parameters and actions sampled from the curve with the demonstration.
Evaluation samples one absolute action per environment step and aligns each
new segment with the last executed command.

## Running a checkpoint with temporal scaling

```bash
python run.py [checkpoint] --speedup 2
```

`--speedup` (alias `--spline-speedup`) implements temporal mapping
`a_exec(t) = S(speedup * t)` using simulated time. Use `1` for original speed,
`2` for twice the curve speed, `0.5` for half speed, etc.


### Running a checkpoint with novel blending for handoffs

Add `--blend-seconds 0.1` for a 100 ms transition between sampled arm commands.
The default is `0` (disabled). The runner requests a replacement before the old
segment ends, advances both curves at the selected `--speedup`, and transfers
arm control with a quintic smoothstep. The gripper follows the replacement
segment directly. The transition is shortened if either curve lacks sufficient
remaining support. Duration is rounded to control ticks; at 100 Hz, 0.1 seconds
is 10 intervals.

The blend lasts a fixed amount of simulated time even when `--speedup` changes.
Thus each component spline is temporally scaled, but the blended trajectory is
not an exact time-scaled copy across speed settings.


