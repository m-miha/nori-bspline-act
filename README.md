# B-spline ACT for ManiSkill

This project adapts [ACT](https://arxiv.org/pdf/2304.13705) to predict cubic B-spline control points and knots as in [B-spline Policy](https://arxiv.org/abs/2607.09648). The action interface uses absolute joint-position targets.

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```bash
conda create -n act-ms python=3.9
conda activate act-ms
pip install -e .
```

## Setup

Read through the [imitation learning setup documentation](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) which details everything you need to know regarding running imitation learning baselines in ManiSkill. It includes details on how to download demonstration datasets, preprocess them, evaluate policies fairly for comparison, as well as suggestions to improve performance and avoid bugs.

## Collecting Nori demonstrations

Run from this directory in a graphical session after installing the dependencies with
`pip install -e .`. The Nori task is registered by the collector. Its `state`
observations include cube and goal positions, and its recorded actions use
`pd_joint_pos`, with seven absolute arm targets and one gripper command.

```bash
python collect_nori_demos.py --num-successes 20 --trajectory-name trajectory_abs
```

Drag the **left gripper TCP** gizmo and the end effector follows as you move it.
Press `g` to toggle the gripper, `n` to recenter the gizmo, `c` to finish an attempt,
`r` to discard it and retry the same seed, `b` to discard it and load a new cube/goal
layout, or `q` to quit. Center the cube between
the fingers with the fingers clear of the table; after pressing `g`, hold the gizmo
still for 28 control steps (1.4 simulated seconds) before lifting. Only episodes
successful at their last step are saved. The collector writes
`demos/Nori-Pick-Place-v1/teleop/trajectory_abs.h5` and the matching `.json`.
Use `--trajectory-name` for another run; an existing dataset is never overwritten.
The tool uses SAPIEN IK and joint-space steps, without collision-aware path
planning. Make short moves around the table and inspect the robot's motion.

The Nori-specific `nori/nori_act.urdf` fixes Nori's uncommanded wheels,
lift, and right arm. The agent disables internal robot collisions and uses
one gripper mimic mechanism. The collector advances absolute joint targets
in small steps and checks for unstable arm positions.
Both collection and policy evaluation use this same robot setup. Once you have
successful episodes, start state-based training with:

```bash
python train.py --env-id Nori-Pick-Place-v1 \
  --demo-path demos/Nori-Pick-Place-v1/teleop/trajectory_abs.h5 \
  --control-mode pd_joint_pos --sim-backend physx_cpu \
  --num-demos 20 --batch-size 16 --max-episode-steps 500 \
  --num-eval-envs 1 --total-iters 30000 --no-capture-video
```

Choose `--batch-size` below the total number of recorded action steps. The policy's
training process imports the Nori task, and checks the control mode in the
trajectory metadata. Previously collected `pd_joint_delta_pos` demonstrations
cannot be used with this absolute-position policy; record new demonstrations.

Each demonstration is fitted to a cubic B-spline. At each observation, the
model predicts a local segment of control points and knots. Training compares
those parameters and actions sampled from the curve with the demonstration.
Evaluation samples one absolute action per environment step and aligns each
new segment with the last executed command. `--spline-speedup` changes how
quickly curve time advances per environment step; the default is 1. The
environment's control frequency is unchanged, so this flag does not create a
higher-frequency low-level controller.

## Training

We provide scripts to train ACT on demonstrations.

Demonstrations can exceed the default episode limit. Set `--max-episode-steps`
high enough to cover the task; `baselines.sh` contains the Nori state training
command for the absolute-position dataset.

Example state-based training for PickCube-v1 requires a trajectory recorded in
`pd_joint_pos` mode:

```bash
seed=1
demos=100
python train.py --env-id PickCube-v1 \
  --demo-path /path/to/pickcube.state.pd_joint_pos.physx_cpu.h5 \
  --control-mode "pd_joint_pos" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 100 \
  --total_iters 30000 --log_freq 100 --eval_freq 5000 \
  --exp-name=act-PickCube-v1-state-${demos}_motionplanning_demos-$seed \
  --track # track training on wandb
```

## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/rss/ZhaoKLF23,
  author       = {Tony Z. Zhao and
                  Vikash Kumar and
                  Sergey Levine and
                  Chelsea Finn},
  editor       = {Kostas E. Bekris and
                  Kris Hauser and
                  Sylvia L. Herbert and
                  Jingjin Yu},
  title        = {Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  booktitle    = {Robotics: Science and Systems XIX, Daegu, Republic of Korea, July
                  10-14, 2023},
  year         = {2023},
  url          = {https://doi.org/10.15607/RSS.2023.XIX.016},
  doi          = {10.15607/RSS.2023.XIX.016},
  timestamp    = {Thu, 20 Jul 2023 15:37:49 +0200},
  biburl       = {https://dblp.org/rec/conf/rss/ZhaoKLF23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
