# Action Chunking with Transformers (ACT)

Code for running the ACT algorithm based on ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/pdf/2304.13705). It is adapted from the [original code](https://github.com/tonyzhaozh/act).

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```bash
conda create -n act-ms python=3.9
conda activate act-ms
pip install -e .
```

## Setup

Read through the [imitation learning setup documentation](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) which details everything you need to know regarding running imitation learning baselines in ManiSkill. It includes details on how to download demonstration datasets, preprocess them, evaluate policies fairly for comparison, as well as suggestions to improve performance and avoid bugs.

## Collecting Nori demonstrations for ACT

Run from this directory in a graphical session after installing the dependencies with
`pip install -e .`. The Nori task is registered by the collector. Its `state`
observations include cube and goal positions, and its recorded actions use
`pd_joint_delta_pos`, the mode supported by ACT's state dataset loader.

```bash
python collect_nori_demos.py --num-successes 20
```

Drag the **left gripper TCP** gizmo and the end effector follows as you move it.
Press `g` to toggle the gripper, `n` to recenter the gizmo, `c` to finish an attempt,
`r` to discard it and retry the same seed, `b` to discard it and load a new cube/goal
layout, or `q` to quit. Center the cube between
the fingers with the fingers clear of the table; after pressing `g`, hold the gizmo
still for 28 control steps (1.4 simulated seconds) before lifting. Only episodes
successful at their last step are saved. The collector writes
`demos/Nori-Pick-Place-v1/teleop/trajectory.h5` and the matching `.json`.
Use `--trajectory-name` for another run; an existing dataset is never overwritten.
The tool uses SAPIEN IK and joint-space steps, without collision-aware path
planning. Make short moves around the table and inspect the robot's motion.

The ACT-specific `nori/nori_act.urdf` fixes Nori's uncommanded wheels,
lift, and right arm. The agent disables internal robot collisions and uses
one gripper mimic mechanism; its delta arm controller holds the last target
when given zero arm action. The collector checks for unstable arm positions.
Both collection and ACT evaluation use this same robot setup. Once you have
successful episodes, start state-based training with:

```bash
python train.py --env-id Nori-Pick-Place-v1 \
  --demo-path demos/Nori-Pick-Place-v1/teleop/trajectory.h5 \
  --control-mode pd_joint_delta_pos --sim-backend physx_cpu \
  --num-demos 20 --batch-size 16 --max-episode-steps 500 \
  --num-eval-envs 1 --total-iters 30000 --no-capture-video
```

Choose `--batch-size` below the total number of recorded action steps. ACT's
training process imports the Nori task, and checks the control mode in the
trajectory metadata.

## Training

We provide scripts to train ACT on demonstrations.

Note that some demonstrations are slow (e.g. motion planning or human teleoperated) and can exceed the default max episode steps which can be an issue as imitation learning algorithms learn to solve the task at the same speed the demonstrations solve it. In this case, you can use the `--max-episode-steps` flag to set a higher value so that the policy can solve the task in time. General recommendation is to set `--max-episode-steps` to about 2x the length of the mean demonstrations length you are using for training. We have tuned baselines in the `baselines.sh` script that set a recommended `--max-episode-steps` for each task.

Example state-based training, learning from 100 demonstrations generated via motionplanning in the PickCube-v1 task.

```bash
seed=1
demos=100
python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "physx_cpu" --num_demos $demos --max_episode_steps 100 \
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
