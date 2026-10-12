#!/usr/bin/env bash
set -euo pipefail

# Record absolute-position demonstrations with collect_nori_demos.py first.
demo_path=${DEMO_PATH:-demos/Nori-Pick-Place-v1/teleop/trajectory_abs.h5}
num_demos=${NUM_DEMOS:-20}

python train.py --env-id Nori-Pick-Place-v1 \
  --demo-path "$demo_path" --control-mode pd_joint_pos \
  --sim-backend physx_cpu --num-demos "$num_demos" \
  --batch-size 16 --max-episode-steps 500 --num-eval-envs 1 \
  --total-iters 30000 --no-capture-video
