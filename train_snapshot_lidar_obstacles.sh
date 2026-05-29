#!/usr/bin/env bash
set -euo pipefail

SNAPSHOT_DIR="${SNAPSHOT_DIR:-/home/lzh/drone/quad-swarm-rl-single-snapshot-lidar-20260529}"
PYTHON_BIN="${PYTHON_BIN:-/home/lzh/miniconda3/envs/swarm-rl/bin/python}"

if [ ! -d "$SNAPSHOT_DIR" ]; then
  echo "Snapshot repo not found: $SNAPSHOT_DIR"
  echo "Expected the downloaded snapshot branch directory there."
  exit 1
fi

cd "$SNAPSHOT_DIR"

exec "$PYTHON_BIN" -m sample_factory.launcher.run \
  --run=swarm_rl.runs.single_quad.single_quad_obstacles_lidar \
  --max_parallel=1 \
  --pause_between=1 \
  --experiments_per_gpu=1 \
  --num_gpus=1
