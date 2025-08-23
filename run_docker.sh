#!/usr/bin/env bash
# helper scripts to build & run quickly

set -e

if [[ "$1" == "gpu" ]]; then
  docker build -f Dockerfile.gpu -t level1-gnn:gpu .
  docker run --rm --gpus all -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data level1-gnn:gpu
else
  docker build -f Dockerfile.cpu -t level1-gnn:cpu .
  docker run --rm -v $(pwd)/outputs:/app/outputs -v $(pwd)/data:/app/data level1-gnn:cpu
fi
