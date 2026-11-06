#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python stage1_vq/train_vq.py \
  --batch-size 128 \
  --num-workers 8 \
  --width 512 \
  --lr 1e-4 \
  --total-iter 100000 \
  --lr-scheduler 200000 \
  --code-dim 512 \
  --nb-code 512 \
  --down-t 2 \
  --depth 3 \
  --dilation-growth-rate 3 \
  --out-dir ./output/output_energy_multiscale \
  --dataname energy \
  --vq-act relu \
  --quantizer ema_reset_sim \
  --exp-name VQVAE_MS \
  --window-size 24 \
  --commit 0.001 \
  --gpu 0 \
  --multi-scale-vq \
  --multi-scale-dims 256 256 \
  --multi-scale-depths 2 4 \
  --multi-scale-dilations 1 3 \
  --multi-scale-weights 0.5 0.5
