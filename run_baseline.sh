#!/usr/bin/env bash
set -euo pipefail

alpha=1.0
dim=128
min_count=200

echo "Running: alpha=$alpha dim=$dim min_count=$min_count"
python src/main.py \
  --alpha "$alpha" \
  --dim "$dim" \
  --min_count "$min_count" \
  --dataset all \
  --method all \
  --scenario both \
  --lambda_val inf \
  --output_dir results/ \
  --disable_pca_plots
