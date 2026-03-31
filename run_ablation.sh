#!/usr/bin/env bash
set -euo pipefail

alphas=(0.5 1.0 2.0)
dims=(32 64 128)
min_counts=(50 100 200)

for alpha in "${alphas[@]}"; do
  for dim in "${dims[@]}"; do
    for min_count in "${min_counts[@]}"; do
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
    done
  done
done
