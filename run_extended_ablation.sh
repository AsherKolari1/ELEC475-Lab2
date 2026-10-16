#!/bin/bash

mkdir -p results/extended_ablation/logs

BATCH_SIZES=(64 128 256)
SEEDS=(17 73 149)

for BATCH in "${BATCH_SIZES[@]}"; do
    for SEED in "${SEEDS[@]}"; do

        LOG="results/extended_ablation/logs/b${BATCH}_seed${SEED}.txt"

        echo "============================================================"
        echo "Running: Batch Size=$BATCH, Seed=$SEED"
        echo "Log: $LOG"
        echo "============================================================"

        python train.py \
            --model alexnet \
            --norm bn \
            --dataset cifar10 \
            --epochs 15 \
            --batch-size "$BATCH" \
            --lr 0.001 \
            --seed "$SEED" \
            --no-augment \
            2>&1 | tee "$LOG"

        echo
    done
done

echo "============================================================"
echo "ALL EXTENDED ABLATION RUNS COMPLETE"
echo "============================================================"
