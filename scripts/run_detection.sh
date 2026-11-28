#!/bin/bash
# Full ADFA-LD detection matrix, one sequential CPU job (peak RSS ~0.8 GB).
# Seed-0 models first: their checkpoints are needed by the live experiments.
cd "$(dirname "$0")/.."
PY=${PY:-python3}
mkdir -p results/logs
$PY scripts/train_eval.py --threads 4 --seeds 0 --models StaticSeccompRuleBaseline,VortexGuard,LSTMIDS,TransformerIDS >> results/logs/detection.log 2>&1
$PY scripts/train_eval.py --threads 4 --seeds 1,2 --models VortexGuard,LSTMIDS,TransformerIDS >> results/logs/detection.log 2>&1
$PY scripts/train_eval.py --threads 4 --seeds 0,1,2 --models VG-FluxOnly,VG-SSMOnly,VG-NoGate >> results/logs/detection.log 2>&1
echo ALL_DONE >> results/logs/detection.log
