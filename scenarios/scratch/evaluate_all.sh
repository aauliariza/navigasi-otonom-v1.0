#!/usr/bin/env bash
# Skenario SCRATCH: evaluasi seluruh model lalu pemilihan EWM-SAW. Hasil hanya ditulis ke scenarios/scratch/outputs/.
# Argumen tambahan = override evaluasi, mis.: bash scenarios/scratch/evaluate_all.sh device=0
source "$(dirname "$0")/common.sh"
refuse_if_training_running
python -m src.evaluation.evaluate_detection --config $SCRATCH_ROOT/configs/evaluation/detection.yaml --include-extra --set split=val "$@"
python -m src.evaluation.evaluate_depth     --config $SCRATCH_ROOT/configs/evaluation/depth.yaml     --include-extra --set split=val "$@"
python -m src.optimization.model_selection  --config $SCRATCH_ROOT/configs/evaluation/selection.yaml
