#!/bin/bash
set -e
cd /workspace/diffusionjscc-speed
python -m src.train.train_model_based --config configs/model_based.yaml
