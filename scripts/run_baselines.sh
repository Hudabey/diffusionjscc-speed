#!/bin/bash
set -e
cd /workspace/diffusionjscc-speed
python -m src.eval.run_baselines --config configs/base.yaml
