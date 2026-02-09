#!/bin/bash
set -e
cd /workspace/diffusionjscc-speed
python -m src.train.train_diffusion --config configs/diffusion_jscc.yaml
