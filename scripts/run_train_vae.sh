#!/bin/bash
set -e
cd /workspace/diffusionjscc-speed
python -m src.train.train_vae --config configs/vae_jscc.yaml
