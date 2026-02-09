#!/bin/bash
set -e

echo "=== Setting up diffusionjscc-speed ==="

# Install dependencies
pip install -r requirements.txt --break-system-packages

# Download CIFAR-10 (small, fast)
python -c "import torchvision; torchvision.datasets.CIFAR10('data', download=True)"

# Download Kodak (24 images)
mkdir -p data/kodak
for i in $(seq -w 1 24); do
    wget -q -nc "http://r0k.us/graphics/kodak/kodak/kodim${i}.png" -P data/kodak/ || true
done
echo "Kodak: $(ls data/kodak/*.png | wc -l) images"

# Download DIV2K (large — skip if already exists)
if [ ! -d "data/DIV2K_train_HR" ]; then
    echo "Downloading DIV2K train..."
    wget -q "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" -P data/
    unzip -q data/DIV2K_train_HR.zip -d data/
    rm data/DIV2K_train_HR.zip
fi
if [ ! -d "data/DIV2K_valid_HR" ]; then
    echo "Downloading DIV2K valid..."
    wget -q "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" -P data/
    unzip -q data/DIV2K_valid_HR.zip -d data/
    rm data/DIV2K_valid_HR.zip
fi
echo "DIV2K train: $(ls data/DIV2K_train_HR/*.png | wc -l) images"
echo "DIV2K valid: $(ls data/DIV2K_valid_HR/*.png | wc -l) images"

# Run tests
python -m pytest tests/ -q

echo "=== Setup complete ==="
