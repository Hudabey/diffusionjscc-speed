# Comprehensive Evaluation Results

Dataset: Kodak (24 images, full resolution)
Bandwidth ratio: 0.25 (1/4)

| Method | Params | SNR=0dB | SNR=5dB | SNR=10dB | SNR=15dB | SNR=20dB | Enc (ms) | Dec (ms) |
|--------|--------| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| JPEG + Ideal Code | - | 12.2 | 12.2 | 26.5 | 34.1 | 34.1 | - | - |
| WebP + Ideal Code | - | 12.2 | 12.2 | 26.2 | 36.7 | 36.7 | - | - |
| Shannon Bound | - | 0.8 | 1.5 | 2.6 | 3.8 | 5.0 | - | - |
| DeepJSCC (Ours) | 17.5M | 25.2 | 25.6 | 25.8 | 25.8 | 25.8 | 5.4 | 5.6 |
| Model-Based (Ours) | 2.6M | 22.2 | 23.4 | 23.8 | 23.7 | 23.6 | 2.8 | 5.9 |
