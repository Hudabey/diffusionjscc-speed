# DiffusionJSCC: ML-Based Channel Coding for Continuous-Valued Source Transmission

> **Target**: Fraunhofer IIS — Intern / Master Thesis: ML-Based Channel Coding for Continuous-Valued Source Symbol Transmission
> **Approach**: Model-based deep learning (Shlezinger et al., 2021) + generative AI (VAE, Diffusion) for analog JSCC
> **GPU**: RTX 5090 (RunPod, 50GB disk + 20GB persistent volume)

---

## Project Overview

This project designs and evaluates **ML-based channel coding schemes for continuous-valued source data**, combining classical communication-theoretic structure (Shannon-Kotel'nikov mappings, power constraints, bandwidth ratios) with learned neural encoders/decoders. We implement three generative transmission schemes — **VAE-JSCC**, **Diffusion-JSCC**, and a **model-based hybrid** — and rigorously benchmark them against digital baselines (JPEG + LDPC, BPG + Polar) across AWGN and Rayleigh fading channels.

### Why This Goes Beyond the Job Description

| JD Requirement | What We Deliver |
|---|---|
| Literature review on generative models for physical layer | Structured survey covering VAE, GAN, Diffusion, and model-based DL for JSCC (included in `docs/`) |
| Design generative AI-based transmission schemes | Three schemes: VAE-JSCC, Diffusion-JSCC, Model-Based Hybrid |
| Evaluate against digital baselines | JPEG+LDPC, BPG+Polar at matched rates; Shannon limit plotted |
| *Not asked but impressive* | Rayleigh fading channel (not just AWGN) |
| *Not asked but impressive* | Latency profiling + Pareto frontier (quality vs. speed) |
| *Not asked but impressive* | Model-based deep learning: unrolled optimization inspired by Shlezinger et al. |
| *Not asked but impressive* | Bandwidth expansion/compression experiments (N:M mappings) |
| *Not asked but impressive* | Kodak + DIV2K evaluation (not toy CIFAR-10) |

---

## Architecture

```
Source x ∈ ℝ^n  →  [Learned Encoder f_θ]  →  z ∈ ℝ^k  →  [Power Constraint]  →  [Channel]  →  ẑ  →  [Learned Decoder g_φ]  →  x̂ ∈ ℝ^n
                                                                    ↑
                                                              AWGN / Rayleigh
```

**Bandwidth ratio** ρ = k/n controls compression. When ρ < 1: bandwidth compression. When ρ > 1: bandwidth expansion (redundancy for error protection).

### Scheme 1: VAE-JSCC (Milestone 2)
- Encoder outputs μ, σ → reparameterization → channel input z
- KL divergence regularizes latent space → graceful degradation
- Decoder reconstructs from noisy ẑ
- Loss: MSE + β·KL + power penalty

### Scheme 2: Diffusion-JSCC (Milestone 3)
- JSCC encoder/decoder as backbone
- Diffusion model as channel denoiser at receiver
- Score-based refinement conditioned on SNR
- Consistency distillation for fast inference (1-4 steps)

### Scheme 3: Model-Based Hybrid (Milestone 4)
- Inspired by Shlezinger et al. [4]: unrolled ISTA/ADMM with learned parameters
- Classical S-K mapping structure with learned nonlinearities
- Domain knowledge: power constraint, bandwidth ratio, channel model baked into architecture
- Smallest model, most interpretable, fastest inference

---

## Milestones

### Milestone 0: Infrastructure & Data (Stage 1)
**Status**: 🔴 Not started
**Time**: 2-3 hours
**Output**: Working repo with CI, data pipeline, config system

- [ ] Set up RunPod: RTX 5090, 50GB disk, 20GB persistent volume at `/workspace`
- [ ] Push skeleton to `github.com/Hudabey/diffusionjscc-speed`
- [ ] Data pipeline:
  - CIFAR-10 (dev/debug only — fast iteration)
  - **Kodak** (24 images, 768×512 — standard eval benchmark)
  - **DIV2K** (800 train / 100 val — real training data)
- [ ] Config system (YAML-based, hydra-style)
- [ ] Channel models: AWGN + Rayleigh fading (differentiable)
- [ ] Digital baselines: JPEG+LDPC and BPG+Polar wrappers
- [ ] Evaluation metrics: PSNR, SSIM, LPIPS, MS-SSIM
- [ ] Reproducibility: seeded, deterministic, logged

**Key files**:
```
configs/
  base.yaml              # shared hyperparams
  vae_jscc.yaml
  diffusion_jscc.yaml
  model_based.yaml
src/
  channel/
    awgn.py              # AWGN channel (differentiable)
    rayleigh.py          # Rayleigh fading channel
    utils.py             # power normalization, SNR conversion
  data/
    datasets.py          # CIFAR-10, Kodak, DIV2K loaders
    transforms.py        # crop, normalize, augment
  eval/
    metrics.py           # PSNR, SSIM, LPIPS, MS-SSIM
    digital_baselines.py # JPEG+LDPC, BPG+Polar
  utils/
    config.py
    seed.py
    logging.py
scripts/
  setup_pod.sh           # one-command pod setup
tests/
  test_channel.py
  test_data.py
  test_metrics.py
```

**Definition of Done**: `pytest tests/ -q` passes, data loads, AWGN/Rayleigh channels verified against theory.

---

### Milestone 1: Digital Baselines (Stage 1-2)
**Status**: 🔴 Not started
**Time**: 3-4 hours
**Output**: PSNR vs SNR curves for JPEG+LDPC and BPG+Polar on Kodak

- [ ] Implement JPEG compression at multiple quality levels
- [ ] Implement simulated LDPC/Polar channel coding (BER → packet error model)
- [ ] Sweep SNR from -5 to 25 dB, bandwidth ratios ρ ∈ {1/16, 1/8, 1/4, 1/2}
- [ ] Generate baseline curves showing **cliff effect** (digital falls off a cliff below threshold SNR)
- [ ] Plot **Shannon limit** (rate-distortion bound) as theoretical reference
- [ ] Save all results to `outputs/baselines/`

**Key files**:
```
src/eval/digital_baselines.py
scripts/run_baselines.sh
outputs/baselines/
  psnr_vs_snr_jpeg_ldpc.json
  psnr_vs_snr_bpg_polar.json
  figures/cliff_effect.pdf
```

**Definition of Done**: Clear cliff-effect plot showing digital baselines collapse below threshold SNR. Shannon bound plotted.

---

### Milestone 2: VAE-JSCC (Stage 2-4)
**Status**: 🔴 Not started
**Time**: 8-12 hours
**Output**: Trained VAE-JSCC, PSNR vs SNR curves on Kodak, comparison with baselines

- [ ] Encoder: ResBlocks → μ, log_σ → reparameterize → power normalize
- [ ] Decoder: ResBlocks from noisy latent → reconstruction
- [ ] Loss: MSE + β·KL(q(z|x) || N(0,1)) + λ·power_penalty
- [ ] SNR-adaptive: condition encoder/decoder on channel SNR
- [ ] Train on DIV2K (random crops 256×256), evaluate on Kodak
- [ ] Sweep bandwidth ratios ρ ∈ {1/16, 1/8, 1/4, 1/2}
- [ ] Sweep SNR: train at multiple SNRs or use SNR-adaptive conditioning
- [ ] Plot PSNR vs SNR vs baselines → show **graceful degradation** (no cliff effect)
- [ ] Ablation: β-sweep showing rate-distortion tradeoff

**Key files**:
```
src/models/vae_jscc/
  encoder.py
  decoder.py
  model.py              # full VAE-JSCC pipeline
src/train/train_vae.py
outputs/vae_jscc/
  checkpoints/
  figures/
  metrics.json
```

**Definition of Done**: VAE-JSCC outperforms JPEG+LDPC at low SNR, shows graceful degradation, competitive at high SNR.

---

### Milestone 3: Diffusion-JSCC (Stage 4-7)
**Status**: 🔴 Not started
**Time**: 12-16 hours
**Output**: Trained Diffusion-JSCC receiver, quality improvement over VAE-JSCC

- [ ] Phase A: Train lightweight JSCC backbone (encoder-decoder, no VAE)
- [ ] Phase B: Train diffusion model as SNR-conditioned channel denoiser
  - Input: noisy reconstruction from backbone
  - Condition: estimated SNR
  - Architecture: UNet with SNR embedding
  - Training: standard DDPM objective
- [ ] Phase C: Consistency distillation for fast inference
  - Distill 50-step DDPM → 1-4 step consistency model
  - Maintain quality while reducing latency 10-50×
- [ ] Evaluate on Kodak at ρ ∈ {1/16, 1/8, 1/4, 1/2}
- [ ] Compare: Backbone alone vs. Backbone + Diffusion vs. VAE-JSCC vs. Baselines
- [ ] Perceptual metrics: LPIPS, FID (diffusion should excel here)

**Key files**:
```
src/models/diffusion_jscc/
  backbone.py            # lightweight JSCC encoder-decoder
  diffusion.py           # DDPM/DDIM denoiser
  consistency.py         # consistency distillation
  model.py               # full pipeline
src/train/
  train_backbone.py
  train_diffusion.py
  train_consistency.py
```

**Definition of Done**: Diffusion-JSCC beats VAE-JSCC on perceptual metrics (LPIPS). Consistency model achieves <100ms inference.

---

### Milestone 4: Model-Based Hybrid (Stage 7-9)
**Status**: 🔴 Not started
**Time**: 8-10 hours
**Output**: Model-based DL scheme showing communication-theoretic structure + learned flexibility

This is the **"show them you speak their language"** milestone. Directly references Shlezinger et al. [4].

- [ ] Design: Unrolled iterative decoder with learned step sizes and denoisers
  - K iterations of: ẑ_k+1 = ẑ_k - α_k · ∇f(ẑ_k) + D_θ(ẑ_k, SNR)
  - α_k: learned step sizes
  - D_θ: small learned denoiser (3-layer CNN)
  - ∇f: model-based gradient from channel likelihood
- [ ] Encoder: Learned nonlinear mapping inspired by S-K mappings
  - Architecture: MLP with residual connections
  - Constraint: output power = P, bandwidth ratio = ρ
  - Initialization: linear mapping (matched to source statistics)
- [ ] Train end-to-end with known channel model
- [ ] Compare parameter efficiency: ~100K params vs. millions for VAE/Diffusion
- [ ] Show interpretability: learned step sizes, denoiser activations
- [ ] Evaluate on Kodak

**Key files**:
```
src/models/model_based/
  encoder.py             # learned S-K-inspired mapping
  unrolled_decoder.py    # K-iteration unrolled optimizer
  model.py               # full pipeline
src/train/train_model_based.py
```

**Definition of Done**: Competitive PSNR with 10-100× fewer parameters. Visualize learned mappings. Show convergence of unrolled iterations.

---

### Milestone 5: Comprehensive Evaluation (Stage 9-11)
**Status**: 🔴 Not started
**Time**: 6-8 hours
**Output**: Publication-quality figures, comparison tables, latency analysis

- [ ] **Main comparison table**: All methods × all metrics × all SNRs × all ρ
- [ ] **Figure 1**: PSNR vs SNR (all methods + baselines + Shannon bound)
  - AWGN channel
  - Show cliff effect for digital, graceful degradation for learned
- [ ] **Figure 2**: PSNR vs SNR on Rayleigh fading
  - Same as above but fading → show robustness advantage
- [ ] **Figure 3**: PSNR vs Bandwidth Ratio ρ at fixed SNR
- [ ] **Figure 4**: Perceptual comparison (LPIPS/FID)
  - Diffusion should win here
- [ ] **Figure 5**: Latency vs Quality Pareto frontier
  - x-axis: inference time (ms)
  - y-axis: PSNR
  - Each method as a point; diffusion with varying steps as a curve
- [ ] **Figure 6**: Model-based interpretability
  - Learned S-K mapping visualization
  - Unrolled iteration convergence
  - Learned step sizes
- [ ] **Figure 7**: Visual examples
  - Side-by-side reconstructions at low SNR
  - Zoom-in on details
- [ ] **Table**: Parameter count, FLOPs, inference time, PSNR @ 0dB, PSNR @ 10dB

**Key files**:
```
src/eval/
  eval_all.py            # orchestrates full evaluation
  eval_latency.py        # latency profiling
  plot_figures.py         # matplotlib publication figures
outputs/evaluation/
  figures/               # all PDFs
  tables/                # LaTeX tables
  raw/                   # JSON results
```

**Definition of Done**: 7 publication-quality figures, 1 comparison table. All reproducible from `scripts/run_eval.sh`.

---

### Milestone 6: Documentation & Report (Stage 11-12)
**Status**: 🔴 Not started
**Time**: 4-6 hours
**Output**: Technical report, polished README, literature review

- [ ] **Technical report** (3-5 pages, workshop-paper format):
  - Introduction: problem of continuous-valued source coding
  - Related work: SK mappings, DeepJSCC, DiffJSCC, model-based DL
  - Method: three schemes with architectural diagrams
  - Experiments: all figures and tables
  - Discussion: when to use which scheme, limitations, future work
- [ ] **README.md**: professional repo documentation
  - Problem statement
  - Quick start (one-command reproduce)
  - Architecture diagrams
  - Results summary with figures
  - References
- [ ] **Literature review** (`docs/literature_review.md`):
  - Classical analog JSCC (Gastpar et al., Floor & Ramstad)
  - Deep JSCC (Bourtsoulatze et al., Kurka & Gündüz)
  - Diffusion for JSCC (DiffJSCC, SGD-JSCC, CDDM)
  - Model-based DL (Shlezinger et al.)
  - Gap analysis → motivation for this project

**Key files**:
```
docs/
  technical_report.pdf
  literature_review.md
  method_note.md
README.md
```

**Definition of Done**: Report reads like a workshop submission. README makes the repo look professional and immediately runnable.

---

## Timeline Summary

| Stage | Milestone | Deliverable |
|-------|-----------|-------------|
| 1 | M0 + M1 | Infra, data, baselines with cliff-effect plot |
| 2-4 | M2 | VAE-JSCC trained, graceful degradation shown |
| 4-7 | M3 | Diffusion-JSCC + consistency distillation |
| 7-9 | M4 | Model-based hybrid (Shlezinger-inspired) |
| 9-11 | M5 | All figures, tables, latency profiling |
| 11-12 | M6 | Report, README, lit review |

**Total**: ~12 working stages. Can compress to 7-8 stages with intense sprints.

---

## RunPod Setup

```bash
# RTX 5090 pod config
# Template: RunPod PyTorch 2.x
# GPU: RTX 5090 (32GB VRAM)
# Disk: 50 GB (container storage)
# Volume: 20 GB persistent at /workspace
#   → checkpoints survive pod restarts
#   → mount: /workspace

# On pod startup:
git clone https://github.com/Hudabey/diffusionjscc-speed.git /workspace/diffusionjscc-speed
cd /workspace/diffusionjscc-speed
pip install -r requirements.txt
bash scripts/setup_pod.sh
```

---

## Key References

1. Gastpar, Rimoldi & Vetterli, "To code or not to code," IEEE ISIT 2000
2. Floor & Ramstad, "Shannon–Kotel'nikov Mappings for Analog Point-to-Point Communications," IEEE TIT 2024
3. Xuan & Narayanan, "Low-Delay Analog JSCC With Deep Learning," IEEE TCOM 2023
4. **Shlezinger, Whang, Eldar & Dimakis, "Model-Based Deep Learning: Key Approaches and Design Guidelines," IEEE DSLW 2021** ← directly cited in JD
5. Bourtsoulatze, Kurka & Gündüz, "Deep JSCC for Wireless Image Transmission," IEEE TCCN 2019
6. Yang et al., "DiffJSCC: Diffusion-Aided JSCC for High Realism Transmission," arXiv 2024
7. Zhang et al., "SGD-JSCC: Semantics-Guided Diffusion for DeepJSCC," arXiv 2025

---

## Repo Structure (Final)

```
diffusionjscc-speed/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── vae_jscc.yaml
│   ├── diffusion_jscc.yaml
│   └── model_based.yaml
├── src/
│   ├── channel/
│   │   ├── awgn.py
│   │   ├── rayleigh.py
│   │   └── utils.py
│   ├── data/
│   │   ├── datasets.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── vae_jscc/
│   │   │   ├── encoder.py
│   │   │   ├── decoder.py
│   │   │   └── model.py
│   │   ├── diffusion_jscc/
│   │   │   ├── backbone.py
│   │   │   ├── diffusion.py
│   │   │   ├── consistency.py
│   │   │   └── model.py
│   │   └── model_based/
│   │       ├── encoder.py
│   │       ├── unrolled_decoder.py
│   │       └── model.py
│   ├── train/
│   │   ├── train_vae.py
│   │   ├── train_backbone.py
│   │   ├── train_diffusion.py
│   │   ├── train_consistency.py
│   │   └── train_model_based.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── digital_baselines.py
│   │   ├── eval_all.py
│   │   ├── eval_latency.py
│   │   └── plot_figures.py
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       └── logging.py
├── scripts/
│   ├── setup_pod.sh
│   ├── run_baselines.sh
│   ├── run_train_all.sh
│   └── run_eval.sh
├── tests/
│   ├── test_channel.py
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_models.py
├── docs/
│   ├── technical_report.pdf
│   ├── literature_review.md
│   └── method_note.md
└── outputs/           # gitignored, on persistent volume
    ├── baselines/
    ├── vae_jscc/
    ├── diffusion_jscc/
    ├── model_based/
    └── evaluation/
```
