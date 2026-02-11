# Deep Joint Source-Channel Coding: Convolutional, Diffusion, and Model-Based Approaches for Robust Image Transmission

**Hudheifa** — SRH Berlin University of Applied Sciences

---

## Abstract

Classical digital communication systems apply source and channel coding separately, which is optimal asymptotically but introduces a *cliff effect* in finite-blocklength regimes: image quality degrades catastrophically once the channel quality drops below a threshold. We implement and compare three learned joint source-channel coding (JSCC) schemes on a unified codebase: (1) a fully-convolutional DeepJSCC autoencoder with SNR-adaptive channel attention, (2) a diffusion-based receiver refinement module using conditional DDPM, and (3) a model-based JSCC with an unrolled iterative decoder inspired by Shannon-Kotel'nikov mappings. Evaluated on the Kodak dataset over AWGN channels, our DeepJSCC achieves 25.2 dB PSNR at SNR = 0 dB — a +13 dB improvement over JPEG with ideal channel coding — while maintaining graceful degradation across all SNR levels. The model-based approach achieves competitive quality (23.8 dB at SNR = 10 dB) with 6.6x fewer parameters (2.6M vs 17.5M). We provide an honest assessment of limitations, including weak SNR adaptation at high SNR, and identify concrete directions for future work.

## 1. Introduction

Shannon's separation theorem establishes that source and channel coding can be optimized independently without loss of optimality — but only in the limit of infinite blocklength and vanishing error probability [4]. In practical systems operating at finite blocklength, this separation is suboptimal. Digital communication systems exhibit a characteristic *cliff effect*: the received signal is either decoded perfectly (above a capacity threshold) or fails completely (below it), with no intermediate graceful degradation.

Joint source-channel coding (JSCC) maps source signals directly to channel symbols, bypassing the separate compress-then-protect pipeline. This approach can exploit the analog nature of the source to achieve graceful degradation — image quality degrades smoothly as channel conditions worsen, rather than collapsing to noise. Early theoretical work by Gastpar et al. [4] showed that uncoded transmission can be optimal for Gaussian sources over Gaussian channels, motivating the study of learned analog mappings.

Recent deep learning advances have produced several JSCC architectures. Bourtsoulatze et al. [1] introduced DeepJSCC, a fully-convolutional autoencoder that directly maps images to channel symbols. Subsequent work extended this with attention mechanisms (ADJSCC) and Swin Transformer backbones (SwinJSCC [3]) for improved performance at high SNR. Separately, model-based deep learning [2] proposes combining domain knowledge with learned components, using algorithm unrolling to produce interpretable, parameter-efficient architectures.

**Our contribution.** We provide a unified comparison of three JSCC paradigms implemented on a shared codebase with common evaluation infrastructure:

1. **DeepJSCC** — A fully-convolutional autoencoder with SNR-adaptive channel attention gating, trained with combined MSE and MS-SSIM loss.
2. **Diffusion-JSCC** — A conditional DDPM trained to refine initial DeepJSCC reconstructions, using DDIM sampling for controllable quality-latency tradeoff.
3. **Model-Based JSCC** — A Shannon-Kotel'nikov inspired encoder paired with an unrolled iterative decoder (K=6 iterations), achieving competitive quality with 6.6x fewer parameters.

All methods are evaluated against JPEG and WebP digital baselines with ideal (capacity-achieving) channel codes, providing a fair comparison framework.

## 2. System Model

We consider the transmission of natural images **x** in [0, 1]^(3 x H x W) over an additive white Gaussian noise (AWGN) channel. The encoder maps the source image to a latent representation **z** = f_enc(**x**, SNR), which is power-normalized to unit average energy and transmitted through the channel:

**y** = **z** + **n**, where **n** ~ N(0, sigma^2 * **I**)

The decoder reconstructs the image as **x_hat** = f_dec(**y**, SNR). Both encoder and decoder are conditioned on the channel signal-to-noise ratio (SNR) to enable adaptive behavior.

**Bandwidth ratio.** The compression ratio is defined as rho = k / (3 * H * W), where k is the number of real-valued channel uses. For our DeepJSCC with latent channels C_lat = 192 and spatial downsampling factor 16x, the bandwidth ratio is rho = C_lat / (3 * 16^2) = 0.25.

**Metrics.** We evaluate using PSNR (dB), SSIM, MS-SSIM, and LPIPS (learned perceptual metric). PSNR and SSIM measure pixel-level fidelity, while LPIPS captures perceptual quality.

**Datasets.** Models are trained on DIV2K (800 high-resolution images) with random 128x128 crops and horizontal flips. Evaluation is performed on the Kodak dataset (24 full-resolution images at 512x768).

## 3. Methods

### 3.1 Digital Baselines

We implement JPEG and WebP compression at multiple quality levels paired with ideal capacity-achieving channel codes. The channel capacity for AWGN at bandwidth ratio rho is:

C = rho * log2(1 + SNR_linear) bits/pixel

If the source coding rate exceeds the channel capacity, transmission fails completely and the receiver outputs the mean image (PSNR ~ 12.2 dB). This produces the characteristic cliff effect: JPEG jumps from 12.2 dB at SNR = 5 dB to 26.5 dB at SNR = 10 dB, with no intermediate quality levels.

We also compute a Shannon bound as a theoretical lower reference, representing the distortion achievable by an ideal Gaussian source at the capacity-limited rate.

### 3.2 DeepJSCC with SNR-Adaptive Attention

Our primary learned method is a fully-convolutional autoencoder following the DeepJSCC paradigm [1], with SNR-adaptive channel attention for conditioning.

**Encoder.** Four stride-2 downsampling stages produce a 16x spatial reduction. Each stage consists of a 5x5 strided convolution followed by two residual blocks with PReLU activation. Channel dimensions progress as 3 -> 64 -> 128 -> 256 -> 192, where the final 192 channels form the latent representation. An SNRAttention module is applied after each stage: the scalar SNR is projected through a 2-layer MLP and passed through a sigmoid gate to produce per-channel scaling factors, enabling the network to adapt its feature allocation based on channel quality.

**Decoder.** Four stride-2 transposed convolution stages mirror the encoder. Each stage applies two residual blocks followed by SNRAttention, then upsamples via transposed convolution. The output is projected to 3 channels with a sigmoid activation to produce the reconstruction in [0, 1].

**Channel interface.** The encoder output is power-normalized to unit average energy, then corrupted by AWGN at the specified SNR. This normalization ensures a fair comparison across methods and SNR levels.

**Loss function.** We use a combined loss: L = 0.7 * MSE + 0.3 * (1 - MS-SSIM). The MSE term provides stable gradients, while MS-SSIM encourages preservation of multi-scale structural features.

**Training.** 500 epochs on DIV2K with batch size 8, learning rate 1e-4 (Adam), and random SNR sampled uniformly from [-2, 20] dB per batch. The model has 17.5M parameters.

**Key design choice.** An earlier version used a VAE architecture with KL divergence regularization. This produced flat PSNR (~23.8 dB) across all SNR levels — the stochastic bottleneck destroyed SNR-dependent information. Removing the VAE components and using a deterministic autoencoder restored the expected graceful degradation behavior.

### 3.3 Model-Based JSCC

Inspired by Shlezinger et al. [2], our model-based approach combines domain knowledge (power normalization, channel model) with learned components via algorithm unrolling.

**Encoder.** A Shannon-Kotel'nikov (S-K) inspired encoder with three stride-2 stages (8x spatial reduction) maps images to a 12-channel latent space. FiLM (Feature-wise Linear Modulation) conditioning adapts features based on SNR. Built-in power normalization ensures unit energy per channel use.

**Unrolled iterative decoder.** The decoder runs K=6 iterations of a learned proximal gradient algorithm:

z_{k+1} = z_k - alpha_k * grad_channel(z_k) + D_theta_k(z_k, SNR)

Each iteration applies: (1) a gradient step with respect to the channel likelihood (known AWGN model), weighted by a *learned* step size alpha_k, and (2) a compact learned denoiser D_theta_k (~50K parameters each) that acts as a proximal operator. The learned step sizes provide interpretability — they can be inspected to understand how aggressively each iteration corrects the estimate.

**Reconstruction.** After K iterations, a reconstruction network with three 2x upsampling stages and FiLM conditioning maps the refined latent back to image space.

**Parameter efficiency.** The entire model has 2.6M parameters — 6.6x fewer than DeepJSCC. The compact design comes from exploiting the known channel model (AWGN gradient is analytical) rather than learning it from data.

**Convergence analysis.** The unrolled iterations are critical: K=1 yields only 10.1 dB, improving through 12.3 dB (K=2), 14.0 dB (K=3), 15.2 dB (K=4), 18.9 dB (K=5), to 23.8 dB (K=6) at SNR = 10 dB. This steep improvement curve suggests additional iterations could further improve quality.

### 3.4 Diffusion-Based Receiver

We implement a conditional DDPM [5] trained to refine the initial DeepJSCC reconstruction at the receiver. A UNet denoiser is conditioned on both the initial reconstruction and the channel SNR. At inference, we use DDIM sampling [6] with a configurable number of steps, enabling a quality-latency tradeoff.

**Current status.** The diffusion model was trained on the v1 (VAE-based) DeepJSCC backbone. Since the backbone was rewritten to v2, the diffusion refinement module needs retraining to match the updated feature distribution. Results from the diffusion module are therefore not included in the main comparison and are noted as future work.

## 4. Experimental Results

All methods are evaluated on the Kodak dataset (24 images, full resolution) at bandwidth ratio rho = 0.25 over an AWGN channel.

### 4.1 Graceful Degradation vs. Cliff Effect

The central result is shown in Figure 1 (`psnr_vs_snr_all_methods.png`). Digital baselines exhibit a sharp cliff effect: JPEG achieves 12.2 dB (mean image) for SNR <= 5 dB, jumps to 26.5 dB at SNR = 10 dB, and saturates at 34.1 dB for SNR >= 15 dB. WebP shows similar behavior with a slightly higher ceiling (36.7 dB).

In contrast, both learned methods degrade gracefully:

| Method | Params | SNR = -5 dB | SNR = 0 dB | SNR = 10 dB | SNR = 20 dB |
|--------|--------|-------------|------------|-------------|-------------|
| JPEG + Ideal Code | — | 12.2 | 12.2 | 26.5 | 34.1 |
| WebP + Ideal Code | — | 12.2 | 12.2 | 26.2 | 36.7 |
| DeepJSCC (Ours) | 17.5M | 23.9 | **25.2** | **25.8** | 25.8 |
| Model-Based (Ours) | 2.6M | 19.3 | 22.2 | 23.8 | 23.6 |

At SNR = 0 dB, DeepJSCC achieves **+13.0 dB over JPEG** (25.2 vs 12.2 dB). Even at the extremely harsh SNR = -5 dB, DeepJSCC still produces usable images at 23.9 dB, while digital methods output pure noise.

The cliff effect is clearly annotated in Figure 2 (`cliff_effect_annotated.png`), showing the critical SNR threshold (~8-10 dB) below which digital baselines fail completely.

### 4.2 Parameter Efficiency

Figure 3 (`param_efficiency.png`) illustrates the parameter-performance tradeoff. Model-Based JSCC achieves 23.8 dB at SNR = 10 dB with only 2.6M parameters, compared to DeepJSCC's 25.8 dB at 17.5M parameters. This is a 6.6x parameter reduction for a 2.0 dB PSNR cost — a compelling tradeoff for resource-constrained deployment.

Figure 4 (`model_based_convergence.png`) shows the iterative decoder's convergence: PSNR improves from 10.1 dB (K=1) to 23.8 dB (K=6), with the largest single-iteration jump between K=5 and K=6 (+4.9 dB), suggesting the algorithm has not yet converged and additional iterations may help.

### 4.3 Inference Latency

Figure 5 (`latency_comparison.png`) compares encoding and decoding times. DeepJSCC encodes in 5.4 ms and decodes in 5.6 ms. Model-Based JSCC encodes faster (2.8 ms, due to its smaller encoder) but decodes slower (5.9 ms, due to 6 iterative passes). Total end-to-end latency is comparable: ~11 ms for DeepJSCC vs ~8.7 ms for Model-Based. Both are suitable for real-time applications.

### 4.4 Visual Quality

Figure 6 (`visual_comparison.png`) shows reconstructed Kodak images at SNR = 5 dB. JPEG fails entirely at this SNR (below the cliff), producing the dataset mean. DeepJSCC preserves major structures and colors with mild blurring. Model-Based JSCC shows more blurring but retains recognizable content. At higher SNR (>= 12 dB), digital baselines produce sharper images than our learned methods due to their higher effective capacity ceiling.

### 4.5 Perceptual Metrics

SSIM results (`ssim_vs_snr.png`) closely track PSNR trends. DeepJSCC achieves SSIM = 0.75 across the SNR range, while Model-Based JSCC ranges from 0.39 (SNR = -5 dB) to 0.64 (SNR = 10 dB). LPIPS (`lpips_vs_snr.png`) confirms that DeepJSCC produces perceptually closer reconstructions (LPIPS = 0.38) compared to Model-Based (LPIPS = 0.52).

### 4.6 Limitations and Discussion

**We present an honest assessment of our results:**

**Weak SNR adaptation.** DeepJSCC PSNR ranges only from 23.9 dB (SNR = -5 dB) to 25.8 dB (SNR = 25 dB) — a span of merely ~2 dB across a 30 dB SNR range. A well-adapted system should show ~7-10 dB variation. The channel attention gating mechanism does not differentiate strongly enough: at high SNR, the network should allocate bandwidth to fine details, but instead produces the same reconstruction regardless. This is the single biggest limitation.

**High-SNR underperformance.** Above SNR = 12 dB, both JPEG (32.9 dB) and WebP (34.3 dB) substantially outperform DeepJSCC (25.8 dB). Our learned methods saturate early while digital baselines exploit the additional capacity. This gap would narrow with Transformer-based architectures (SwinJSCC) or explicit rate adaptation mechanisms.

**Model-Based performance ceiling.** The model-based decoder's 8x spatial reduction (vs 16x for DeepJSCC) results in a higher-dimensional but lower-capacity latent space (12 channels vs 192), limiting its reconstruction quality ceiling.

**Limited training data.** DIV2K contains only 800 training images. Training on ImageNet-scale data (1.2M images) would likely improve generalization and absolute performance.

**CNN architecture limitations.** Our convolutional backbone has limited receptive field and cannot capture long-range dependencies. Transformer-based approaches like SwinJSCC [3] address this at the cost of increased complexity.

**Diffusion module not integrated.** The diffusion refinement module was trained on the v1 backbone and needs retraining, leaving the three-way comparison incomplete.

## 5. Future Work

Several directions could address the identified limitations:

**Stronger SNR conditioning.** Replace channel attention with Transformer-based spatial modulation, as in SwinJSCC [3], where the attention mechanism can adapt spatial feature allocation based on SNR — not just channel-wise scaling.

**Rate adaptation.** Implement variable bandwidth ratio within a single model by masking subsets of latent channels at inference time, enabling bandwidth-quality tradeoffs without retraining.

**Swin Transformer backbone.** Replace the CNN encoder/decoder with Swin Transformer blocks for improved long-range dependency modeling, particularly benefiting high-SNR performance where fine detail preservation matters.

**Fading channels.** Extend evaluation to Rayleigh fading channels with estimated or perfect CSI. The AWGN channel module is already implemented; the fading channel and CSI estimation modules provide a natural extension.

**Model-based diffusion hybrid.** Combine the unrolled iterative decoder with diffusion-based refinement: use unrolled iterations for coarse reconstruction, then diffusion steps for perceptual enhancement. This could combine parameter efficiency with perceptual quality.

**Larger training data.** Scale training to ImageNet or the CLIC dataset for improved generalization. The training pipeline supports arbitrary image datasets with minimal modification.

**Consistency distillation.** Apply consistency models to reduce diffusion sampling from multiple steps to a single forward pass, enabling real-time diffusion-enhanced JSCC.

## 6. Conclusion

We implemented and compared three joint source-channel coding approaches on a unified codebase with shared evaluation infrastructure. Our DeepJSCC autoencoder achieves graceful degradation with +13 dB over digital baselines at SNR = 0 dB, confirming the fundamental advantage of learned JSCC in low-SNR regimes. The model-based approach demonstrates that domain knowledge (known channel model, power normalization constraints) can be leveraged for 6.6x parameter reduction with competitive quality, making it attractive for edge deployment. Honest analysis reveals that weak SNR adaptation — the network's inability to exploit high-SNR conditions — is the primary bottleneck, and points to Transformer-based spatial modulation as the most promising solution. This codebase provides a foundation for continued research toward a master thesis investigating stronger conditioning mechanisms and fading channel generalization.

## References

[1] E. Bourtsoulatze, D. Burth Kurka, and D. Gunduz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," *IEEE Trans. Cognitive Communications and Networking*, vol. 5, no. 3, pp. 567–579, 2019.

[2] N. Shlezinger, J. Whang, Y. C. Eldar, and A. G. Dimakis, "Model-Based Deep Learning: Key Approaches and Design Guidelines," *IEEE Data Science and Learning Workshop (DSLW)*, 2021.

[3] M. Yang, C. Bian, and H.-S. Kim, "SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding," *IEEE Trans. Cognitive Communications and Networking*, 2024.

[4] M. Gastpar, B. Rimoldi, and M. Vetterli, "To Code or Not to Code: Lossy Source-Channel Communication Revisited," *IEEE Trans. Information Theory*, vol. 49, no. 5, pp. 1147–1158, 2003.

[5] J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models," *NeurIPS*, 2020.

[6] J. Song, C. Meng, and S. Ermon, "Denoising Diffusion Implicit Models," *ICLR*, 2021.

[7] P. A. Floor and K. Ramstad, "Shannon-Kotel'nikov Mappings for Analog Point-to-Point Communications," *IEEE Trans. Information Theory*, 2024.
