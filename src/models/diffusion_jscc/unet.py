"""Conditional UNet for diffusion-based channel decoding.

Takes concatenated (x_noisy_t, x_init) as input, conditioned on diffusion
timestep t and channel SNR. Predicts noise epsilon.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    """Pick a valid GroupNorm group count for the given channel count."""
    for g in (32, 16, 8, 4, 1):
        if channels % g == 0:
            return g
    return 1


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        timesteps: (B,) integer timesteps.
        dim: Embedding dimension (must be even).

    Returns:
        (B, dim) embedding vectors.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU, and conditioning injection."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # FiLM conditioning: (1 + gamma) * features + beta
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, in_ch, H, W) input features.
            cond: (B, cond_dim) conditioning vector (time + SNR).

        Returns:
            (B, out_ch, H, W) output features.
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        # Inject conditioning via FiLM
        gamma_beta = self.cond_proj(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        h = (1 + gamma) * self.norm2(h) + beta

        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention for spatial feature maps."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # (B, heads, head_dim, N) -> (B, heads, N, head_dim) for SDPA
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2)  # back to (B, heads, head_dim, N)

        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DownBlock(nn.Module):
    """Downsampling block: ResBlocks + optional attention + downsample."""

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int,
        num_res_blocks: int = 2, use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            self.res_blocks.append(ResBlock(in_ch if i == 0 else out_ch, out_ch, cond_dim))
            self.attn_blocks.append(SelfAttention(out_ch) if use_attention else nn.Identity())
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        h = x
        for res, attn in zip(self.res_blocks, self.attn_blocks):
            h = res(h, cond)
            h = attn(h)
            skips.append(h)
        h = self.downsample(h)
        return h, skips


class UpBlock(nn.Module):
    """Upsampling block: upsample + ResBlocks with skip connections + optional attention."""

    def __init__(
        self, in_ch: int, out_ch: int, cond_dim: int,
        num_res_blocks: int = 2, use_attention: bool = False,
        skip_ch: int | None = None,
    ) -> None:
        super().__init__()
        if skip_ch is None:
            skip_ch = out_ch
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
        )
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            # First block takes skip connection (in_ch + skip_ch)
            ch_in = in_ch + skip_ch if i == 0 else out_ch
            self.res_blocks.append(ResBlock(ch_in, out_ch, cond_dim))
            self.attn_blocks.append(SelfAttention(out_ch) if use_attention else nn.Identity())

    def forward(
        self, x: torch.Tensor, skips: list[torch.Tensor], cond: torch.Tensor
    ) -> torch.Tensor:
        h = self.upsample(x)
        # Pad if sizes don't match after upsample
        if h.shape[2:] != skips[-1].shape[2:]:
            h = F.pad(h, (0, skips[-1].shape[3] - h.shape[3],
                          0, skips[-1].shape[2] - h.shape[2]))
        for i, (res, attn) in enumerate(zip(self.res_blocks, self.attn_blocks)):
            if i == 0:
                h = torch.cat([h, skips.pop()], dim=1)
            h = res(h, cond)
            h = attn(h)
        return h


class ConditionalUNet(nn.Module):
    """Conditional UNet for diffusion noise prediction.

    Takes concatenated (x_noisy_t, x_init) as input, conditioned on
    diffusion timestep and channel SNR via sinusoidal + learned embeddings.
    """

    # SNR normalization range (shared with VAE-JSCC)
    SNR_MIN = -5.0
    SNR_MAX = 25.0

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_levels: tuple[int, ...] = (2,),
        time_embed_dim: int = 256,
        snr_embed_dim: int = 256,
    ) -> None:
        """Initialize conditional UNet.

        Args:
            in_channels: Input channels (x_noisy + x_init concatenated).
            out_channels: Output channels (predicted noise).
            base_channels: Base channel count.
            channel_mults: Channel multipliers per level.
            num_res_blocks: ResBlocks per level.
            attention_levels: Which levels get self-attention (0-indexed).
            time_embed_dim: Timestep embedding dimension.
            snr_embed_dim: SNR embedding dimension.
        """
        super().__init__()
        cond_dim = time_embed_dim  # time + SNR combined

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # SNR embedding
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, time_embed_dim),
        )

        self.time_embed_dim = time_embed_dim

        # Input conv
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = level in attention_levels
            self.down_blocks.append(
                DownBlock(ch, out_ch, cond_dim, num_res_blocks, use_attn)
            )
            ch = out_ch

        # Middle
        self.mid_res1 = ResBlock(ch, ch, cond_dim)
        self.mid_attn = SelfAttention(ch)
        self.mid_res2 = ResBlock(ch, ch, cond_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(channel_mults))):
            skip_ch = base_channels * channel_mults[level]  # encoder's output at this level
            out_ch = base_channels * channel_mults[level - 1] if level > 0 else base_channels
            use_attn = level in attention_levels
            self.up_blocks.append(
                UpBlock(ch, out_ch, cond_dim, num_res_blocks, use_attn, skip_ch=skip_ch)
            )
            ch = out_ch

        # Output conv
        self.output_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        x_init: torch.Tensor,
        t: torch.Tensor,
        snr_db: float,
    ) -> torch.Tensor:
        """Predict noise in x_noisy, conditioned on x_init, timestep, and SNR.

        Args:
            x_noisy: Noisy image at diffusion step t (B, 3, H, W).
            x_init: VAE-JSCC reconstruction condition (B, 3, H, W).
            t: Diffusion timesteps (B,) integers.
            snr_db: Channel SNR in dB (scalar).

        Returns:
            Predicted noise (B, 3, H, W).
        """
        B = x_noisy.shape[0]

        # Time embedding
        t_emb = sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)

        # SNR embedding (normalized to [0, 1])
        snr_norm = (snr_db - self.SNR_MIN) / (self.SNR_MAX - self.SNR_MIN)
        snr_t = torch.tensor([[snr_norm]], device=x_noisy.device, dtype=x_noisy.dtype).expand(B, 1)
        snr_emb = self.snr_mlp(snr_t)

        # Combined conditioning
        cond = t_emb + snr_emb

        # Concatenate input and condition image
        h = torch.cat([x_noisy, x_init], dim=1)
        h = self.input_conv(h)

        # Encoder — collect skip connections
        all_skips = []
        for down in self.down_blocks:
            h, skips = down(h, cond)
            all_skips.append(skips)

        # Middle
        h = self.mid_res1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_res2(h, cond)

        # Decoder — consume skip connections in reverse
        for up, skips in zip(self.up_blocks, reversed(all_skips)):
            h = up(h, skips, cond)

        # Output
        h = self.output_conv(F.silu(self.output_norm(h)))
        return h

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
