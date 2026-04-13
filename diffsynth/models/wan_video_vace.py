import torch
import torch.nn as nn
from .wan_video_dit import DiTBlock, CrossAttention, flash_attention
from ..core.gradient import gradient_checkpoint_forward


class GlyphCrossAttention(nn.Module):
    """Cross-attention for glyph feature injection.

    Unlike text cross-attention (which attends to language tokens),
    this attends to spatial glyph features that encode WHERE and WHAT
    text characters should be rendered. Each spatial position in the
    VACE hidden states queries the compressed glyph tokens to determine
    how the target text should appear at that location.

    Zero-initialized output projection ensures the model starts from
    pretrained VACE behavior and gradually learns to use glyph guidance.
    """

    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim, eps=eps)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        # Zero-init output projection so glyph attention starts as no-op
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x, glyph_tokens):
        """
        Args:
            x: VACE hidden states (B, seq_len, dim)
            glyph_tokens: compressed glyph features (B, num_tokens, dim)
        Returns:
            x + glyph_attn_output (B, seq_len, dim)
        """
        residual = x
        x = self.norm(x)
        q = self.q(x)
        k = self.k(glyph_tokens)
        v = self.v(glyph_tokens)
        out = flash_attention(q, k, v, self.num_heads)
        return residual + self.o(out)


class VaceWanAttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs, glyph_tokens=None):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)

        # Glyph cross-attention (injected after text cross-attn + FFN)
        if glyph_tokens is not None and hasattr(self, 'glyph_cross_attn'):
            c = self.glyph_cross_attn(c, glyph_tokens)

        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class GlyphEncoder(nn.Module):
    """Lightweight encoder that compresses glyph latents into tokens
    for cross-attention in VACE blocks.

    glyph_latent (16ch, from VAE) → Conv3D patch embed → spatial pooling
    → compact token sequence that each VACE block attends to.
    """

    def __init__(self, in_channels=16, dim=1536, num_tokens=64, patch_size=(1, 2, 2)):
        super().__init__()
        self.num_tokens = num_tokens

        # Patch embedding (same architecture as vace_patch_embedding)
        self.patch_embed = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

        # Compress spatial sequence to fixed number of tokens via cross-attention pooling
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.pool_norm = nn.LayerNorm(dim)

        # Output projection (zero-init for safe initialization)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, glyph_latent):
        """
        Args:
            glyph_latent: (B, 16, T, H, W) from VAE-encoded glyph video
        Returns:
            glyph_tokens: (B, num_tokens, dim) compressed glyph features
        """
        # Patch embed: (B, dim, T, H/2, W/2)
        x = self.patch_embed(glyph_latent)
        B = x.shape[0]
        # Flatten spatial: (B, dim, N) → (B, N, dim)
        x = x.flatten(2).transpose(1, 2)

        # Cross-attention pooling: compress N spatial tokens → num_tokens
        queries = self.query_tokens.expand(B, -1, -1).to(dtype=x.dtype, device=x.device)
        x_normed = self.pool_norm(x)
        pooled, _ = self.pool_attn(queries, x_normed, x_normed)

        return self.out_proj(pooled)


class VaceWanModel(torch.nn.Module):
    def __init__(
        self,
        vace_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        vace_in_dim=96,
        glyph_channels=0,
        glyph_num_tokens=64,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        self.glyph_channels = glyph_channels
        self.vace_layers_mapping = {i: n for n, i in enumerate(self.vace_layers)}

        # VACE blocks
        self.vace_blocks = torch.nn.ModuleList([
            VaceWanAttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.vace_layers
        ])

        # VACE patch embedding (original 96 channels, no glyph concat)
        self.vace_patch_embedding = torch.nn.Conv3d(
            vace_in_dim, dim, kernel_size=patch_size, stride=patch_size
        )

        # Glyph pathway (only if glyph_channels > 0)
        if glyph_channels > 0:
            # Separate glyph encoder: compress glyph latents into tokens
            self.glyph_encoder = GlyphEncoder(
                in_channels=glyph_channels,
                dim=dim,
                num_tokens=glyph_num_tokens,
                patch_size=patch_size,
            )
            # Add glyph cross-attention to each VACE block
            for block in self.vace_blocks:
                block.glyph_cross_attn = GlyphCrossAttention(dim, num_heads, eps)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.glyph_channels > 0:
            # Glyph modules are new and won't be in pretrained checkpoints.
            # First, materialize any meta-tensor parameters so they can be assigned.
            for name, param in self.named_parameters():
                if param.is_meta:
                    materialized = torch.zeros(param.shape, dtype=param.dtype, device='cpu')
                    # Use the module hierarchy to set the parameter
                    parts = name.split('.')
                    module = self
                    for p in parts[:-1]:
                        module = getattr(module, p)
                    setattr(module, parts[-1], torch.nn.Parameter(materialized))

            result = super().load_state_dict(state_dict, strict=False, assign=assign)

            # Re-initialize glyph modules with proper values (they were loaded as zeros above)
            if hasattr(self, 'glyph_encoder'):
                # Re-init the modules that should NOT be zero
                for name, module in self.glyph_encoder.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv3d)):
                        if 'out_proj' not in name:  # keep out_proj as zero-init
                            torch.nn.init.xavier_uniform_(module.weight)
                            if module.bias is not None:
                                torch.nn.init.zeros_(module.bias)
                # query_tokens
                if hasattr(self.glyph_encoder, 'query_tokens'):
                    torch.nn.init.normal_(self.glyph_encoder.query_tokens, std=0.02)

            return result
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self, x, vace_context, context, t_mod, freqs,
        glyph_latent=None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])

        # Encode glyph features into compact tokens (if available)
        glyph_tokens = None
        if glyph_latent is not None and hasattr(self, 'glyph_encoder'):
            glyph_tokens = self.glyph_encoder(glyph_latent)

        for block in self.vace_blocks:
            c = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing,
                use_gradient_checkpointing_offload,
                c, x, context, t_mod, freqs, glyph_tokens
            )

        hints = torch.unbind(c)[:-1]
        return hints
