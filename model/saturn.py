# =============================================================================
# model/saturn.py
# Saturn 0.1 — Math/Code-only Transformer
# MLA (Multi-head Latent Attention) + SwiGLU + Grouped Query Attention
# Designed for 4GB VRAM via layer-streaming, master weights in system RAM
# =============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLAAttention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek V2/V3 style).
    Compresses KV into a low-rank latent vector to reduce VRAM usage.
    """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, kv_rank: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.kv_rank = kv_rank

        # Q projection (full rank)
        self.w_q = nn.Linear(d_model, d_model, bias=False)

        # KV compression: project down to low-rank latent
        self.w_kv = nn.Linear(d_model, kv_rank * 2, bias=False)  # [k_latent, v_latent]
        # Decompress: from latent back to per-head K and V
        self.w_k_up = nn.Linear(kv_rank, n_kv_heads * self.head_dim, bias=False)
        self.w_v_up = nn.Linear(kv_rank, n_kv_heads * self.head_dim, bias=False)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # RoPE cache
        self.register_buffer("cos_cached", torch.empty(0))
        self.register_buffer("sin_cached", torch.empty(0))

    def _build_rope(self, seq_len: int, device: torch.device):
        if self.cos_cached.shape[0] >= seq_len:
            return
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]  # (1, 1, seq, head_dim)
        self.sin_cached = emb.sin()[None, None, :, :]

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_heads, seq, head_dim)
        batch, n_heads, seq, head_dim = x.shape
        self._build_rope(seq, x.device)
        cos = self.cos_cached[:, :, :seq, :]  # (1, 1, seq, head_dim)
        sin = self.sin_cached[:, :, :seq, :]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (batch, seq, d_model)
        batch, seq, _ = x.shape

        # Q: full projection then reshape to heads
        q = self.w_q(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, seq, head_dim)

        # KV: compress to latent, then decompress per KV head
        kv_latent = self.w_kv(x)  # (batch, seq, 2 * kv_rank)
        k_latent, v_latent = kv_latent.chunk(2, dim=-1)

        k = self.w_k_up(k_latent).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v_up(v_latent).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE on Q and K
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # Expand KV heads to match Q heads (GQA)
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # (batch, n_heads, seq, seq)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (batch, n_heads, seq, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.w_o(out)


class SaturnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, kv_rank: int, norm_eps: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model, norm_eps)
        self.attn = MLAAttention(d_model, n_heads, n_kv_heads, kv_rank)
        self.norm2 = RMSNorm(d_model, norm_eps)
        self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class SaturnTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.vocab_size = config["vocab_size"]
        self.context_len = config["context_len"]
        self.n_layers = config["n_layers"]

        self.embed = nn.Embedding(config["vocab_size"], config["d_model"])
        self.layers = nn.ModuleList([
            SaturnBlock(
                d_model=config["d_model"],
                n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"],
                d_ff=config["d_ff"],
                kv_rank=config["mla_kv_rank"],
                norm_eps=config["norm_eps"],
            )
            for _ in range(config["n_layers"])
        ])
        self.norm_f = RMSNorm(config["d_model"], config["norm_eps"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)

        # Weight tying
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # input_ids: (batch, seq)
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    def get_layer_params(self, layer_idx: int) -> dict:
        """Return state_dict slice for a specific layer (for CPU<->GPU streaming)."""
        prefix = f"layers.{layer_idx}."
        return {k[len(prefix):]: v for k, v in self.state_dict().items() if k.startswith(prefix)}

    def set_layer_params(self, layer_idx: int, params: dict):
        """Load a specific layer's params (for streaming back from GPU/CPU)."""
        prefix = f"layers.{layer_idx}."
        full = {prefix + k: v for k, v in params.items()}
        self.load_state_dict(full, strict=False)

    def causal_mask(self, seq: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq, seq, device=device)).view(1, 1, seq, seq)
        return mask
