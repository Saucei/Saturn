# =============================================================================
# runtime/gpu_window.py
# Saturn 0.1 — GPU Compute Window (4GB VRAM)
#
# Holds only the ACTIVE layer's weights on GPU at any time.
# Weights are streamed FP16 from ParamServer (RAM), activations streamed back.
# This keeps VRAM usage bounded regardless of total model size.
# =============================================================================
import torch
import torch.nn.functional as F
from model.saturn import SaturnTransformer, SaturnBlock


class GPUWindow:
    def __init__(self, model: SaturnTransformer, gpu_device: torch.device,
                 vram_limit_gb: float = 3.5):
        self.model = model
        self.device = gpu_device
        self.vram_limit_bytes = int(vram_limit_gb * 1024**3)
        self.n_layers = model.n_layers

        # Track which layer is currently resident on GPU
        self.current_layer = None
        self.layer_weights_gpu = {}  # layer_idx -> dict of FP16 tensors on GPU

        # Causal mask (built once, reused)
        self.register_buffer("causal_mask", model.causal_mask(model.context_len, gpu_device))

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def load_layer(self, layer_idx: int, param_server):
        """Load layer `layer_idx` weights from ParamServer into GPU (FP16)."""
        if self.current_layer == layer_idx and layer_idx in self.layer_weights_gpu:
            return  # Already resident

        # Evict previous layer to free VRAM
        if self.current_layer is not None and self.current_layer != layer_idx:
            del self.layer_weights_gpu[self.current_layer]
            torch.cuda.empty_cache()

        # Fetch FP32 from RAM, convert to FP16 on GPU
        layer_params = {}
        for name, fp32_tensor in param_server.master_params.items():
            if name.startswith(f"layers.{layer_idx}."):
                short_name = name[len(f"layers.{layer_idx}."):]
                layer_params[short_name] = fp32_tensor.half().to(
                    self.device, non_blocking=True
                )

        self.layer_weights_gpu[layer_idx] = layer_params
        self.current_layer = layer_idx

    def forward_layer(self, layer_idx: int, x: torch.Tensor,
                      param_server) -> torch.Tensor:
        """Run forward pass for one layer using GPU-resident weights."""
        self.load_layer(layer_idx, param_server)
        params = self.layer_weights_gpu[layer_idx]

        # Get the actual module to use its forward logic, but override weights
        block: SaturnBlock = self.model.layers[layer_idx]

        # Temporarily swap weights onto GPU params for computation
        # We do functional forward to avoid moving the whole module to GPU
        return self._functional_forward(block, x, params)

    def _functional_forward(self, block: SaturnBlock, x: torch.Tensor,
                            params: dict) -> torch.Tensor:
        """Manual forward through a SaturnBlock using provided GPU params."""
        # Attention
        h = block.norm1(x)
        # norm1 weight
        h = h * block.norm1.weight.half().to(self.device)
        h = h / torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + block.norm1.eps)

        # Q
        q = F.linear(h, params["attn.w_q.weight"])
        q = q.view(q.shape[0], q.shape[1], block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

        # KV latent
        kv_latent = F.linear(h, params["attn.w_kv.weight"])
        k_latent, v_latent = kv_latent.chunk(2, dim=-1)
        k = F.linear(k_latent, params["attn.w_k_up.weight"])
        v = F.linear(v_latent, params["attn.w_v_up.weight"])
        k = k.view(k.shape[0], k.shape[1], block.attn.n_kv_heads, block.attn.head_dim).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], block.attn.n_kv_heads, block.attn.head_dim).transpose(1, 2)

        # RoPE
        q = self._rope(q)
        k = self._rope(k)

        # GQA expand
        if block.attn.n_kv_heads < block.attn.n_heads:
            rep = block.attn.n_heads // block.attn.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        scale = 1.0 / math.sqrt(block.attn.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        seq = q.shape[-2]
        mask = self.causal_mask[:, :, :seq, :seq]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(out.shape[0], seq, -1)
        out = F.linear(out, params["attn.w_o.weight"])
        x = x + out

        # MLP
        h = block.norm2(x)
        h = h * block.norm2.weight.half().to(self.device)
        h = h / torch.sqrt(h.pow(2).mean(dim=-1, keepdim=True) + block.norm2.eps)
        h = F.silu(F.linear(h, params["mlp.w1.weight"])) * F.linear(h, params["mlp.w3.weight"])
        h = F.linear(h, params["mlp.w2.weight"])
        x = x + h

        return x

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_heads, seq, head_dim = x.shape
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim))
        t = torch.arange(seq, device=x.device).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def forward_all_layers(self, x: torch.Tensor, param_server) -> torch.Tensor:
        """Forward through all layers sequentially, streaming one at a time."""
        for i in range(self.n_layers):
            x = self.forward_layer(i, x, param_server)
        return x

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy loss on GPU (last layer only)."""
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

    def clear(self):
        self.layer_weights_gpu.clear()
        self.current_layer = None
        torch.cuda.empty_cache()
