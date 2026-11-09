# =============================================================================
# runtime/param_server.py
# Saturn 0.1 — FP32 Master Parameter Server (lives in SYSTEM RAM)
#
# Holds all model weights, gradients, and optimizer state in system RAM.
# Serves FP16 copies to the GPU compute window one layer at a time.
# Accepts FP16 gradients back from GPU and accumulates in FP32.
#
# This is the "use all 16GB RAM" component: weights + optimizer + grads
# all live here, never evicted to disk during training.
# =============================================================================
import torch
from collections import OrderedDict
from model.saturn import SaturnTransformer


class ParamServer:
    def __init__(self, model: SaturnTransformer, lr: float = 3e-4,
                 weight_decay: float = 0.01, betas=(0.9, 0.95), eps: float = 1e-8):
        self.master_params = OrderedDict()
        self.shapes = {}
        self.param_names = []

        for name, param in model.named_parameters():
            # FP32 master in system RAM (pinned for fast GPU transfer)
            self.master_params[name] = param.detach().clone().float()
            self.shapes[name] = tuple(param.shape)
            self.param_names.append(name)
            # Detach original so model doesn't hold duplicate GPU memory
            param.data = None

        self.param_count = sum(p.numel() for p in self.master_params.values())
        print(f"[param-server] Master params: {self.param_count:,} "
              f"({self.param_count * 4 / 1024**2:.1f} MB FP32)")

        # Optimizer state (FP32 AdamW, lives in RAM)
        decay = []
        no_decay = []
        for name, p in self.master_params.items():
            if any(nd in name for nd in ["bias", "norm", "embed", "lm_head"]):
                no_decay.append(p)
            else:
                decay.append(p)

        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=betas, eps=eps,
        )

        self.step_count = 0

    def get_fp16(self, name: str, gpu_device: torch.device) -> torch.Tensor:
        """Return FP16 copy of master param on GPU (pinned, non-blocking)."""
        return self.master_params[name].half().to(gpu_device, non_blocking=True)

    def put_grad(self, name: str, grad_fp16_gpu: torch.Tensor):
        """Accumulate FP16 gradient from GPU into FP32 master grad in RAM."""
        grad_fp32 = grad_fp16_gpu.float().to("cpu", non_blocking=True)
        p = self.master_params[name]
        if p.grad is None:
            p.grad = grad_fp32.clone()
        else:
            p.grad += grad_fp32

    def zero_grad(self):
        for p in self.master_params.values():
            p.grad = None

    def step(self):
        self.optimizer.step()
        self.zero_grad()
        self.step_count += 1

    def state_dict(self) -> dict:
        return {name: p.clone() for name, p in self.master_params.items()}

    def load_state_dict(self, state: dict):
        for name, p in self.master_params.items():
            p.copy_(state[name])

    def memory_mb(self) -> float:
        total = sum(p.numel() * p.element_size() for p in self.master_params.values())
        # Include optimizer state (exp_avg + exp_avg_sq per param, ~2x)
        opt_total = total * 2
        return (total + opt_total) / 1024**2
