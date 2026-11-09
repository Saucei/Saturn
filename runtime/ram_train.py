"""
Saturn 0.1 — RAM-Primary Training Script (Windows 10)

Design philosophy:
- Model weights, optimizer states, and gradients live in SYSTEM RAM (16 GB).
- GPU (Quadro M1200, 4 GB VRAM) is SECONDARY — used only for optional matmul offload.
- Works entirely on CPU if no CUDA drivers are available.
- Memory-mapped dataset allows corpus larger than RAM.

Run:
    python runtime/ram_train.py --config configs/ram-training.yaml --model Qwen/Qwen2.5-1.5B-Instruct --strategy full
    python runtime/ram_train.py --config configs/ram-training.yaml --strategy from_scratch
"""

import os
import sys
import argparse
import math
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_lr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.saturn import SaturnTransformer
from data.pipeline import MathTokenizer, MathCorpusDataset, collate_batch


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def estimate_ram_params(param_count: int, strategy: str, dtype_bytes: int = 2) -> dict:
    """Estimate RAM usage for training."""
    if strategy == "full":
        # weights (FP16) + adam (FP32 x2) + grads (FP16) = ~12 bytes/param
        bytes_per_param = dtype_bytes + 8 + dtype_bytes
    elif strategy == "qlora":
        # base model INT4 (0.5 bytes) + adapters FP16 + adapter optimizer only
        bytes_per_param = 0.5 + 2 + 8  # rough upper bound for adapters
    else:  # from_scratch
        bytes_per_param = dtype_bytes + 8 + dtype_bytes

    total_gb = (param_count * bytes_per_param) / (1024 ** 3)
    return {"bytes_per_param": bytes_per_param, "estimated_gb": total_gb}


def build_model_from_scratch(cfg: dict, device: torch.device):
    model = SaturnTransformer(cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    est = estimate_ram_params(n_params, "full")
    print(f"[model] From-scratch Saturn: {n_params:,} parameters")
    print(f"[model] Estimated RAM for full fine-tune: {est['estimated_gb']:.2f} GB")
    return model


def load_base_model(model_id: str, cfg: dict, device: torch.device):
    print(f"[model] Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load into system RAM first (CPU), optionally offload matmul to GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=None,        # Keep in RAM; move batches to GPU manually if needed
        trust_remote_code=True,
        low_cpu_mem_usage=True, # Reduce peak RAM during load
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Base model loaded: {n_params:,} parameters (~{n_params * 2 / 1024**2:.1f} MB FP16)")

    return model, tokenizer


def apply_qlora(model, cfg: dict):
    """Apply QLoRA adapters for parameter-efficient fine-tuning (RAM-friendly)."""
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError:
        print("[warn] peft not installed. Install with: pip install peft")
        raise

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class RamTrainer:
    def __init__(self, config_path: str, model_id: str = None, resume_from: str = None):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # --- Device: CPU primary, GPU optional ---
        self.use_gpu = self.cfg["training"].get("use_gpu_offload", False) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"[trainer] Primary device: {self.device}")
        if self.use_gpu:
            print(f"[trainer] GPU: {torch.cuda.get_device_name(0)} (VRAM secondary)")
        else:
            print(f"[trainer] Running on CPU. System RAM is the primary constraint.")

        self.model_id = model_id
        self.strategy = self.cfg["training"]["strategy"]

        # --- Model ---
        if self.strategy == "from_scratch":
            self.model = build_model_from_scratch(self.cfg, self.device)
            self.tokenizer = MathTokenizer(vocab_target=self.cfg["data"]["vocab_target"])
        else:
            self.model, self.tokenizer = load_base_model(model_id, self.cfg, self.device)
            if self.strategy == "qlora":
                self.model = apply_qlora(self.model, self.cfg)
            elif self.strategy == "full":
                pass

        # Move model to device (CPU or GPU)
        self.model.to(self.device)

        # --- Data (memory-mapped; dataset can exceed RAM) ---
        self.dataset = MathCorpusDataset(
            tokenizer=self.tokenizer,
            mmap_path=self.cfg["data"]["mmap_path"],
            seq_len=self.cfg["data"]["seq_len"],
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=self.use_gpu,  # Pin only if GPU transfers
            collate_fn=collate_batch,
        )

        # --- Optimizer (FP32 master weights in RAM) ---
        if self.cfg["training"].get("optimizer_8bit", False):
            try:
                from bitsandbytes.optim import AdamW8bit
                self.optimizer = AdamW8bit(
                    self.model.parameters(),
                    lr=self.cfg["training"]["learning_rate"],
                    weight_decay=self.cfg["training"]["weight_decay"],
                )
                print("[trainer] Using 8-bit Adam (reduced optimizer RAM)")
            except ImportError:
                print("[warn] bitsandbytes not installed; falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.cfg["training"]["learning_rate"],
                    weight_decay=self.cfg["training"]["weight_decay"],
                )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg["training"]["learning_rate"],
                weight_decay=self.cfg["training"]["weight_decay"],
            )

        self.scheduler = get_cosine_schedule_with_lr(
            self.optimizer,
            num_warmup_steps=self.cfg["training"]["warmup_steps"],
            num_training_steps=self.cfg["training"]["max_steps"],
        )

        # --- Training state ---
        self.step = 0
        self.accum_steps = self.cfg["training"]["gradient_accumulation_steps"]
        self.max_steps = self.cfg["training"]["max_steps"]
        self.grad_clip = self.cfg["training"]["grad_clip"]
        self.grad_checkpoint = self.cfg["training"].get("gradient_checkpointing", True)

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("data/tokenized", exist_ok=True)

    def train_step(self, batch) -> float:
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        targets = batch["targets"].to(self.device)

        seq = input_ids.shape[1]
        # Causal mask on the same device as input
        mask = torch.tril(torch.ones(seq, seq, device=self.device)).view(1, 1, seq, seq)

        outputs = self.model(input_ids, attention_mask=mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), targets.view(-1)
        )
        loss = loss / self.accum_steps
        loss.backward()

        return loss.item()

    def optimizer_step(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.step += 1

    def save_checkpoint(self, path: str):
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
        }
        torch.save(state, path)
        print(f"[trainer] Checkpoint saved: {path} ({os.path.getsize(path) / 1024**2:.1f} MB)")

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step = state["step"]
        print(f"[trainer] Resumed from step {self.step}")

    def train(self, resume_from: str = None):
        if resume_from and os.path.exists(resume_from):
            self.load_checkpoint(resume_from)

        print(f"[trainer] Starting training at step {self.step}...")
        print(f"[trainer] Strategy: {self.strategy} | Device: {self.device}")
        print(f"[trainer] Max steps: {self.max_steps} | Accum steps: {self.accum_steps}")

        dataloader_iter = iter(self.dataloader)

        while self.step < self.max_steps:
            total_loss = 0.0
            for micro in range(self.accum_steps):
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(self.dataloader)
                    batch = next(dataloader_iter)

                loss = self.train_step(batch)
                total_loss += loss

            self.optimizer_step()

            if self.step % 100 == 0:
                print(f"[trainer] step={self.step} loss={total_loss:.4f}")

            if self.step % self.cfg["training"]["save_every"] == 0:
                self.save_checkpoint(f"checkpoints/saturn-0.1-step{self.step}.pt")

        self.save_checkpoint("checkpoints/saturn-0.1-final.pt")
        print(f"[trainer] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ram-training.yaml")
    parser.add_argument("--model", default=None, help="HuggingFace model ID (e.g., Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--strategy", default=None, choices=["full", "qlora", "from_scratch"])
    parser.add_argument("--resume", default=None)
    parser.add_argument("--gpu-offload", action="store_true", help="Offload matmul to GPU")
    args = parser.parse_args()

    # Override config from CLI
    import yaml as _yaml
    with open(args.config) as f:
        cfg = _yaml.safe_load(f)
    if args.strategy:
        cfg["training"]["strategy"] = args.strategy
    if args.gpu_offload:
        cfg["training"]["use_gpu_offload"] = True
    with open(args.config, "w") as f:
        _yaml.dump(cfg, f)

    trainer = RamTrainer(args.config, model_id=args.model, resume_from=args.resume)
    trainer.train()
"""
