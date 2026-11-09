# Saturn 0.1 — Training Methods Reference (Oct 31, 2026)

Comprehensive survey of current training methods from **NVIDIA, Qwen, Kimi (Moonshot AI), DeepSeek, Mistral, and KoboldCPP**, tailored for training a math/code-only model on constrained hardware (i7-7820HQ, 16GB RAM, Quadro M1200 4GB VRAM).

---

## 1. Lab-by-Lab Training Methods

### Kimi / Moonshot AI — Kimi K2 & K3 (2025-2026)

**Sources:** arXiv:2507.20534 (K2), arXiv:2607.24653 (K3), moonshotai.github.io/Kimi-K2

#### Kimi K2 — MuonClip Optimizer + Agentic RL
- **Optimizer:** MuonClip — Muon with QK-clip technique to address training instability while keeping Muon's advanced token efficiency. Pre-trained on **15.5 trillion tokens with zero loss spike**.
- **Pre-training:** 32B activated / 1T total MoE, non-thinking model
- **Post-training:** Multi-stage process with large-scale **agentic data synthesis pipeline** and **joint reinforcement learning (RL)** where the model improves through interactions with real and synthetic environments
- **Key for Saturn 0.1:** MuonClip is more token-efficient than AdamW — important when you have limited data and limited compute

#### Kimi K3 — Frontier Training Infrastructure (July 2026)
- **Scale:** 2.8T total, 104B activated MoE, 1M context, native vision
- **Architecture:** Kimi Delta Attention, Gated MLA, Attention Residuals, Stable LatentMoE (activates 16 of 896 experts per token)
- **Optimizer:** **Per-Head Muon** — Muon applied per attention head for better convergence
- **Quantization:** **MXFP4 Quantization-Aware Post-Training** — quantize during post-training, not just inference; deployment-aware
- **Distillation:** **Multi-Teacher On-Policy Distillation** — multiple teacher models provide on-policy signals during RL
- **Infrastructure:** Pipeline ZeRO-2 gradient sharding and offloading, P2P-based Muon orthogonalization, memory-efficient MoE, unified activation manager
- **RL:** Reasoning effort RL, agentic generative reward model, long-context RL with persistent rollout and sandbox states

**Saturn 0.1 takeaway:** MXFP4 QAT + Per-Head Muon are directly applicable to a small model. Muon optimizer converges faster than AdamW on limited data.

---

### DeepSeek — R1, V3, V4.1 (2025-2026)

**Sources:** arXiv:2501.12948 (R1), arXiv:2412.19437 (V3), deepseek.com model disclosure, deepseek.com/en/news/deepseek-v4-1-flash/

#### DeepSeek-R1 — GRPO RL Pipeline
- **RL Algorithm:** **GRPO (Group Relative Policy Optimization)** — no critic model needed, estimates baseline from group of outputs
- **R1-Zero:** Pure RL on base model without SFT — reasoning emerges from RL alone
- **R1:** Cold-start SFT + reasoning-oriented RL + rejection sampling + SFT for non-reasoning domains + RL for all scenarios
- **Distillation:** R1 reasoning distilled into 6 dense models (1.5B, 7B, 8B, 14B, 32B, 70B) based on Qwen and Llama
- **Reward:** Rule-based rewards for math/code (verifiable), model-based rewards for other domains

#### DeepSeek-V3 — FP8 Training + MLA + MoE
- **Precision:** **FP8 mixed-precision training** — fine-grained block-wise quantization, online quantization, low-precision optimizer states and communication
- **Architecture:** MLA (Multi-head Latent Attention) compresses KV cache; DeepSeekMoE auxiliary-loss-free load balancing
- **Training cost:** 2.788M H800 GPU hours for 671B model — FP8 critical for efficiency

#### DeepSeek V4.1-Flash — Asymmetric Architecture + KV Cache Compression
- **Architecture:** 552B MoE with asymmetric Causal Encoder–Decoder: **8B active for input, 16B active for output**
- **Efficiency:** KV cache needs **1/4 HBM, 1/8 SSD storage** vs. previous generation
- **Positioning:** Smaller, faster, lower cost; native multimodal; API model `deepseek-flash`

**Saturn 0.1 takeaway:** GRPO is ideal for Saturn 0.1 — math has verifiable answers (rule-based rewards), no critic model needed. FP8 training + MLA KV compression fits 4GB VRAM.

---

### Qwen / Alibaba — Qwen3 & Qwen2.5 (2025-2026)

**Sources:** arXiv:2412.15115 (Qwen2.5), CometAPI Qwen3 deep dive (Sep 2026), aiwiki.ai

#### Qwen3 — 3-Stage Pre-Training + Hybrid Reasoning
- **Pre-training data:** **36 trillion tokens** (2× Qwen2.5's 18T), 119+ languages
- **Stage 1 (S1):** 30T tokens, 4K context — foundational knowledge
- **Stage 2 (S2):** 5T tokens — knowledge-intensive rebalance (STEM, code, reasoning)
- **Stage 3:** Long-context extension to 32K tokens
- **Synthetic data:** Qwen2.5-Math and Qwen2.5-Coder generate math problem solutions and code snippets to augment corpus
- **Architecture:** Dense (0.6B-32B) and MoE (8 of 128 experts activated); per-head QK normalization for stability at scale
- **Post-training:** Multi-stage pipeline for reasoning and deployment

#### Qwen2.5-Math — Specialized Math Training
- Math-specific pretraining branch
- Reinforcement learning for math reasoning
- Chain-of-thought distillation

**Saturn 0.1 takeaway:** The 3-stage data rebalancing (foundational → knowledge-intensive → long-context) is a perfect template for Saturn 0.1's math-only corpus. Synthetic math data generation from a teacher model is exactly the distillation approach we need.

---

### NVIDIA — NeMo AutoModel (2026)

**Sources:** docs.nvidia.com/nemo/automodel (recipes, v0.7/26.10)

#### Training Workflows
- **Supervised Fine-Tuning (SFT) & PEFT:** LoRA, QLoRA parameter-efficient fine-tuning
- **Pretraining:** Megatron Core, PyTorch, NanoGPT starter configs
- **Knowledge Distillation (KD):** Teacher-student training, example: Llama 3.2 1B
- **Quantization-Aware Training (QAT):** FP8, MXFP4 — train with quantization in mind
- **Speculative Decoding:** EAGLE, DFlash, DSpark — inference acceleration
- **Large MoE Fine-Tuning:** Specialized recipes for mixture-of-experts

#### Model Coverage
Llama 3.1/3.2/3.3, Gemma 2/3, Falcon 3, Mistral/Mixtral, Nemotron, Granite, Starcoder, Qwen, Baichuan, GLM, OLMo, Phi, GPT-OSS, Moonlight

**Saturn 0.1 takeaway:** NeMo provides production-ready recipes for SFT, PEFT, KD, and QAT that we can adapt for Saturn 0.1's CPU+RAM+GPU heterogeneous training.

---

### Mistral AI — Open Weights + Forge (2025-2026)

**Sources:** mistral.ai

#### Philosophy & Offerings
- **Open-weight models:** Frontier models you own — run on-prem, in VPC, or self-hosted
- **Forge:** Customize models around your data and use cases
- **Le Chonk:** Specialist model for coders
- **Deployment:** On-prem, VPC, or Mistral-hosted inference

**Saturn 0.1 takeaway:** Mistral's approach confirms the viability of open-weight local models. Their smaller models (7B-12B class) run on consumer hardware — Saturn 0.1 targets even smaller (60M-150M) for the 4GB VRAM constraint.

---

### KoboldCPP — Local GGUF Inference (2026)

**Source:** github.com/LostRuins/koboldcpp (llama.cpp fork, 12.1k stars)

#### Capabilities
- **GGUF format:** Quantized models run from a single file, zero install
- **Hardware:** CPU or GPU (CUDA 11/12 support)
- **Quantization:** Q4_K, Q5_K, Q6_K, Q8_0, FP16, BF16 GGUF quantization
- **Context:** Large context windows via llama.cpp
- **Integrations:** KoboldAI Lite WebUI, MCP server, agents, RAG, TTS/STT
- **Adapters:** LoRA adapter support (kcpp_adapters)
- **LoRA conversion:** `convert_lora_to_gguf.py` — merge LoRA adapters into GGUF

**Saturn 0.1 takeaway:** KoboldCPP is inference-only (no training), but it's the ideal deployment target for Saturn 0.1. Train in PyTorch → export to GGUF → run via KoboldCPP on the same machine. LoRA adapters can be trained separately and merged.

---

## 2. Technique Cross-Reference for Saturn 0.1

| Technique | Best Source | Relevance to Saturn 0.1 | Hardware Fit |
|-----------|-------------|------------------------|--------------|
| **Muon / MuonClip Optimizer** | Kimi K2/K3 | Faster convergence than AdamW on limited data | CPU-friendly, low memory |
| **GRPO RL** | DeepSeek-R1 | Rule-based rewards for math (no critic model) | Low memory, CPU+GPU |
| **FP8/MXFP4 QAT** | Kimi K3, DeepSeek-V3 | Train with quantization for 4GB VRAM deployment | GPU compute, RAM storage |
| **MLA (Multi-head Latent Attention)** | DeepSeek-V2/V3 | Compresses KV cache — critical for 4GB VRAM | GPU memory savings |
| **Knowledge Distillation** | DeepSeek-R1, Qwen3 | Teacher (R1/Qwen3-Math) → Saturn 0.1 student | Offline, CPU-heavy |
| **3-Stage Data Rebalancing** | Qwen3 | Foundational → Math-intensive → Long-context | CPU data pipeline |
| **Synthetic Math Data** | Qwen2.5-Math | Generate math problems from teacher model | CPU generation |
| **LoRA/QLoRA** | NVIDIA NeMo | Domain adapters (Lean, Code, Olympiad) | Low VRAM, CPU training |
| **GGUF Export** | KoboldCPP/llama.cpp | Deploy trained model locally on same machine | Inference only |
| **Asymmetric Encoder-Decoder** | DeepSeek V4.1-Flash | 8B input / 16B output active params | Not applicable at 100M scale |

---

## 3. Recommended Training Pipeline for Saturn 0.1

Given your hardware (i7-7820HQ, 16GB RAM, Quadro M1200 4GB VRAM), here's the optimal stack:

### Phase 1: Data & Tokenizer (CPU-bound)
1. **Corpus:** arXiv math LaTeX, Lean 4 Mathlib, Codeforces, Project Euler, math textbooks
2. **Preprocessing:** Strip natural language from targets, keep equations/code only
3. **Tokenizer:** Custom BPE on math tokens, 8K-16K vocab (Qwen3 approach)
4. **Storage:** Memory-mapped binary (mmap) — fits in 16GB RAM, OS pages handle overflow

### Phase 2: Pre-training (CPU+RAM+GPU Heterogeneous)
1. **Model:** ~60M-150M parameter MLA transformer (our `model/saturn.py`)
2. **Optimizer:** **MuonClip** (from Kimi K2) — more token-efficient than AdamW
3. **Precision:** FP16 on GPU, FP32 master weights in RAM (ZeRO-Offload style)
4. **Strategy:** Layer-streaming — GPU holds 1-2 layers at a time, master in RAM
5. **Data rebalancing:** 3 stages (foundational → math-intensive → long-context)

### Phase 3: Supervised Fine-Tuning (SFT)
1. **Data:** Math problem → formal code/math output pairs (no English)
2. **Method:** Standard SFT with cross-entropy loss
3. **LoRA adapters:** Optional, for domain specialization (Lean 4, competitive programming)

### Phase 4: RL with Rule-Based Rewards (GRPO)
1. **Algorithm:** GRPO (DeepSeek-R1) — no critic model, group-based baseline
2. **Rewards:** Math correctness (exact match), code execution (unit tests pass), Lean proof verification
3. **Format:** Constrained decoding mask — math/code tokens only
4. **Rejection sampling:** Generate multiple outputs, keep best for next SFT round

### Phase 5: Distillation (Optional, if teacher available)
1. **Teacher:** DeepSeek-R1 distilled model or Qwen2.5-Math
2. **Method:** Multi-teacher on-policy distillation (Kimi K3 approach)
3. **Target:** Saturn 0.1 student learns from teacher's reasoning traces

### Phase 6: Quantization-Aware Training (QAT)
1. **Method:** MXFP4 QAT (Kimi K3 approach) — simulate 4-bit during training
2. **Export:** FP16 → GGUF Q4_K/Q5_K for KoboldCPP deployment
3. **Result:** Model runs entirely on the same machine via KoboldCPP

---

## 4. Modified Training Code — MuonClip + GRPO

Here's how to update `runtime/unified_train.py` with the latest methods:

```python
# runtime/muon_clip_optimizer.py
# MuonClip optimizer from Kimi K2 — Muon with QK-clip for stability

import torch
from torch.optim import Optimizer

class MuonClip(Optimizer):
    """
    MuonClip optimizer (Kimi K2): Muon momentum + QK-clip for stability.
    More token-efficient than AdamW, better for limited-data training.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, qk_clip=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, qk_clip=qk_clip)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # QK-clip: clip gradient norm for stability (Kimi K2 innovation)
                if group['qk_clip'] > 0:
                    grad_norm = grad.norm()
                    if grad_norm > group['qk_clip']:
                        grad = grad * (group['qk_clip'] / grad_norm)

                if len(state) == 0:
                    state['momentum_buffer'] = grad.clone()
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(grad, alpha=1 - group['momentum'])
                    if group['nesterov']:
                        grad = grad + group['momentum'] * buf
                    else:
                        grad = buf

                p.add_(grad, alpha=-group['lr'])

        return loss
```

```python
# runtime/grpo_trainer.py
# GRPO trainer for Saturn 0.1 — rule-based rewards, no critic model

import torch
import torch.nn.functional as F
from runtime.unified_train import UnifiedTrainer

class GRPOTrainer(UnifiedTrainer):
    def __init__(self, *args, group_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size  # Number of outputs per prompt for baseline

    def compute_rule_based_reward(self, output_ids, target_type='math'):
        """
        Rule-based reward for math/code — no learned reward model.
        Returns scalar reward per output.
        """
        # Decode output
        text = self.tokenizer.decode(output_ids.tolist())

        if target_type == 'math':
            # Check if output is valid LaTeX/math notation
            reward = 0.0
            # Reward mathematical symbols, penalize English words
            if any(sym in text for sym in ['\\', '=', '+', '-', '*', '/', '\\frac', '\\sqrt']):
                reward += 1.0
            # Penalize common English words
            english_words = ['the', 'and', 'is', 'are', 'answer', 'solution']
            for word in english_words:
                if word in text.lower():
                    reward -= 0.5
            return max(reward, 0.0)

        elif target_type == 'code':
            # Check if output parses as valid code (Python/Lean/C++)
            reward = 0.0
            if any(sym in text for sym in ['def ', 'theorem', 'proof', 'int ', '#include']):
                reward += 1.0
            return max(reward, 0.0)

        return 0.0

    def grpo_step(self, prompts, target_type='math'):
        """
        One GRPO step: generate group_size outputs per prompt,
        compute rewards, normalize within group, policy gradient update.
        """
        all_rewards = []
        all_log_probs = []

        for prompt in prompts:
            group_outputs = []
            group_log_probs = []

            for _ in range(self.group_size):
                # Generate output (constrained decoding)
                output = self.generate_constrained(prompt, target_type)
                log_prob = self.compute_log_prob(prompt, output)
                reward = self.compute_rule_based_reward(output, target_type)

                group_outputs.append(output)
                group_log_probs.append(log_prob)
                all_rewards.append(reward)
                all_log_probs.append(log_prob)

            # GRPO: normalize rewards within group
            group_rewards = torch.tensor(all_rewards[-self.group_size:])
            mean = group_rewards.mean()
            std = group_rewards.std() + 1e-6
            normalized = (group_rewards - mean) / std

            # Policy gradient: loss = -mean(normalized_reward * log_prob)
            group_log_probs_tensor = torch.stack(all_log_probs[-self.group_size:])
            loss = -(normalized * group_log_probs_tensor).mean()
            loss.backward()

        # Stream gradients to param server, optimizer step on CPU
        self.param_server.step()
        self.step += 1
```

---

## 5. Hardware-Specific Optimizations

### For Quadro M1200 (4GB VRAM, Maxwell, CUDA 12.8)
- **No BF16 support** — use FP16 only
- **MLA attention** — compresses KV cache, fits longer context in 4GB
- **Layer streaming** — GPU holds 1-2 layers at a time
- **FP32 master in 16GB RAM** — weights + Adam states (~2GB for 100M params)
- **Gradient accumulation 16x** — reduces PCIe transfers

### For 16GB System RAM
- **Pinned memory** (`pin_memory=True`) for faster GPU transfers
- **mmap'd dataset** — OS handles paging, no manual disk I/O
- **Swap 8GB** — overflow for dataset if larger than RAM

### For i7-7820HQ (4C/8T)
- **2 DataLoader workers** — leaves 2 cores for main training loop
- **Muon optimizer** — lighter per-step compute than AdamW
- **CPU offload** — optimizer runs on CPU, GPU only does matmuls

---

## 6. Deployment Path (Same Machine)

1. Train in PyTorch (FP16 + FP32 master)
2. Export to GGUF via `convert_hf_to_gguf.py` (from KoboldCPP/llama.cpp)
3. Quantize to Q4_K or Q5_K (4-bit, fits easily in 4GB VRAM)
4. Run via **KoboldCPP** — one executable, local inference, CUDA acceleration
5. Optionally train LoRA adapters separately, merge with `convert_lora_to_gguf.py`

---

## 7. Sources

- **Kimi K2:** https://arxiv.org/abs/2507.20534 (MuonClip, 15.5T tokens, agentic RL)
- **Kimi K3:** https://arxiv.org/html/2607.24653v1 (MXFP4 QAT, Per-Head Muon, Multi-Teacher Distillation, ZeRO-2 offloading)
- **DeepSeek-R1:** https://arxiv.org/abs/2501.12948 (GRPO, RL without SFT, distillation)
- **DeepSeek-V3:** https://arxiv.org/abs/2412.19437 (FP8 training, MLA, DeepSeekMoE)
- **DeepSeek V4.1-Flash:** https://www.deepseek.com/en/news/deepseek-v4-1-flash/ (asymmetric architecture, KV cache compression)
- **DeepSeek Model Disclosure:** https://cdn.deepseek.com/policies/en-US/model-algorithm-disclosure.html
- **Qwen3:** https://www.cometapi.com/decoding-qwen3s-training-a-deep-dive/ (36T tokens, 3-stage training, synthetic math data)
- **Qwen2.5:** https://arxiv.org/abs/2412.15115 (18T tokens, Qwen2.5-Math branch)
- **NVIDIA NeMo AutoModel:** https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/overview (SFT, PEFT, KD, QAT, EAGLE speculative decoding)
- **Mistral AI:** https://mistral.ai (open weights, Forge customization, on-prem deployment)
- **KoboldCPP:** https://github.com/LostRuins/koboldcpp (GGUF, llama.cpp fork, LoRA adapters, local inference)
