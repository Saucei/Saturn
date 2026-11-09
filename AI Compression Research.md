# AI Model Compression & Efficient Training — Research Catalogue

Research on **Quantization, Pruning, Knowledge Distillation, and LoRA/Factorization**
from Qwen, DeepSeek, GLM (Zhipu AI), Flint (Recursive), and the broader literature.

Compiled for the **Saturn 0.1** project — a math/code-only model.

---

## 1. Quantization

Quantization reduces the numerical precision of model weights and activations (e.g., BF16 → FP8 → INT4/INT2) to shrink memory footprint, storage, and inference cost.

### Qwen (Alibaba)
- **Source:** "An Empirical Study of Qwen3 Quantization" (arXiv:2505.02214, May 2025)
- **Scope:** Qwen3 post-training models (0.6B, 1.8B, 4B, 7B, 14B, 72B) and base variants
- **PTQ methods evaluated (5):** Round-To-Nearest (RTN), GPTQ, AWQ, SmoothQuant, BiLLM
- **Bit-widths tested:** 1 to 8 bits (weight-only); activations typically 4 or 8 bits
- **Calibration:** 128 samples from C4, seq len 2048; per-group block size 128
- **Key finding:** Qwen3 holds up at moderate bit-widths; **ultra-low precision (≤3-bit) causes notable degradation**, especially on linguistic tasks. Weight-only quantization is more stable than weight+activation at low bits.
- **HF:** `Qwen/Qwen3-*` (0.6B–235B)

### DeepSeek (deepseek-ai)
- **Source:** DeepSeek-V3 Technical Report (arXiv:2412.19437); FlashMLA docs; DeepSeek V4.1 Flash news (Sept 2026)
- **FP8 Training:** Full mixed-precision FP8 training framework (Section 3.3)
  - Fine-grained (block-wise) quantization; increased accumulation precision; mantissa-over-exponents; online quantization
  - Low-precision optimizer states, activations, and communication
  - FP8 vs BF16 ablation shows parity (Appendix B.1)
- **FlashMLA:** FP8 KV-cache memory format for decode-phase efficiency; tile-based scalable layout
- **DeepSeek-V4.1-Flash-1017:** Native FP8; 552B MoE with asymmetric encoder-decoder (8B active input / 16B active output); KV cache = 1/4 HBM, 1/8 SSD storage vs. previous gen
- **HF:** `deepseek-ai/DeepSeek-V3`, `DeepSeek-V2-Lite`, `DeepSeek-Coder-V2-Lite`, `DeepSeek-V4.1-Flash-1017` (GGUF/quantized variants available via community)

### GLM / Zhipu AI (zai-org)
- **Source:** GLM-5 technical report (arXiv:2602.15763); GLM-5.3-Flash blog (z.ai/blog/glm-5.3-flash); Semianalysis inferencex
- **GLM-5.3-Flash:** Native FP8 (e4m3, dynamic scaling); 320B total / 18B active; 1M context
- **Architecture efficiency:** DSA (Delta Sparse Attention) + IndexShare sparse-attention indexer (2.9× FLOPs reduction at 1M context); hybrid sparse + linear attention; Manifold-Constrained Hyper-Connections (mHC)
- **GLM-5:** 744B MoE (~40B active), DSA to reduce training/inference cost while keeping long-context fidelity
- **HF:** `zai-org/GLM-5.3-Flash`, `GLM-4.1V-9B-Thinking`

### Flint (Recursive / Element Labs)
- **Source:** LM Studio model card (recursivesi/flint); The Decoder / intelligentliving coverage
- **Flint:** 39.8B MoE coding agent model, ~2.5B active per token; 262K context; Apache 2.0
- **Quantization:** GGUF distribution; **NVFP4** quantization tested
  - Flint BF16 vs NVFP4: Terminal-Bench 54.9 → 53.7; SWE-bench Pro 89.7 → 88.5; minimal degradation at 4-bit
- **Built with:** Eureka Machine (automated AI research system that proposes/validates architecture/training/inference improvements)

### Cross-Lab Quantization Summary

| Lab | Flagship quant method | Precision | Key result |
|-----|----------------------|-----------|------------|
| **Qwen** | PTQ (RTN/GPTQ/AWQ/SmoothQuant/BiLLM) | 1–8 bit weight-only | Stable at moderate bits; degrade ≤3-bit |
| **DeepSeek** | Native FP8 training + FP8 KV cache | FP8 (e4m3/e5m2) | FP8 ≈ BF16 quality; 93.3% KV-cache reduction (V2) |
| **GLM** | Native FP8 + DSA/IndexShare | FP8 e4m3 dynamic | 320B/18B Flash; 1/10th price of GLM-5.2 |
| **Flint** | GGUF / NVFP4 | 4-bit NVFP4 | <1.5 pt drop on coding benchmarks vs BF16 |

---

## 2. Pruning

Pruning removes redundant weights, neurons, or structures to create sparsity and reduce compute/storage.

### Taxonomy (from literature)
- **Unstructured (fine-grained):** Zero out individual weights below a magnitude threshold. Model shape unchanged; matmuls run on original slabs → **no wallclock speedup on GPUs without sparse kernels**. Benefit shows at distribution time (gzip collapses zero runs). (posttrainllm.com; apxml.com)
- **Structured (coarse-grained):** Remove entire attention heads, MLP intermediates, layers, or experts. Shape changes → **direct latency and memory savings**. Dominates literature (~55%) due to hardware compatibility. (Springer systematic review, 2026)
- **Semi-structured:** N:M sparsity patterns (e.g., 2:4) supported by tensor cores on modern GPUs.

### Criteria & schedules
- **Criteria:** Weight magnitude, gradient × weight, activation-based importance, Hessian-aware (OBD/OBS)
- **Schedules:** One-shot pruning, gradual/iterative pruning, pruning-then-fine-tuning

### Practical note for Saturn
For a math/code-only model, **structured pruning of non-math experts/heads** post-training could yield a smaller inference model with minimal capability loss, especially if combined with distillation.

---

## 3. Knowledge Distillation

A smaller "student" model learns to imitate a larger "teacher" model rather than learning from raw data alone.

### Mechanisms
- **Logit / soft-target distillation:** Student matches teacher's output probability distribution (KL-divergence loss), usually with temperature scaling
- **Feature-based distillation:** Student matches intermediate layer representations
- **Sequence-level / CoT distillation:** Student learns teacher's chain-of-thought reasoning traces (critical for math/reasoning models)
- **Data augmentation (DA) paradigm:** Small seed of data → teacher generates more domain-specific data → student trains on it

### DeepSeek R1 → V3 distillation (key precedent)
- **Source:** DeepSeek-V3 Technical Report, Section 5.4.1 "Distillation from DeepSeek-R1"
- DeepSeek-V3's post-training includes **knowledge distillation from DeepSeek-R1** (long-CoT reasoning model) into standard LLM format
- Transfers reasoning/verification/reflection patterns from R1 into V3 while controlling output style and length
- Produces the final chat-optimized model after SFT + RL

### Saturn 0.1 implication
Distillation is **central** to Saturn 0.1's design: a math-specialist teacher (e.g., DeepSeek-R1 math traces, or GLM-5.3-Flash math outputs) distills into a compact math/code-only student.

---

## 4. LoRA / Low-Rank Adaptation / Factorization

LoRA (Low-Rank Adaptation, Microsoft, 2021) is a parameter-efficient fine-tuning technique.

### How it works
- Freezes pre-trained weights; injects trainable low-rank matrices A (d×r) and B (r×k) into attention/MLP layers
- Update: W' = W + BA, where rank r ≪ min(d,k)
- Reduces trainable parameters by **100–1000×** vs. full fine-tuning
- Adapters can be swapped per task without reloading the base model

### QLoRA (Quantized LoRA)
- 4-bit quantizes the base model (NF4) + LoRA adapters in BF16
- Enables fine-tuning billion-parameter models on modest hardware (single GPU)
- Double quantization, paged optimizers for memory management

### Saturn 0.1 implication
- LoRA/QLoRA lets Saturn 0.1 be adapted to new math domains (e.g., Lean proofs, competitive programming) without full retraining
- Factorization (low-rank weight decomposition) can also compress the base model post-training

---

## 5. Saturn 0.1 — Design Specification

**Goal:** A model trained **entirely on mathematical data**. Input: text (natural-language math problems, theorem statements, code-like math). Output: **code/math only — no written language.**

### Core design constraints
| Constraint | Specification |
|------------|---------------|
| Input domain | Mathematical text (word problems, theorems, proofs, equations, code) |
| Output domain | Formal code / mathematical notation only (no natural language prose) |
| Training data | Purely mathematical — no general web text, no chat, no general QA |
| Model type | Causal decoder-only Transformer (or MoE) |

### Recommended architecture (start small, scale up)

**Phase 1 — Saturn 0.1-Small (prototype):**
- Base: 1B–3B parameter dense decoder (Qwen2.5-Math-1.5B or DeepSeek-Math-7B as reference architecture)
- Vocabulary: Mathematical tokens + code tokens (LaTeX, Lean, Python, C++, Julia) + minimal punctuation; **strip natural-language tokens** where possible
- Context: 32K–128K tokens
- Attention: MLA (Multi-head Latent Attention, DeepSeek style) for efficient long-context math reasoning
- FFN: SwiGLU or GELU

**Phase 2 — Saturn 0.1-MoE (production):**
- MoE with routed experts specialized by math domain (algebra, calculus, number theory, geometry, formal verification, competitive programming)
- Activated params: 2–5B; total: 20–100B
- Follow DeepSeek-V2/V3 MoE + auxiliary-loss-free load balancing

### Training pipeline (compression-technique stack)

1. **Pre-training** on pure mathematical corpus (arXiv math, proofs, competition problems, GitHub code repos filtered to math/algo)
2. **Supervised Fine-Tuning (SFT)** on math→code/math pairs (input problem → output formal proof/code, no English explanations)
3. **Knowledge Distillation:** Teacher (DeepSeek-R1 or GLM-5.3-Flash math traces) generates chain-of-thought reasoning + formal code outputs; Saturn 0.1 student learns from soft targets + CoT traces
4. **Reinforcement Learning (RL):** Rule-based reward (correct math answer, code passes unit tests, Lean proof verifies) via GRPO (DeepSeek-R1 style)
5. **Pruning:** Structured pruning of redundant heads/experts post-training
6. **Quantization:** FP8 training (DeepSeek style) → INT4/NVFP4 inference (Flint/llama.cpp GGUF style)
7. **LoRA adapters:** Domain-specific adapters (Lean 4, Olympiad math, competitive programming) for task adaptation without full retraining

### Output filter (no written language)
- **Constrained decoding:** Restrict vocabulary at inference to math/code tokens only
- **Format enforcement:** Output must parse as valid LaTeX, Lean, Python, or C++; reject natural-language tokens
- **Reward:** Penalize any non-math-token output during RL

### Suggested starting points (open weights to distill from / adapt)
- **Teacher:** DeepSeek-R1 (reasoning traces), DeepSeek-Math, Qwen2.5-Math, GLM-5.3-Flash (math)
- **Base architecture:** Qwen2.5-Math, DeepSeek-V2-Lite (efficient MoE reference), Flint (coding agent reference)
- **Quantization reference:** DeepSeek FP8 training, Qwen3 PTQ study, Flint NVFP4 GGUF
- **Distillation reference:** DeepSeek-V3 §5.4.1 (R1→V3), arXiv:2402.13116 (KD survey)
- **LoRA reference:** Microsoft LoRA (2021), QLoRA (4-bit NF4)

### Files to create next
- `Saturn/data/` — mathematical corpus curation pipeline
- `Saturn/configs/saturn-0.1-small.yaml` — model config
- `Saturn/train/` — pre-train + SFT + RL scripts
- `Saturn/eval/` — math correctness + code execution + Lean verification benchmarks
- `Saturn/decode/` — constrained math-only decoding layer

---

## Sources

- Qwen3 Quantization: https://arxiv.org/abs/2505.02214
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeek-V4.1-Flash news: https://www.deepseek.com/en/news/deepseek-v4-1-flash/
- DeepSeek V4.1-Flash-1017 (HF): https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- FlashMLA FP8 KV cache: https://github.com/deepseek-ai/FlashMLA
- GLM-5 technical report: https://arxiv.org/abs/2602.15763
- GLM-5.3-Flash blog: https://z.ai/blog/glm-5.3-flash
- GLM-5.3-Flash (HF): https://huggingface.co/zai-org/GLM-5.3-Flash
- Flint model card: https://lmstudio.ai/models/recursive-flint
- Pruning survey: https://link.springer.com/article/10.1007/s10115-026-02721-5
- Knowledge Distillation survey (LLM): https://arxiv.org/abs/2402.13116
- LoRA (Microsoft, 2021): https://en.wikipedia.org/wiki/LoRA_(machine_learning)
- QLoRA: https://www.geeksforgeeks.org/deep-learning/qlora-quantized-low-rank-adapter/
- Model Compression survey: https://arxiv.org/abs/2308.07633
