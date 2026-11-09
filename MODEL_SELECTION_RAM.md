# Model Selection — RAM Primary (16 GB System RAM, 4 GB VRAM Secondary)

**Goal:** Choose a base model or architecture whose weights, optimizer states, and gradients fit primarily in 16 GB system RAM. VRAM is secondary — used only for optional compute offload.

You must manually select one option below. Reply with your choice (e.g., "Option C" or "Custom from-scratch 300M").

---

## RAM Training Math

For **full fine-tuning** in FP16 with FP32 Adam optimizer:
- Weights (FP16): 2 bytes/param
- Adam states (FP32 × 2): 8 bytes/param
- Gradients (FP16): 2 bytes/param
- **Total: ~12 bytes per parameter**
- 12 GB usable RAM ÷ 12 bytes ≈ **1.3 billion parameters max** for full fine-tuning

For **QLoRA / LoRA fine-tuning**:
- Base model stored in RAM (quantized INT4 or FP16)
- Only small adapter matrices trained
- 3B–7B base models fit comfortably in RAM

For **from-scratch training**:
- 100M–500M parameters is comfortable; up to ~1B possible

---

## Tier 1 — Tiny Models (Comfortable Full Fine-Tune in RAM)

### Option A: TinyLlama 1.1B
**Repo:** `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`

| Spec | Value |
|------|-------|
| Parameters | 1.1B |
| FP16 size | ~2.2 GB |
| RAM needed for full fine-tune | ~13 GB (tight, fits in 16 GB) |
| Architecture | Llama-style dense |

**Pros:**
- Extremely small; full fine-tuning fits in 16 GB RAM with room for dataset cache
- Fastest training on your i7-7820HQ CPU
- Open MIT license
- Good for learning the pipeline before scaling up

**Cons:**
- Older architecture; not optimized for math/reasoning
- Limited general knowledge
- 1.1B may struggle with complex mathematical proofs

**Best for:** Quick prototyping, proving the math-only output constraint works.

---

### Option B: Qwen2.5-0.5B Instruct
**Repo:** `Qwen/Qwen2.5-0.5B-Instruct`

| Spec | Value |
|------|-------|
| Parameters | 0.5B |
| FP16 size | ~1.0 GB |
| RAM needed for full fine-tune | ~6 GB (very comfortable) |
| Architecture | Qwen dense |

**Pros:**
- Tiny memory footprint; leaves 8+ GB RAM for data and activations
- Modern architecture with strong reasoning for its size
- Apache 2.0 license
- Excellent tokenizer for code/math symbols
- Fastest inference and training

**Cons:**
- 0.5B may be too small for complex mathematical reasoning
- Less community fine-tuning documentation than larger models

**Best for:** Maximum headroom in RAM; if you want to experiment with large datasets and batch sizes.

---

### Option C: Qwen2.5-1.5B Instruct (Recommended)
**Repo:** `Qwen/Qwen2.5-1.5B-Instruct`

| Spec | Value |
|------|-------|
| Parameters | 1.5B |
| FP16 size | ~3.0 GB |
| RAM needed for full fine-tune | ~18 GB (slightly over; use gradient checkpointing or CPU offload) |
| Architecture | Qwen dense |

**Pros:**
- Strong math and reasoning capability relative to size
- Modern Qwen architecture, excellent code/math tokenization
- Apache 2.0 license
- Good balance of capability and RAM footprint
- Can full fine-tune on 16 GB RAM with gradient accumulation + gradient checkpointing

**Cons:**
- Tighter fit than 0.5B/1.1B; requires careful memory management (gradient checkpointing, smaller batch size)
- Slightly slower training than sub-1B models

**Best for:** **Recommended starting point** — best capability-to-RAM ratio for a math/code model.

---

### Option D: SmolLM2 1.7B Instruct
**Repo:** `HuggingFaceTB/SmolLM2-1.7B-Instruct`

| Spec | Value |
|------|-------|
| Parameters | 1.7B |
| FP16 size | ~3.4 GB |
| RAM needed for full fine-tune | ~20 GB (requires QLoRA or CPU offload for full fine-tune) |
| Architecture | SmolLM dense |

**Pros:**
- Designed specifically for edge/local deployment
- Strong reasoning benchmarks relative to size
- Apache 2.0 license
- Good instruction-following

**Cons:**
- At the upper limit for 16 GB RAM full fine-tuning; likely needs QLoRA
- Less math-specialized than Qwen variants

**Best for:** Edge deployment priority; if you plan to quantize heavily after training.

---

## Tier 2 — Medium Models (QLoRA Only; Base in RAM)

### Option E: Phi-3.5-mini 3.8B Instruct (QLoRA)
**Repo:** `microsoft/Phi-3.5-mini-instruct`

| Spec | Value |
|------|-------|
| Parameters | 3.8B |
| INT4 size | ~2.0 GB |
| FP16 size | ~7.6 GB |
| RAM needed (QLoRA) | ~8–10 GB (base INT4 + adapters + optimizer) |
| Architecture | Phi dense |

**Pros:**
- Exceptional reasoning and math capability
- Microsoft research quality
- 4K context window
- Fits easily in 16 GB RAM with QLoRA

**Cons:**
- Cannot full fine-tune on 16 GB RAM (would need ~45 GB for FP32 Adam)
- Requires QLoRA/PEFT setup (more complex)
- Still uses CPU primarily; GPU optional

**Best for:** If you want maximum math capability and are willing to use QLoRA.

---

### Option F: Qwen2.5-3B (QLoRA)
**Repo:** `Qwen/Qwen2.5-3B-Instruct`

| Spec | Value |
|------|-------|
| Parameters | 3B |
| INT4 size | ~1.8 GB |
| FP16 size | ~6 GB |
| RAM needed (QLoRA) | ~7–9 GB |
| Architecture | Qwen dense |

**Pros:**
- Stronger than 1.5B for math/code
- Still fits comfortably in 16 GB RAM with QLoRA
- Apache 2.0 license

**Cons:**
- QLoRA required; full fine-tune exceeds RAM
- Slightly more complex setup

**Best for:** Mid-tier capability with comfortable RAM headroom.

---

## Tier 3 — From-Scratch Custom Model

### Option G: Custom From-Scratch MLA Transformer (60M–500M parameters)
**Architecture:** Multi-head Latent Attention (MLA) + SwiGLU + Grouped Query Attention

| Spec | Value |
|------|-------|
| Parameters | 60M–500M (your choice) |
| FP16 size | 0.12 GB – 1.0 GB |
| RAM needed for full fine-tune | 0.7 GB – 6 GB |
| Architecture | Custom MLA decoder |

**Pros:**
- Complete control over architecture, vocabulary, and output constraints
- Designed specifically for math/code-only output (no natural language)
- Fits easily in 16 GB RAM; can train larger batches
- No license restrictions
- Can use all 16 GB RAM for dataset cache and activations

**Cons:**
- Requires curating your own math dataset from scratch
- No pretrained knowledge to bootstrap from
- Training from scratch takes longer to reach capability

**Best for:** If you want a purpose-built math-only model and have time to train from scratch.

---

## Architecture Recommendations for RAM-Primary Training

| Feature | Recommendation | Reason |
|---------|---------------|--------|
| Precision | FP16 weights, FP32 optimizer | Maxwell GPU doesn't support BF16; FP16 halves RAM vs FP32 |
| Optimizer | AdamW 8-bit (bitsandbytes) or standard AdamW | 8-bit optimizer cuts optimizer state RAM by ~50% |
| Gradient Checkpointing | Enabled | Trades compute for RAM; essential for 1.5B+ models |
| CPU Offload | ZeRO-Inference style | Model lives in RAM; GPU only for matmul batches |
| Gradient Accumulation | 8–16 steps | Compensates for small batch size due to RAM limits |
| Data Storage | Memory-mapped (mmap) binary | Dataset larger than RAM; OS pages handle overflow |
| Vocabulary | 8K–16K math/code tokens | Smaller vocab = smaller embedding table = less RAM |

---

## Your Selection

Reply with one of:
- **A** — TinyLlama 1.1B
- **B** — Qwen2.5-0.5B Instruct
- **C** — Qwen2.5-1.5B Instruct (recommended)
- **D** — SmolLM2 1.7B Instruct
- **E** — Phi-3.5-mini 3.8B (QLoRA)
- **F** — Qwen2.5-3B (QLoRA)
- **G** — Custom from-scratch MLA (60M–500M)

Once you select, I will generate the exact training code, configuration, and data pipeline for that model, optimized to run primarily in your 16 GB system RAM on Windows 10.
