# Saturn 0.1 — Math/Code-Only Language Model

> **Status: 🚧 IN ACTIVE DEVELOPMENT** — Research, architecture, and training pipeline are being actively built. Not yet ready for production use.

---

## Vision

**Saturn 0.1** is a specialized language model trained **exclusively on mathematical and code data**. It accepts natural language or mathematical text as input and outputs **only formal mathematics (LaTeX, Lean 4) or executable code (Python, C++, Julia)** — never natural language prose.

**Core philosophy:** A model that *does* mathematics rather than *talks about* mathematics.

---

## Current Status

| Component | Status |
|-----------|--------|
| Research & literature review | ✅ Complete |
| Hardware analysis & budget planning | ✅ Complete |
| Model architecture selection | ✅ Qwen2.5-Math-1.5B-Instruct (QLoRA/full 8-bit fine-tune) |
| Training pipeline design | ✅ RAM-primary architecture designed |
| Math-only constrained decoding | ✅ Implemented |
| Dataset curation pipeline | ✅ Pipeline code ready; corpus collection in progress |
| Training code (CPU/GPU) | ✅ `ram_train.py`, `qwen_math_train.py`, `vastai_qlora_train.py` |
| Vast.ai QLoRA test run | 🔄 Ready to execute ($2.17 budget) |
| Full fine-tune on 16GB RAM | ⏳ Pending 8-bit Adam validation |
| Ubuntu Server migration | ⏳ Planned (next stage) |

---

## Model Architecture

**Base Model:** Qwen2.5-Math-1.5B-Instruct (Apache 2.0)  
**Parameters:** ~1.53B  
**Specialization:** Mathematical reasoning, theorem proving, code generation  
**Context Window:** 32K tokens (16K used for training)  

**Fine-tuning Strategy:**
- **Primary:** Full fine-tune with **8-bit Adam optimizer** (bitsandbytes) — fits in 16 GB RAM (~13 GB peak)
- **Fallback:** QLoRA (4-bit base + LoRA adapters) — trains in <8 GB RAM
- **From-scratch option:** Custom MLA transformer (60M–150M params) in `model/saturn.py`

**Output Constraint:** Math/code tokens only (LaTeX, Lean 4, Python, C++, Julia). Natural language tokens masked at inference via `math_only/qwen_decode_mask.py`.

---

## Hardware Targets

| Tier | System RAM | GPU VRAM | CPU | Notes |
|------|------------|----------|-----|-------|
| **Dev (current)** | 16 GB | 4 GB (Quadro M1200, Maxwell CC 5.0) | i7-7820HQ 4C/8T | Dev machine; CPU-primary training |
| **Recommended Prod** | 32–64 GB | 12–24 GB (RTX 3060/3090/4090) | Modern 8C+ | On-prem server; full fine-tune comfortable |
| **Minimal Inference** | 8 GB | 4 GB | Any x64 | GGUF Q4_K quantized model |

**Key constraint:** Model lives in **system RAM** (16 GB). GPU is secondary — used only for matmul offload if available.

---

## Training Methodology

### Data Strategy (3-Stage Curriculum)
1. **Foundational** — arXiv math LaTeX, textbooks, Wikipedia math (formal only)
2. **Reasoning-intensive** — Lean 4 Mathlib proofs, competition math (AIME, Olympiad), symbolic derivations
3. **Code/Math synthesis** — Codeforces solutions, Project Euler, symbolic math → code translation

### Training Pipeline
1. **Data preparation** → `data/qwen_math_pipeline.py` (memory-mapped, mmap)
2. **Tokenization** → Qwen2.5-Math tokenizer, math-only preprocessing
3. **Training** → `runtime/qwen_math_train.py` (QLoRA) or `runtime/ram_train.py` (from-scratch)
3. **Constrained decoding** → `math_only/qwen_decode_mask.py` (math-only logits mask)
4. **Export** → GGUF Q4_K via llama.cpp for edge deployment

### RAM-Primary Design
- Model weights + optimizer states + gradients live in **system RAM**
- GPU (if present) used only for matmul offload via `use_gpu_offload: true`
- Memory-mapped dataset allows corpora larger than RAM
- Gradient accumulation (16 steps) compensates for small batch size

---

## Project Structure

```
Saturn/
├── configs/
│   ├── qwen2.5-math-1.5b.yaml    # Qwen2.5-Math-1.5B training config
│   ├── ram-training.yaml          # RAM-primary generic config
│   └── saturn-0.1-small.yaml      # From-scratch small model config
├── data/
│   ├── pipeline.py                # Generic math tokenizer + mmap dataset
│   └── qwen_math_pipeline.py      # Qwen-specific tokenizer + dataset
├── math_only/
│   ├── decode_mask.py             # Generic math-only logits mask
│   └── qwen_decode_mask.py        # Qwen-specific math-only mask
├── model/
│   └── saturn.py                  # From-scratch MLA transformer
├── os/
│   ├── linux-optimize.sh          # Kernel/sysctl tuning for training
│   └── storage_partition.md       # Disk partitioning for Linux install
├── runtime/
│   ├── qwen_math_train.py         # Qwen2.5-Math QLoRA/full fine-tune
│   ├── vastai_qlora_train.py      # Vast.ai QLoRA burst training
│   ├── ram_train.py               # RAM-primary from-scratch/CPU trainer
│   ├── unified_train.py           # Original unified CPU+RAM+GPU loop
│   ├── param_server.py            # FP32 master weights in RAM
│   └── gpu_window.py              # GPU compute window
├── os/
│   ├── linux-optimize.sh
│   └── storage_partition.md
├── checkpoints/                   # Model checkpoints (runtime)
├── data/tokenized/                # Memory-mapped tokenized corpus (runtime)
├── AI Compression Research.md     # Quantization, pruning, KD, LoRA survey
├── Ai development.md              # DeepSeek, GLM, Kimi research catalogue
├── MODEL_SELECTION_RAM.md         # Model options for 16 GB RAM
├── RAM_BUDGET_QWEN_1.5B.md        # Exact RAM calculations for Qwen 1.5B
├── TRAINING_METHODS.md            # SOTA training methods (Nvidia, Qwen, Kimi, DeepSeek)
├── SYSTEM_SPECS.md                # Sanitized hardware specs
└── SETUP_WINDOWS_QWEN.md          # Windows 10 setup guide
```

---

## Development Roadmap

### Phase 1: Foundation (Current)
- [x] Research complete
- [x] Qwen2.5-Math-1.5B QLoRA test on Vast.ai ($2.17 budget)
- [ ] Validate full fine-tune with 8-bit Adam on 16 GB RAM
- [ ] Curate 50K–100K high-quality math samples
- [ ] First successful training run with math-only output

### Phase 2: Linux Migration & Scaling
- [ ] Install Ubuntu Server 24.04 LTS (headless)
- [ ] Migrate training pipeline to Linux
- [ ] Full fine-tune Qwen2.5-Math-1.5B on 16 GB RAM with 8-bit Adam
- [ ] Export GGUF Q4_K for KoboldCPP/Ollama inference
- [ ] Benchmark on math evals (MATH, GSM8K, Lean 4 proofs)

### Phase 3: Specialization & Productization
- [ ] Train domain variants: Saturn-Finance, Saturn-Engineering, Saturn-Formal
- [ ] Build FastAPI inference server with math-only constraints
- [ ] On-prem deployment packaging (Docker + systemd)
- [ ] Sovereign AI deployment guide for enterprise customers

### Phase 4: From-Scratch Model
- [ ] Train custom MLA transformer (100M–300M params) from scratch on pure math corpus
- [ ] Compare against fine-tuned Qwen baseline
- [ ] Publish technical report

---

## Quick Start (Windows 10 — Current Dev Environment)

```powershell
cd "C:\Users\Berry\Desktop\Saturn"
python -m venv saturn-env
saturn-env\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml numpy tqdm transformers accelerate peft bitsandbytes

# Build dataset (place raw .txt/.tex files in data/corpus/raw/ first)
python -c "
from data.qwen_math_pipeline import QwenMathTokenizer, QwenMathCorpusDataset
import glob
texts = []
for f in glob.glob('data/corpus/raw/*'):
    with open(f, 'r', encoding='utf-8') as fh: texts.append(fh.read())
tok = QwenMathTokenizer()
ds = QwenMathCorpusDataset(tok, 'data/tokenized/qwen_math_corpus.bin', seq_len=2048, build_vocab=True, raw_texts=texts)
print('Built:', len(ds), 'sequences')
"

# Train QLoRA on Vast.ai (or locally with 8-bit Adam)
python runtime/vastai_qlora_train.py --data data/tokenized/qwen_math_corpus.bin --output adapters/
```

---

## Linux Quick Start (Next Stage)

```bash
# After Ubuntu Server 24.04 LTS install + NVIDIA driver + CUDA 11.8
git clone https://github.com/Saucei/Saturn.git
cd Saturn
bash os/linux-optimize.sh
python -m venv saturn-env && source saturn-env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml numpy tqdm transformers accelerate peft bitsandbytes

# Train full fine-tune with 8-bit Adam
python runtime/ram_train.py --config configs/qwen2.5-math-1.5b.yaml --gpu-offload
```

---

## Contributing

This is a private research project (Saucei/Saturn). Issues and PRs are welcome for:
- Math corpus curation (public domain / CC-BY-SA sources)
- Training pipeline optimizations
- Additional math-specialized model configs
- Evaluation benchmarks (MATH, GSM8K, Lean 4, HumanEval)

---

## License

- **Base model:** Qwen2.5-Math (Apache 2.0)
- **Training code & configs:** MIT (this repository)
- **Datasets:** Respect individual source licenses (arXiv, Lean Mathlib, etc.)

---

## Citation

If you use Saturn 0.1 in research, please cite:
```
@misc{saturn2025,
  title={Saturn 0.1: A Math/Code-Only Language Model Trained Under RAM Constraints},
  author={Berry, Saturn Team},
  year={2025},
  url={https://github.com/Saucei/Saturn}
}
```

---

**Last Updated:** 2026-11-09  
**Current Phase:** QLoRA validation on Vast.ai → Full fine-tune on 16 GB RAM