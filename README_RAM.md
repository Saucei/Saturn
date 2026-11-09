# Saturn 0.1 — RAM-Primary Development (Windows 10)

## Hardware Reality
- **CPU:** Intel i7-7820HQ (4C/8T, 2.9 GHz)
- **System RAM:** 16 GB (PRIMARY constraint for training)
- **GPU:** NVIDIA Quadro M1200, 4 GB VRAM (SECONDARY; optional compute offload)
- **OS:** Windows 10 Pro

## Design Principle
Model weights, optimizer states, and gradients live in **system RAM**. The GPU is secondary — used only to accelerate individual matmul operations if drivers are available. The model can train entirely on CPU.

## Files
| File | Purpose |
|------|---------|
| `SYSTEM_SPECS.md` | Sanitized hardware specifications |
| `MODEL_SELECTION_RAM.md` | Model options with pros/cons for 16 GB RAM |
| `configs/ram-training.yaml` | RAM-optimized training configuration |
| `runtime/ram_train.py` | CPU-primary training loop (Windows compatible) |
| `model/saturn.py` | MLA transformer (from-scratch architecture) |
| `data/pipeline.py` | Memory-mapped math tokenizer + dataset |
| `math_only/decode_mask.py` | Constrained decoding (math/code output only) |

## Next Step
Read `MODEL_SELECTION_RAM.md` and choose one model (A–G). Then I will generate the exact training script, data pipeline, and configuration for your selected model.

## Quick Start (after model selection)
```powershell
cd "C:\Users\Berry\Desktop\Saturn"
python -m venv saturn-env
saturn-env\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyyaml numpy tqdm transformers peft
python runtime/ram_train.py --config configs/ram-training.yaml --strategy from_scratch
```

Note: `--index-url .../cpu` installs CPU-only PyTorch (no CUDA needed). Add `--gpu-offload` later if you install CUDA drivers.
