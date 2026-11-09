# Saturn 0.1 — Project Structure

Saturn/
├── os/
│   ├── linux-optimize.sh          # Kernel/sysctl tuning for training
│   └── storage_partition.md       # Disk partitioning advice
├── configs/
│   └── saturn-0.1-small.yaml      # Model + training + hardware config
├── model/
│   └── saturn.py                  # MLA transformer (math/code only)
├── runtime/
│   ├── unified_train.py           # Main training loop (CPU+RAM+GPU)
│   ├── param_server.py            # FP32 master weights in system RAM
│   └── gpu_window.py              # GPU compute window (layer streaming)
├── data/
│   └── pipeline.py                # Math tokenizer + mmap dataset
├── math_only/
│   └── decode_mask.py             # Constrained decoding (no natural language)
├── checkpoints/                   # Saved model checkpoints (created at runtime)
└── data/tokenized/                # Pre-tokenized math corpus (created at runtime)

Quick start (Linux):
  1. Install Ubuntu 24.04 LTS (headless), partition per os/storage_partition.md
  2. Install NVIDIA driver 572+, CUDA 12.8, Python 3.11+
  3. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  4. pip install pyyaml numpy
  5. bash os/linux-optimize.sh
  6. Prepare corpus: put math text files in data/corpus/raw/
  7. python runtime/unified_train.py --config configs/saturn-0.1-small.yaml

Hardware target:
  - i7-7820HQ (4C/8T) + 16GB RAM + Quadro M1200 (4GB VRAM)
  - Model: ~100M parameters, FP32 master in RAM, FP16 compute on GPU
  - Output: math/code tokens only (constrained decoding)
== EOF ==
