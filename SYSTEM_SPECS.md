# System Specifications — Saturn 0.1 Development Host

**Purpose:** Hardware reference for RAM-primary model selection and training configuration.
**Privacy:** No serial numbers, MAC addresses, user names, or identifying information included.

## Processor
- **CPU:** Intel Core i7-7820HQ
- **Cores / Threads:** 4 cores, 8 logical processors
- **Base Frequency:** 2.90 GHz
- **Architecture:** 64-bit

## Graphics (Secondary)
- **GPU:** NVIDIA Quadro M1200
- **VRAM:** 4 GB (GDDR5)
- **Architecture:** Maxwell (Compute Capability 5.0)
- **FP16 Support:** Yes
- **BF16 Support:** No (Maxwell limitation)
- **Role:** Optional compute acceleration; model resides primarily in system RAM

## Memory (Primary)
- **System RAM:** 16 GB
- **Available for Training:** ~10–12 GB usable after Windows 10 overhead
- **Training Constraint:** Model weights + optimizer states + gradients must fit in ~12 GB RAM

## Storage
- **OS Drive:** C: (NTFS, ~499 GB) — Windows 10 Pro
- **Secondary / USB:** D: (FAT32, ~31.4 GB) — project transfer
- **Boot / EFI:** E: (FAT, ~33 MB)

## Operating System
- **OS:** Microsoft Windows 10 Pro
- **Version:** 10.0.19045
- **Note:** Current development targets Windows 10. Linux is the next stage.

## RAM Budget for Training (16 GB Total)
| Component | Estimated Usage |
|-----------|----------------|
| Windows 10 + background | 3–4 GB |
| Model weights (FP16) | 2 bytes / parameter |
| Optimizer states (FP32 Adam) | 8 bytes / parameter |
| Gradients (FP16) | 2 bytes / parameter |
| Activations + dataset cache | 1–3 GB |
| **Usable for training math** | **~1.3B params full fine-tune max** |
| **With QLoRA (adapters only)** | **3B–7B base model possible** |
