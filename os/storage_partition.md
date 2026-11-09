# =============================================================================
# os/storage_partition.md
# Saturn 0.1 — Storage Partitioning Recommendation
#
# Current disk: 50 MB/s read on Windows (likely HDD or slow SATA SSD).
# Training will stream tokenized data constantly; disk is a bottleneck.
#
# Recommended layout on Linux (Ubuntu 24.04):
#
#   /dev/sda1  /              ext4   40 GB   OS, code, Python env
#   /dev/sda2  /home           ext4   remainder  User files, documents
#   /dev/sda3  /saturn        ext4/xfs  dedicated  Training data, checkpoints, mmap files
#   swap                         8 GB   Swap (16GB RAM + 8GB swap for dataset overflow)
#
# Why separate /saturn:
#   - Keeps training I/O isolated from OS page cache pressure
#   - Can tune mount options: noatime, nodiratime, data=writeback (faster, less safe)
#   - Easy to reformat/replace with faster NVMe later without touching OS
#
# Mount options for /saturn (/etc/fstab):
#   UUID=...  /saturn  ext4  defaults,noatime,nodiratime,commit=60  0  2
#
# tmpfs ramdisk for active training shards (optional, if dataset < 8GB):
#   mkdir -p /saturn/active
#   mount -t tmpfs -o size=8G,mode=1777 tmpfs /saturn/active
#   Copy current epoch shard to /saturn/active for RAM-speed access
#
# After partitioning:
#   1. Run os/linux-optimize.sh
#   2. Install NVIDIA driver, CUDA, PyTorch
#   3. Pre-tokenize corpus to /saturn/data/tokenized/math_corpus.bin
#   4. Run runtime/unified_train.py
#
# Expected speedup vs Windows:
#   - ext4 vs NTFS: ~10-20% better sequential read
#   - No Windows Defender scanning training files: significant
#   - No WSL2 overhead: direct CUDA, full GPU VRAM available
# =============================================================================
