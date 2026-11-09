#!/usr/bin/env bash
# =============================================================================
# os/linux-optimize.sh
# Saturn 0.1 — Linux system tuning for unified CPU+RAM+GPU training
# Run as root on fresh Ubuntu 24.04 LTS install
# =============================================================================
set -euo pipefail

echo "[saturn-os] Applying kernel and system optimizations..."

# --- 1. CPU governor: performance (disable powersave) ---
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" > "$cpu" 2>/dev/null || true
done

# --- 2. Swappiness: reduce swap pressure, keep training data in RAM ---
sysctl -w vm.swappiness=10
sysctl -w vm.vfs_cache_pressure=50

# --- 3. Transparent Huge Pages: can help with large allocations ---
echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
echo always > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true

# --- 4. Increase max map count (mmap'd datasets) ---
sysctl -w vm.max_map_count=262144

# --- 5. NIC and IRQ affinity: isolate CPU cores for training if desired ---
# (Optional: echo 1 > /proc/irq/*/smp_affinity_list for specific cores)

# --- 6. File limits ---
ulimit -n 65536
echo 65536 > /proc/sys/fs/file-max

# --- 7. Disable transparent huge page defrag latency spikes (optional, uncomment if needed) ---
# echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

# --- 8. Persist sysctl settings ---
cat > /etc/sysctl.d/99-saturn.conf <<'EOF'
vm.swappiness=10
vm.vfs_cache_pressure=50
vm.max_map_count=262144
EOF

echo "[saturn-os] Kernel tuning complete."
echo "[saturn-os] Next: install NVIDIA driver 572+, CUDA 12.8, PyTorch 2.x, then run setup_env.sh"
