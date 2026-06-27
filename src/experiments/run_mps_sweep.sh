#!/bin/bash
# Full mitigation sweep: lock clocks, test with/without MPS, test L2.
#
# Usage: sudo bash src/experiments/run_mps_sweep.sh

set -e

REPO=/home/sriram-bh/gpu-determinism-benchmark
RESULTS=$REPO/results
DOCKER_IMG=nvcr.io/nvidia/pytorch:25.08-py3
MPS_PIPE=/tmp/nvidia-mps
MPS_LOG=/tmp/nvidia-mps-log
DOCKER_COMMON="--runtime nvidia --rm --network host -v /home/sriram-bh:/workspace_home -e CUBLAS_WORKSPACE_CONFIG=:4096:8"

echo "=== Full Mitigation Sweep ==="

# Step 0: Lock clocks
echo ""
echo "--- Locking GPU clocks (jetson_clocks) ---"
jetson_clocks
jetson_clocks --show 2>&1 | grep -i gpu
echo "Clocks locked."

# Kill any existing MPS
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1

# === Run 1: WITHOUT MPS (same-process contention only + L2) ===
echo ""
echo "=========================================="
echo "  RUN 1: WITHOUT MPS (baseline + L2 test)"
echo "=========================================="
sudo docker run $DOCKER_COMMON \
    $DOCKER_IMG bash -c "
cd /workspace_home/gpu-determinism-benchmark
pip install -q -r requirements.txt 2>&1 | tail -1
echo '--- NO MPS ---'
python3 -u src/experiments/run_full_mitigation_sweep.py --l2 2>&1
" | tee $RESULTS/sweep_no_mps.txt

# === Run 2: WITH MPS (stream priority, SM partitioning, combo) ===
echo ""
echo "=========================================="
echo "  RUN 2: WITH MPS"
echo "=========================================="

export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE
export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG
mkdir -p $MPS_PIPE $MPS_LOG

echo "Starting MPS daemon..."
nvidia-cuda-mps-control -d
sleep 2
echo "MPS daemon running."

sudo docker run $DOCKER_COMMON \
    -e CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE \
    -e CUDA_MPS_LOG_DIRECTORY=$MPS_LOG \
    -v $MPS_PIPE:$MPS_PIPE \
    -v $MPS_LOG:$MPS_LOG \
    $DOCKER_IMG bash -c "
cd /workspace_home/gpu-determinism-benchmark
pip install -q -r requirements.txt 2>&1 | tail -1
echo '--- MPS ON ---'
python3 -u src/experiments/run_full_mitigation_sweep.py --mps 2>&1
" | tee $RESULTS/sweep_mps.txt

# Cleanup
echo ""
echo "--- Stopping MPS daemon ---"
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
echo ""
echo "Results in:"
echo "  $RESULTS/sweep_no_mps.txt"
echo "  $RESULTS/sweep_mps.txt"
echo "ALL DONE."
