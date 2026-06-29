#!/bin/bash
# L2 persistence test under MPS contention.
# Usage: sudo bash src/experiments/run_l2_persistence_sweep.sh

set -e

REPO=/home/sriram-bh/gpu-determinism-benchmark
RESULTS=$REPO/results
DOCKER_IMG=nvcr.io/nvidia/pytorch:25.08-py3
MPS_PIPE=/tmp/nvidia-mps
MPS_LOG=/tmp/nvidia-mps-log

echo "=== L2 Persistence Test ==="

# Lock clocks
echo "--- Locking GPU clocks ---"
jetson_clocks
jetson_clocks --show 2>&1 | grep -i gpu

# Start MPS
echo "--- Starting MPS daemon ---"
export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE
export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG
mkdir -p $MPS_PIPE $MPS_LOG
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1
nvidia-cuda-mps-control -d
sleep 2
echo "MPS running."

# Run
sudo docker run --runtime nvidia --rm --network host \
    -v /home/sriram-bh:/workspace_home \
    -e CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE \
    -e CUDA_MPS_LOG_DIRECTORY=$MPS_LOG \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v $MPS_PIPE:$MPS_PIPE \
    -v $MPS_LOG:$MPS_LOG \
    $DOCKER_IMG bash -c "
cd /workspace_home/gpu-determinism-benchmark
pip install -q -r requirements.txt 2>&1 | tail -1
python3 -u src/experiments/run_l2_persistence_test.py 2>&1
" | tee $RESULTS/l2_persistence_output.txt

# Cleanup
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
echo "DONE."
