#!/bin/bash
# MPS mitigation experiment: compare cross-process contention
# WITH and WITHOUT MPS to isolate MPS's contribution.
#
# Usage: sudo bash src/experiments/run_mps_mitigation.sh

set -e

REPO=/home/sriram-bh/gpu-determinism-benchmark
RESULTS=$REPO/results
DOCKER_IMG=nvcr.io/nvidia/pytorch:25.08-py3
MPS_PIPE=/tmp/nvidia-mps
MPS_LOG=/tmp/nvidia-mps-log
DOCKER_COMMON="--runtime nvidia --rm --network host -v /home/sriram-bh:/workspace_home -e CUBLAS_WORKSPACE_CONFIG=:4096:8"

echo "=== MPS Mitigation Experiment ==="

# --- Run 1: WITHOUT MPS ---
echo ""
echo "=========================================="
echo "  RUN 1: WITHOUT MPS (cross-process baseline)"
echo "=========================================="

# Make sure MPS is off
export CUDA_MPS_PIPE_DIRECTORY=""
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1

sudo docker run $DOCKER_COMMON \
    $DOCKER_IMG bash -c "
cd /workspace_home/gpu-determinism-benchmark
pip install -q -r requirements.txt
echo '--- MPS OFF ---'
python3 -u src/experiments/run_mps_test.py 2>&1
" | tee $RESULTS/mps_output_OFF.txt

# --- Run 2: WITH MPS ---
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

sudo docker run $DOCKER_COMMON \
    -e CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE \
    -e CUDA_MPS_LOG_DIRECTORY=$MPS_LOG \
    -v $MPS_PIPE:$MPS_PIPE \
    -v $MPS_LOG:$MPS_LOG \
    $DOCKER_IMG bash -c "
cd /workspace_home/gpu-determinism-benchmark
pip install -q -r requirements.txt
echo '--- MPS ON ---'
python3 -u src/experiments/run_mps_test.py 2>&1
" | tee $RESULTS/mps_output_ON.txt

# Cleanup
echo ""
echo "--- Stopping MPS daemon ---"
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
echo "Done. Results in $RESULTS/mps_output_OFF.txt and mps_output_ON.txt"
