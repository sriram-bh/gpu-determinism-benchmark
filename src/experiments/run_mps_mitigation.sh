#!/bin/bash
# MPS mitigation experiment: run inference with dedicated SM partition
# while contention runs on a separate partition.
#
# This script runs OUTSIDE Docker (on the Thor host) because MPS requires
# the daemon to run on the host, with separate client processes.
#
# Usage: sudo bash src/experiments/run_mps_mitigation.sh

set -e

REPO=/home/sriram-bh/gpu-determinism-benchmark
RESULTS=$REPO/results
DOCKER_IMG=nvcr.io/nvidia/pytorch:25.08-py3
MPS_PIPE=/tmp/nvidia-mps
MPS_LOG=/tmp/nvidia-mps-log

echo "=== MPS Mitigation Experiment ==="
echo ""

# --- Step 1: Check MPS support ---
echo "--- Step 1: Checking MPS support ---"
export CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE
export CUDA_MPS_LOG_DIRECTORY=$MPS_LOG
mkdir -p $MPS_PIPE $MPS_LOG

# Kill any existing MPS daemon
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1

# Start MPS daemon
echo "Starting MPS daemon..."
nvidia-cuda-mps-control -d
sleep 2

# Verify it's running
if ! echo get_server_list | nvidia-cuda-mps-control 2>/dev/null; then
    echo "MPS daemon failed to start. Exiting."
    exit 1
fi
echo "MPS daemon running."

# --- Step 2: Run the experiment inside Docker ---
# MPS works across processes on the same GPU. Docker containers with
# --runtime nvidia share the host's GPU and MPS daemon.
echo ""
echo "--- Step 2: Running experiment ---"

sudo docker run --runtime nvidia --rm --network host \
    -v /home/sriram-bh:/workspace_home \
    -e CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE \
    -e CUDA_MPS_LOG_DIRECTORY=$MPS_LOG \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -v $MPS_PIPE:$MPS_PIPE \
    -v $MPS_LOG:$MPS_LOG \
    $DOCKER_IMG bash -c "
cd /workspace_home/gpu-determinism-benchmark
pip install -q -r requirements.txt
python3 -u src/experiments/run_mps_test.py 2>&1
" | tee $RESULTS/mps_output.txt

# --- Step 3: Cleanup ---
echo ""
echo "--- Stopping MPS daemon ---"
echo quit | nvidia-cuda-mps-control 2>/dev/null || true
echo "Done."
