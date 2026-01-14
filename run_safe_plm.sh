#!/bin/bash
# Force CPU usage and single-threading to prevent macOS Mutex Lock
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""

echo "Starting Disk-Based PLM Generation..."
echo "Model: /Volumes/TRANSCEND/models/esm2_t6_8M_UR50D"
echo "Output: /Volumes/TRANSCEND/cafa6_embeddings_TEST"

# Run the test script
python3 src/test_esm_pipeline.py
