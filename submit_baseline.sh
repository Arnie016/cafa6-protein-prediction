#!/bin/bash
# Use Jan 10 predictions and submit NOW
set -e

echo "================================================================================"
echo "CAFA-6 SUBMISSION - Using Jan 10 Predictions"
echo "================================================================================"

cd /Users/hema/Desktop/cafa-6-protein-function-prediction

# Combine Jan 10 predictions
echo "Step 1: Combining predictions..."
cat predictions/pred_test_features_chunk_*.tsv > predictions/baseline_combined.tsv
echo "✅ Combined: $(wc -l predictions/baseline_combined.tsv | awk '{print $1}') predictions"

# Apply GO hierarchy
echo ""
echo "Step 2: Applying GO hierarchy propagation..."
python3 src/propagate_hierarchy.py \
    --obo Train/go-basic.obo \
    --infile predictions/baseline_combined.tsv \
    --outfile predictions/submission_final.tsv \
    --min_score 0.01

echo "✅ Propagation complete!"

# Submit to Kaggle
echo ""
echo "Step 3: Submitting to Kaggle..."
kaggle competitions submit \
    -c cafa-6-protein-function-prediction \
    -f predictions/submission_final.tsv \
    -m "LightGBM baseline (Jan 10) with GO hierarchy propagation"

echo ""
echo "================================================================================"
echo "🎉 SUBMISSION COMPLETE!"
echo "================================================================================"
echo ""
echo "Check: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/submissions"
