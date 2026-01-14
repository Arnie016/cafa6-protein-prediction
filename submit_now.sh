#!/bin/bash
# Combine predictions and submit
set -e

cd /Users/hema/Desktop/cafa-6-protein-function-prediction

echo "================================================================================"
echo "CAFA-6 SUBMISSION - Combining All Predictions"
echo "================================================================================"

# Combine Jan 10 predictions (F+P+C) with new C
echo "Step 1: Using Jan 10 predictions + new Aspect C..."
cat predictions/pred_test_features_chunk_*.tsv > predictions/baseline_FP.tsv
cp /Volumes/TRANSCEND/cafa6_predictions/lgbm_aspect_C.tsv predictions/baseline_C.tsv

# Combine all
cat predictions/baseline_FP.tsv predictions/baseline_C.tsv > predictions/combined_all.tsv
echo "✅ Combined: $(wc -l predictions/combined_all.tsv | awk '{print $1}') predictions"

# GO propagation
echo ""
echo "Step 2: Applying GO hierarchy propagation..."
python3 src/propagate_hierarchy.py \
    --obo Train/go-basic.obo \
    --infile predictions/combined_all.tsv \
    --outfile predictions/submission.tsv \
    --min_score 0.01

echo "✅ Final submission: $(wc -l predictions/submission.tsv | awk '{print $1}') predictions"

# Submit
echo ""
echo "Step 3: Submitting to Kaggle..."
kaggle competitions submit \
    -c cafa-6-protein-function-prediction \
    -f predictions/submission.tsv \
    -m "LightGBM baseline + new Aspect C + GO propagation"

echo ""
echo "🎉 SUBMITTED!"
echo "Check: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/submissions"
