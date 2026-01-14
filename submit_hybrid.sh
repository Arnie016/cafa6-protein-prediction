#!/bin/bash
# Combine Jan 10 F&P with new Aspect C and submit
set -e

echo "================================================================================"
echo "CAFA-6 HYBRID SUBMISSION"
echo "Jan 10 F&P + New Aspect C + GO Propagation"
echo "================================================================================"

cd /Users/hema/Desktop/cafa-6-protein-function-prediction

# Combine predictions
echo "Step 1: Combining predictions..."
cat predictions/pred_test_features_chunk_*.tsv > predictions/jan10_all.tsv
cp /Volumes/TRANSCEND/cafa6_predictions/lgbm_aspect_C.tsv predictions/new_C.tsv

echo "  Jan 10 F+P+C: $(wc -l predictions/jan10_all.tsv | awk '{print $1}') predictions"  
echo "  New C only: $(wc -l predictions/new_C.tsv | awk '{print $1}') predictions"

# For hybrid: use Jan 10 file as-is (already has all 3 aspects)
cp predictions/jan10_all.tsv predictions/hybrid_baseline.tsv
echo "✅ Using Jan 10 predictions (all aspects)"

# GO propagation
echo ""
echo "Step2: GO hierarchy propagation..."
python3 src/propagate_hierarchy.py \
    --obo Train/go-basic.obo \
    --infile predictions/hybrid_baseline.tsv \
    --outfile predictions/submission_final.tsv \
    --min_score 0.01

echo "✅ Final: $(wc -l predictions/submission_final.tsv | awk '{print $1}') predictions"

# Submit
echo ""
echo "Step 3: Submitting to Kaggle..."
kaggle competitions submit \
    -c cafa-6-protein-function-prediction \
    -f predictions/submission_final.tsv \
    -m "LightGBM baseline (Jan 10) with GO hierarchy propagation"

echo ""
echo "================================================================================"
echo "🎉 SUBMITTED!"
echo "================================================================================"
echo "Check: https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/submissions"
