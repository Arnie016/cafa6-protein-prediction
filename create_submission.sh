#!/bin/bash
# Quick submission pipeline for CAFA-6 with auto-submit to Kaggle

set -e

echo "================================================================================"
echo "CAFA-6 QUICK SUBMISSION PIPELINE (with Auto-Submit)"
echo "================================================================================"

PREDICTIONS="/Volumes/TRANSCEND/cafa6_predictions/lgbm_baseline_new.tsv"
GO_OBO="Train/go-basic.obo"
OUTPUT_DIR="/Volumes/TRANSCEND/cafa6_predictions"
SUBMISSION_FILE="$OUTPUT_DIR/submission.tsv"
COMPETITION="cafa-6-protein-function-prediction"

echo ""
echo "Step 1: Waiting for predictions to complete..."
while [ ! -f "$PREDICTIONS" ]; do
    echo "  Predictions not ready yet... waiting 30s"
    sleep 30
done

echo "✅ Predictions found: $PREDICTIONS"
wc -l "$PREDICTIONS"

echo ""
echo "Step 2: Applying GO hierarchy propagation..."
python3 src/propagate_hierarchy.py \
    --obo "$GO_OBO" \
    --infile "$PREDICTIONS" \
    --outfile "$OUTPUT_DIR/submission_propagated.tsv" \
    --min_score 0.01 \
    --keep_roots

echo "✅ Propagation complete!"

echo ""
echo "Step 3: Creating final submission file..."
cp "$OUTPUT_DIR/submission_propagated.tsv" "$SUBMISSION_FILE"
echo "✅ Submission file ready!"

echo ""
echo "📊 Submission stats:"
wc -l "$SUBMISSION_FILE"
echo ""
echo "Sample predictions:"
head -10 "$SUBMISSION_FILE"

echo ""
echo "Step 4: Submitting to Kaggle..."
kaggle competitions submit \
    -c "$COMPETITION" \
    -f "$SUBMISSION_FILE" \
    -m "LightGBM baseline with GO hierarchy propagation - trained on Kaggle GPU"

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "🎉 SUBMISSION SUCCESSFUL!"
    echo "================================================================================"
    echo ""
    echo "Check your submission at:"
    echo "https://www.kaggle.com/competitions/$COMPETITION/submissions"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "⚠️  Kaggle submission failed - you can submit manually"
    echo "================================================================================"
    echo ""
    echo "File location: $SUBMISSION_FILE"
    echo "Manual upload: https://www.kaggle.com/competitions/$COMPETITION/submit"
    echo ""
fi
