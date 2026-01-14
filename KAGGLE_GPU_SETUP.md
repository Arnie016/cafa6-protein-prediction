# GPU Training Setup on Kaggle

## Step 1: Upload Your Data to Kaggle

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload these files:
   - `features/train_features.parquet`
   - Name it: "cafa-6-features"
4. Make it public or private (your choice)

## Step 2: Create a New Notebook

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Settings (right sidebar):
   - **Accelerator:** GPU T4 x2 (or P100)
   - **Internet:** ON
   - **Persistence:** Files only

## Step 3: Add Your Data

1. In the notebook, click "Add Data" (right sidebar)
2. Search for:
   - "cafa-6-features" (your uploaded dataset)
   - "cafa-6-protein-function-prediction" (the competition data)
3. Add both

## Step 4: Copy the Training Code

1. Open `kaggle_gpu_training.py` (in your project folder)
2. Copy ALL the code
3. Paste into the Kaggle notebook
4. Click "Run All"

## Step 5: Wait for Training

- **Time:** ~15-30 minutes on GPU
- **Progress:** Watch the output logs
- You'll see it training 500 terms per aspect (instead of 200)

## Step 6: Download the Models

After training completes:
1. Click "Output" tab (right sidebar)
2. Download these files:
   - `srp_projection.pkl`
   - `lgbm_F_gpu.pkl`
   - `lgbm_P_gpu.pkl`
   - `lgbm_C_gpu.pkl`

## Step 7: Make Predictions Locally

1. Replace your old models with the new GPU-trained ones
2. Run the prediction script (same as before)
3. Submit the new predictions!

---

## What's Better in GPU Version?

| Feature | CPU (Current) | GPU (New) |
|---------|--------------|-----------|
| Terms per aspect | 200 | 500 |
| Trees per model | 20 | 100 |
| Max depth | 3 | 8 |
| Training time | 2 hours | 15-30 min |
| Expected score | Baseline | 20-30% better |

---

## Troubleshooting

**If you get "GPU not available":**
- Make sure you selected GPU in notebook settings
- Try changing to "GPU P100" instead of "GPU T4 x2"

**If you get "File not found":**
- Check the paths in the script match your dataset names
- Kaggle paths are usually `/kaggle/input/your-dataset-name/`

**If it runs out of memory:**
- Reduce `n_terms_per_aspect` from 500 to 300
- Reduce `BATCH_SIZE` from 10000 to 5000
