"""
Download PROTGOAT pre-computed embeddings from Kaggle
"""

import os

print("=" * 80)
print("PROTGOAT Embeddings Download")
print("=" * 80)

# Create directory
os.makedirs("embeddings/protgoat", exist_ok=True)

print("\n1. Installing kagglehub...")
os.system("pip3 install kagglehub -q")

print("\n2. Downloading PROTGOAT embeddings (this may take a while)...")
print("   Dataset: cafa5-train-test-data")
print("   Size: Several GB")

try:
    import kagglehub
    
    # Download latest version
    path = kagglehub.dataset_download("zmcxjt/cafa5-train-test-data")
    
    print(f"\n✅ Download complete!")
    print(f"Path to dataset files: {path}")
    
    # List files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            size_mb = os.path.getsize(filepath) / (1024**2)
            print(f"  {file} ({size_mb:.1f} MB)")
    
    print(f"\n📁 Files are in: {path}")
    print("   You can access them from there or copy to embeddings/protgoat/")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nIf you see 'Kaggle credentials not found':")
    print("1. Go to: https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New Token'")
    print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
    print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
    print("6. Re-run this script")
