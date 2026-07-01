# House Prices: Advanced Regression Techniques
# ─────────────────────────────────────────────
# Kaggle API download script.
# Requires: pip install kaggle
# Setup: Place your kaggle.json in ~/.kaggle/kaggle.json
#
# Usage:
#   python data/download_data.py
#
# Or manually download from:
#   https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
# and place train.csv, test.csv, sample_submission.csv in this folder.

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent

def download():
    print("Attempting to download dataset via Kaggle API...")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle",
                "competitions", "download",
                "-c", "house-prices-advanced-regression-techniques",
                "-p", str(DATA_DIR),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        print("✅ Dataset downloaded successfully!")
        for f in ["train.csv", "test.csv", "sample_submission.csv"]:
            path = DATA_DIR / f
            if path.exists():
                print(f"   {f} ({path.stat().st_size // 1024} KB)")
    except subprocess.CalledProcessError as e:
        print(f"❌ Kaggle CLI error:\n{e.stderr}")
        print("\nManual download instructions:")
        print("1. Go to: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data")
        print("2. Download: train.csv, test.csv, sample_submission.csv")
        print(f"3. Place them in: {DATA_DIR}")
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        print("   Then configure: place kaggle.json in ~/.kaggle/")

if __name__ == "__main__":
    download()
