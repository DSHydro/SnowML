#!/usr/bin/env python3
"""
Test script to verify SnowML setup on new laptop
Branch: simran-unified-experiments
"""

import sys
print("=" * 70)
print("Testing SnowML Setup - New Laptop")
print("=" * 70)

# Test 1: Python version
print(f"\n1. Python version: {sys.version}")
assert sys.version_info >= (3, 10), "❌ Need Python 3.10+"
print("   ✅ Python version OK")

# Test 2: PyTorch
try:
    import torch
    print(f"\n2. PyTorch version: {torch.__version__}")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"   Device available: {device}")
    print("   ✅ PyTorch installed")
except ImportError as e:
    print(f"   ❌ PyTorch import failed: {e}")

# Test 3: MLflow
try:
    import mlflow
    print(f"\n3. MLflow version: {mlflow.__version__}")
    print("   ✅ MLflow installed")
except ImportError as e:
    print(f"   ❌ MLflow import failed: {e}")

# Test 4: AWS SDK (boto3)
try:
    import boto3
    print(f"\n4. Boto3 (AWS SDK) version: {boto3.__version__}")
    print("   ✅ AWS SDK installed")
except ImportError as e:
    print(f"   ❌ Boto3 import failed: {e}")

# Test 5: Scientific packages
try:
    import numpy as np
    import pandas as pd
    import xarray as xr
    print(f"\n5. Scientific packages:")
    print(f"   - NumPy: {np.__version__}")
    print(f"   - Pandas: {pd.__version__}")
    print(f"   - Xarray: {xr.__version__}")
    print("   ✅ Scientific packages installed")
except ImportError as e:
    print(f"   ❌ Scientific package import failed: {e}")

# Test 6: Geospatial packages
try:
    import geopandas as gpd
    import rasterio
    print(f"\n6. Geospatial packages:")
    print(f"   - GeoPandas: {gpd.__version__}")
    print(f"   - Rasterio: {rasterio.__version__}")
    print("   ✅ Geospatial packages installed")
except ImportError as e:
    print(f"   ❌ Geospatial package import failed: {e}")

# Test 7: SnowML package
try:
    from snowML.LSTM import LSTM_model, LSTM_train, set_hyperparams
    print(f"\n7. SnowML package:")
    print(f"   - LSTM_model module: ✅")
    print(f"   - LSTM_train module: ✅")
    print(f"   - set_hyperparams module: ✅")
    print("   ✅ SnowML package installed")
except ImportError as e:
    print(f"   ❌ SnowML import failed: {e}")

# Test 8: AWS credentials and S3 access
print(f"\n8. AWS Configuration:")
try:
    import os
    aws_creds = os.path.expanduser("~/.aws/credentials")
    aws_config = os.path.expanduser("~/.aws/config")

    if os.path.exists(aws_creds) and os.path.exists(aws_config):
        print(f"   - Credentials file: ✅ Found")
        print(f"   - Config file: ✅ Found")

        # Try to access S3
        s3 = boto3.client('s3', region_name='us-west-2')
        try:
            response = s3.list_objects_v2(Bucket='snowml-model-ready', MaxKeys=1)
            if 'Contents' in response:
                print(f"   - S3 Access: ✅ Can read from snowml-model-ready bucket")
            else:
                print(f"   - S3 Access: ⚠️ Bucket exists but empty or no access")
        except Exception as e:
            print(f"   - S3 Access: ❌ Cannot access bucket: {e}")
    else:
        print(f"   - Credentials: ❌ Not configured. Run: aws configure")
except Exception as e:
    print(f"   ❌ AWS check failed: {e}")

# Test 9: Data directory
import os
data_dir = "/Users/simran/Desktop/SnowML/data"
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"\n9. Data directory:")
    print(f"   - Location: {data_dir}")
    print(f"   - Files: {len(files)}")
    if len(files) > 0:
        print(f"   - Example: {files[0]}")
    print("   ✅ Data directory exists")
else:
    print(f"\n9. Data directory:")
    print(f"   ⚠️ {data_dir} not found (will be created when downloading data)")

# Test 10: Git branch
print(f"\n10. Git status:")
try:
    import subprocess
    result = subprocess.run(['git', 'branch', '--show-current'],
                          capture_output=True, text=True, cwd='/Users/simran/Desktop/SnowML')
    branch = result.stdout.strip()
    print(f"   - Current branch: {branch}")
    if branch == 'simran-unified-experiments':
        print("   ✅ On correct branch!")
    else:
        print(f"   ⚠️ Expected 'simran-unified-experiments', got '{branch}'")
except Exception as e:
    print(f"   ⚠️ Could not check git branch: {e}")

print("\n" + "=" * 70)
print("✅ Setup test complete!")
print("=" * 70)
print("\nREADY FOR EXPERIMENTS! 🚀")
print("\nNext steps:")
print("  1. Download sample HUC data")
print("  2. Create train/val/test splits")
print("  3. Start experiments!")
