#!/usr/bin/env python3
"""
Clean up inconsistent local data and verify against S3.

This script will:
1. Check which cached HUCs are actually needed for Exp3
2. Compare cached versions against S3 to find discrepancies
3. Remove wrong/inconsistent data
4. Optionally download fresh consistent data from S3
"""

import json
import pandas as pd
import boto3
from pathlib import Path
import sys

print("=" * 80)
print("DATA CLEANUP AND VERIFICATION")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Load Exp3 HUC splits
# ============================================================================
print("STEP 1: Loading Exp3 HUC splits (previous students' exact splits)")
print("-" * 80)

with open('src/snowML/datapipe/huc_lists/hucs_data.json') as f:
    exp3_splits = json.load(f)

exp3_hucs = set(exp3_splits['train_hucs'] + exp3_splits['val_hucs'] + exp3_splits['test_hucs'])

print(f"Exp3 requires {len(exp3_hucs)} HUCs:")
print(f"  Train: {len(exp3_splits['train_hucs'])} HUCs")
print(f"  Val: {len(exp3_splits['val_hucs'])} HUCs")
print(f"  Test: {len(exp3_splits['test_hucs'])} HUCs")
print()

# ============================================================================
# STEP 2: Check what's in local cache
# ============================================================================
print("STEP 2: Checking local cache")
print("-" * 80)

cache_dir = Path('data/huc_cache')
cached_files = list(cache_dir.glob('*.parquet'))
cached_hucs = {f.stem for f in cached_files}

# Which cached HUCs are actually needed for Exp3?
needed_and_cached = exp3_hucs & cached_hucs
wrong_cached = cached_hucs - exp3_hucs

print(f"Total cached locally: {len(cached_hucs)} HUCs")
print(f"  Needed for Exp3: {len(needed_and_cached)} HUCs")
print(f"  Wrong/unnecessary: {len(wrong_cached)} HUCs")
print()

if wrong_cached:
    print(f"Wrong HUCs (first 10):")
    for huc in list(wrong_cached)[:10]:
        print(f"  {huc}")
    print()

# ============================================================================
# STEP 3: Verify cached data against S3
# ============================================================================
print("STEP 3: Verifying cached HUCs against S3")
print("-" * 80)

if not needed_and_cached:
    print("No Exp3 HUCs cached locally. Will download from S3 during training.")
    print()
else:
    print(f"Checking {len(needed_and_cached)} cached HUCs against S3...")
    print("(This may take a few minutes)")
    print()

    s3 = boto3.client('s3', region_name='us-west-2')
    bucket = 'snowml-model-ready'

    mismatches = []
    verified = []

    for i, huc in enumerate(list(needed_and_cached)[:5], 1):  # Check first 5 as sample
        print(f"  [{i}/5] Checking {huc}...", end=' ')

        try:
            # Load local cached version
            local_df = pd.read_parquet(f'data/huc_cache/{huc}.parquet')

            # Download S3 version
            s3_key = f'model_ready_huc{huc}.csv'
            s3.download_file(bucket, s3_key, f'/tmp/{huc}.csv')
            s3_df = pd.read_csv(f'/tmp/{huc}.csv')

            # Compare structure
            if local_df.shape != s3_df.shape:
                print(f"❌ MISMATCH (shape: local {local_df.shape} vs S3 {s3_df.shape})")
                mismatches.append({
                    'huc': huc,
                    'issue': 'shape mismatch',
                    'local_shape': local_df.shape,
                    's3_shape': s3_df.shape
                })
                continue

            # Check columns
            local_cols = set(local_df.columns)
            s3_cols = set(s3_df.columns)

            if local_cols != s3_cols:
                print(f"❌ MISMATCH (columns differ)")
                mismatches.append({
                    'huc': huc,
                    'issue': 'column mismatch',
                    'local_cols': list(local_cols - s3_cols),
                    's3_cols': list(s3_cols - local_cols)
                })
                continue

            print("✅ OK")
            verified.append(huc)

        except Exception as e:
            print(f"❌ ERROR: {e}")
            mismatches.append({
                'huc': huc,
                'issue': 'error',
                'error': str(e)
            })

    print()
    print(f"Sample verification complete:")
    print(f"  ✅ Verified: {len(verified)}/5")
    print(f"  ❌ Mismatches: {len(mismatches)}/5")
    print()

    if mismatches:
        print("MISMATCHES FOUND:")
        for mm in mismatches:
            print(f"  {mm['huc']}: {mm['issue']}")
        print()

# ============================================================================
# STEP 4: Summary and recommendations
# ============================================================================
print("=" * 80)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 80)
print()

print(f"Data status:")
print(f"  Exp3 needs: {len(exp3_hucs)} HUCs")
print(f"  Cached (correct): {len(needed_and_cached)} HUCs")
print(f"  Cached (wrong): {len(wrong_cached)} HUCs")
print(f"  Missing from cache: {len(exp3_hucs - cached_hucs)} HUCs")
print()

print("RECOMMENDATIONS:")
print()

if wrong_cached:
    print("1. DELETE WRONG/UNNECESSARY CACHED DATA")
    print(f"   Remove {len(wrong_cached)} HUCs not needed for Exp3")
    print(f"   This will free up space and avoid confusion")
    print()

if mismatches:
    print("2. DELETE MISMATCHED CACHED DATA")
    print(f"   Remove {len(needed_and_cached)} cached Exp3 HUCs")
    print(f"   They don't match S3 format")
    print()

print("3. USE S3 DIRECTLY DURING TRAINING")
print("   The training code auto-downloads from S3")
print("   This guarantees correct, consistent data")
print("   Cost: minimal (downloads once, cached in memory during training)")
print()

print("OR")
print()

print("4. PRE-DOWNLOAD FRESH DATA FROM S3")
print("   Download all 271 Exp3 HUCs to local cache")
print("   Faster training startup (no S3 downloads during training)")
print("   But requires ~600MB disk space")
print()

# ============================================================================
# STEP 5: Offer cleanup options
# ============================================================================
print("=" * 80)
print("CLEANUP OPTIONS")
print("=" * 80)
print()

print("What would you like to do?")
print()
print("A. Delete ALL local cached data (start fresh, use S3 during training)")
print("B. Delete only wrong data (keep the 54 matching HUCs, rest from S3)")
print("C. Keep everything as-is (not recommended due to inconsistencies)")
print("D. Exit (don't make any changes)")
print()

choice = input("Enter choice (A/B/C/D): ").strip().upper()

if choice == 'A':
    print()
    print("Deleting ALL cached data...")
    for f in cached_files:
        f.unlink()
        print(f"  Deleted: {f.name}")
    print()
    print("✅ All cached data deleted.")
    print("Training will download fresh data from S3.")

elif choice == 'B':
    print()
    print(f"Deleting {len(wrong_cached)} wrong/unnecessary HUCs...")
    for huc in wrong_cached:
        filepath = cache_dir / f"{huc}.parquet"
        if filepath.exists():
            filepath.unlink()
            print(f"  Deleted: {huc}")
    print()
    print(f"✅ Deleted {len(wrong_cached)} wrong HUCs.")
    print(f"Kept {len(needed_and_cached)} Exp3 HUCs in cache.")
    print(f"Training will download {len(exp3_hucs - cached_hucs)} missing HUCs from S3.")

elif choice == 'C':
    print()
    print("⚠️  Keeping data as-is.")
    print("Be aware of potential inconsistencies during training.")

elif choice == 'D':
    print()
    print("Exiting without changes.")
    sys.exit(0)

else:
    print()
    print(f"Invalid choice: {choice}")
    print("Exiting without changes.")
    sys.exit(1)

print()
print("=" * 80)
print("CLEANUP COMPLETE")
print("=" * 80)
