#!/usr/bin/env python3
"""
Backup all checkpoints from SageMaker Studio to S3
Run this on SageMaker Studio terminal

Usage:
    python backup_checkpoints_to_s3.py

This will backup:
1. /home/sagemaker-user/checkpoints/ (480 .pth files)
2. /home/sagemaker-user/exp1a_results/ (.csv files)
3. /home/sagemaker-user/exp1b_corrected_results/ (.csv files)
4. /home/sagemaker-user/exp2_finetune_results/ (10 .pth files)
5. /home/sagemaker-user/evaluation_results/ (.csv files)
6. Training logs (exp1a_training.log, exp1b_training.log, finetune.log)

All files uploaded to S3 with organized structure and progress shown.
"""

import boto3
import os
from pathlib import Path
from datetime import datetime
import hashlib

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 Configuration
S3_BUCKET = "snowml-model-ready"  # Use existing bucket or change to your bucket
S3_PREFIX = "checkpoints/simran_thesis"  # Organize by student/project
REGION = "us-west-2"

# Local directories on SageMaker
LOCAL_CHECKPOINT_DIR = Path("/home/sagemaker-user/checkpoints")
LOCAL_EXP1A_RESULTS_DIR = Path("/home/sagemaker-user/exp1a_results")
LOCAL_EXP1B_RESULTS_DIR = Path("/home/sagemaker-user/exp1b_corrected_results")
LOCAL_EXP2_DIR = Path("/home/sagemaker-user/exp2_finetune_results")
LOCAL_EVAL_RESULTS_DIR = Path("/home/sagemaker-user/evaluation_results")

# Backup organization options
ORGANIZE_BY_DATE = True  # Add date folder
DATE_FOLDER = datetime.now().strftime("%Y%m%d")  # e.g., 20260930

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_file_md5(file_path):
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def human_readable_size(size_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def upload_to_s3(local_path, s3_key, bucket, s3_client):
    """Upload file to S3 with progress"""
    file_size = os.path.getsize(local_path)

    try:
        print(f"  📤 Uploading: {local_path.name} ({human_readable_size(file_size)})")

        # Upload with progress callback
        s3_client.upload_file(
            str(local_path),
            bucket,
            s3_key,
            Callback=lambda bytes_transferred: None  # Could add progress bar here
        )

        print(f"  ✅ Uploaded: s3://{bucket}/{s3_key}")
        return True

    except Exception as e:
        print(f"  ❌ Failed: {str(e)}")
        return False

def backup_directory(local_dir, s3_prefix, bucket, s3_client, file_patterns=None):
    """Backup all files in directory to S3"""

    if not local_dir.exists():
        print(f"⚠️  Directory does not exist: {local_dir}")
        return 0, 0, 0

    # Default patterns for checkpoints
    if file_patterns is None:
        file_patterns = ["*.pth", "*.json"]

    # Find all matching files
    all_files = []
    for pattern in file_patterns:
        all_files.extend(list(local_dir.glob(pattern)))

    if not all_files:
        print(f"⚠️  No files found in {local_dir}")
        return 0, 0, 0

    # Count file types
    file_type_counts = {}
    for f in all_files:
        ext = f.suffix
        file_type_counts[ext] = file_type_counts.get(ext, 0) + 1

    print(f"\n📁 Found {len(all_files)} files in {local_dir}")
    for ext, count in sorted(file_type_counts.items()):
        print(f"   - {ext} files: {count}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"   - Total size: {human_readable_size(total_size)}")

    uploaded = 0
    failed = 0

    # Upload each file
    for local_file in all_files:
        # Create S3 key preserving relative structure
        relative_path = local_file.relative_to(local_dir.parent)
        s3_key = f"{s3_prefix}/{relative_path}"

        if upload_to_s3(local_file, s3_key, bucket, s3_client):
            uploaded += 1
        else:
            failed += 1

    return len(all_files), uploaded, failed

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BACKUP CHECKPOINTS TO S3")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize S3 client
    print("[1/4] Initializing S3 client...")
    try:
        s3_client = boto3.client('s3', region_name=REGION)
        # Test connection
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"✅ Connected to S3 bucket: s3://{S3_BUCKET}")
    except Exception as e:
        print(f"❌ Failed to connect to S3: {str(e)}")
        print("\nPossible issues:")
        print("1. AWS credentials not configured")
        print("2. Bucket doesn't exist")
        print("3. No permission to access bucket")
        print("\nTo fix:")
        print("  aws configure")
        print("  # Or check IAM permissions in AWS Console")
        return

    # Build S3 prefix with optional date folder
    if ORGANIZE_BY_DATE:
        base_prefix = f"{S3_PREFIX}/{DATE_FOLDER}"
    else:
        base_prefix = S3_PREFIX

    print(f"\n📦 S3 destination: s3://{S3_BUCKET}/{base_prefix}/")
    print()

    # Backup main checkpoints directory
    print("[2/8] Backing up main checkpoints...")
    total1, uploaded1, failed1 = backup_directory(
        LOCAL_CHECKPOINT_DIR,
        base_prefix,
        S3_BUCKET,
        s3_client,
        file_patterns=["*.pth", "*.json"]
    )

    # Backup Exp1A results
    print("\n[3/8] Backing up Exp1A results...")
    total2, uploaded2, failed2 = backup_directory(
        LOCAL_EXP1A_RESULTS_DIR,
        base_prefix,
        S3_BUCKET,
        s3_client,
        file_patterns=["*.csv", "*.json"]
    )

    # Backup Exp1B results
    print("\n[4/8] Backing up Exp1B results...")
    total3, uploaded3, failed3 = backup_directory(
        LOCAL_EXP1B_RESULTS_DIR,
        base_prefix,
        S3_BUCKET,
        s3_client,
        file_patterns=["*.csv", "*.json"]
    )

    # Backup Exp2 fine-tuning results
    print("\n[5/8] Backing up Exp2 fine-tuning results...")
    total4, uploaded4, failed4 = backup_directory(
        LOCAL_EXP2_DIR,
        base_prefix,
        S3_BUCKET,
        s3_client,
        file_patterns=["*.pth", "*.json"]
    )

    # Backup evaluation results (if exists)
    print("\n[6/8] Backing up evaluation results...")
    total5, uploaded5, failed5 = backup_directory(
        LOCAL_EVAL_RESULTS_DIR,
        base_prefix,
        S3_BUCKET,
        s3_client,
        file_patterns=["*.csv", "*.json"]
    )

    # Backup training logs
    print("\n[7/8] Backing up training logs...")
    log_files = [
        LOCAL_LOGS_DIR / "exp1a_training.log",
        LOCAL_LOGS_DIR / "exp1b_training.log",
        LOCAL_LOGS_DIR / "finetune.log",
        LOCAL_LOGS_DIR / "training_summary.csv"
    ]

    total6 = 0
    uploaded6 = 0
    failed6 = 0

    for log_file in log_files:
        if log_file.exists():
            total6 += 1
            s3_key = f"{base_prefix}/logs/{log_file.name}"
            if upload_to_s3(log_file, s3_key, S3_BUCKET, s3_client):
                uploaded6 += 1
            else:
                failed6 += 1
        else:
            print(f"  ⚠️  Log file not found: {log_file.name}")

    # Summary
    print("\n[8/8] Summary")
    print("=" * 80)

    total_all = total1 + total2 + total3 + total4 + total5 + total6
    uploaded_all = uploaded1 + uploaded2 + uploaded3 + uploaded4 + uploaded5 + uploaded6
    failed_all = failed1 + failed2 + failed3 + failed4 + failed5 + failed6

    print(f"Total files found: {total_all}")
    print(f"Successfully uploaded: {uploaded_all}")
    print(f"Failed: {failed_all}")

    print(f"\nBreakdown by directory:")
    print(f"  Checkpoints:        {uploaded1}/{total1}")
    print(f"  Exp1A Results:      {uploaded2}/{total2}")
    print(f"  Exp1B Results:      {uploaded3}/{total3}")
    print(f"  Exp2 Results:       {uploaded4}/{total4}")
    print(f"  Evaluation Results: {uploaded5}/{total5}")
    print(f"  Training Logs:      {uploaded6}/{total6}")

    if failed_all > 0:
        print(f"\n⚠️  Some files failed to upload. Check errors above.")
    else:
        print(f"\n✅ All files backed up successfully!")

    print(f"\n📍 Location: s3://{S3_BUCKET}/{base_prefix}/")
    print()
    print("To verify backup:")
    print(f"  aws s3 ls s3://{S3_BUCKET}/{base_prefix}/ --recursive --human-readable")
    print()
    print("To download later:")
    print(f"  aws s3 sync s3://{S3_BUCKET}/{base_prefix}/ ./downloaded_checkpoints/")
    print()
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
