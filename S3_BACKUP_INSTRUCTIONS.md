# S3 Checkpoint Backup Instructions

**Date:** September 30, 2026  
**Purpose:** Backup all SageMaker checkpoints to S3 for long-term storage

---

## Why Backup to S3?

**Reasons:**
1. **SageMaker storage is expensive** - EFS storage costs ~$0.30/GB/month
2. **Checkpoints are large** - 240 files × 2 experiments = ~50-100 GB
3. **S3 is cheaper** - Standard S3 is $0.023/GB/month (13× cheaper)
4. **Archival safety** - Checkpoints preserved even if SageMaker instance deleted
5. **Easy sharing** - Can share S3 paths with collaborators/professors
6. **Glacier for long-term** - Can move to Glacier after thesis ($0.004/GB/month)

**Cost Comparison:**
| Storage Type | Cost/GB/month | 100 GB Cost/month |
|--------------|---------------|-------------------|
| SageMaker EFS | $0.30 | $30 |
| S3 Standard | $0.023 | $2.30 |
| S3 Glacier | $0.004 | $0.40 |

---

## Quick Start (Run on SageMaker Studio)

### Step 1: Upload the Backup Script

1. Open SageMaker Studio JupyterLab
2. Upload `backup_checkpoints_to_s3.py` to `/home/sagemaker-user/`
3. Or create it directly in JupyterLab

### Step 2: Configure the Script (Optional)

**Edit these variables if needed:**
```python
S3_BUCKET = "snowml-model-ready"  # Change if using different bucket
S3_PREFIX = "checkpoints/simran_thesis"  # Your folder structure
ORGANIZE_BY_DATE = True  # Creates date-based folders
```

**Default S3 structure:**
```
s3://snowml-model-ready/checkpoints/simran_thesis/20260930/
├── checkpoints/
│   ├── Exp1A_Base_1e-3_epoch1.pth
│   ├── Exp1A_Base_1e-3_epoch2.pth
│   └── ... (all 480 checkpoints: 240 Exp1A + 240 Exp1B)
│
├── exp1a_results/
│   ├── Base_1e-3_Test_A_metrics.csv
│   ├── Base_1e-3_Test_B_metrics.csv
│   ├── Wind_3e-4_Test_A_metrics.csv
│   ├── Wind_3e-4_Test_B_metrics.csv
│   └── ... (all Exp1A evaluation CSVs)
│
├── exp1b_corrected_results/
│   ├── Base_3e-4_Test_A_metrics (1).csv
│   ├── Base_3e-4_Test_B_metrics (1).csv
│   ├── Wind_3e-4_Test_A_metrics (2).csv
│   └── ... (all Exp1B evaluation CSVs)
│
├── exp2_finetune_results/
│   ├── FineTune_Base_epoch1.pth
│   ├── FineTune_Base_epoch2.pth
│   └── ... (10 Exp2 checkpoints + JSON files)
│
├── evaluation_results/
│   └── ... (additional evaluation CSVs if present)
│
└── logs/
    ├── exp1a_training.log
    ├── exp1b_training.log
    ├── finetune.log
    └── training_summary.csv
```

### Step 3: Run the Backup

**In SageMaker terminal:**
```bash
cd /home/sagemaker-user
python backup_checkpoints_to_s3.py
```

**Expected output:**
```
================================================================================
BACKUP CHECKPOINTS TO S3
================================================================================
Started: 2026-09-30 14:30:00

[1/8] Initializing S3 client...
✅ Connected to S3 bucket: s3://snowml-model-ready

📦 S3 destination: s3://snowml-model-ready/checkpoints/simran_thesis/20260930/

[2/8] Backing up main checkpoints...
📁 Found 480 files in /home/sagemaker-user/checkpoints
   - .pth files: 480
   - Total size: 52.34 GB
  📤 Uploading: Exp1A_Base_1e-3_epoch1.pth (112.45 MB)
  ✅ Uploaded: s3://.../checkpoints/Exp1A_Base_1e-3_epoch1.pth
  ...

[3/8] Backing up Exp1A results...
📁 Found 16 files in /home/sagemaker-user/exp1a_results
   - .csv files: 16
   - Total size: 45.23 MB
  ...

[4/8] Backing up Exp1B results...
📁 Found 10 files in /home/sagemaker-user/exp1b_corrected_results
   - .csv files: 10
   - Total size: 28.67 MB
  ...

[5/8] Backing up Exp2 fine-tuning results...
📁 Found 12 files in /home/sagemaker-user/exp2_finetune_results
   - .pth files: 10
   - .json files: 2
   - Total size: 1.12 GB
  ...

[6/8] Backing up evaluation results...
📁 Found 8 files in /home/sagemaker-user/evaluation_results
   - .csv files: 8
   - Total size: 15.34 MB
  ...

[7/8] Backing up training logs...
  📤 Uploading: exp1a_training.log (5.23 MB)
  ✅ Uploaded: s3://.../logs/exp1a_training.log
  📤 Uploading: exp1b_training.log (4.87 MB)
  ✅ Uploaded: s3://.../logs/exp1b_training.log
  📤 Uploading: finetune.log (1.12 MB)
  ✅ Uploaded: s3://.../logs/finetune.log

[8/8] Summary
================================================================================
Total files found: 526
Successfully uploaded: 526
Failed: 0

Breakdown by directory:
  Checkpoints:        480/480
  Exp1A Results:      16/16
  Exp1B Results:      10/10
  Exp2 Results:       12/12
  Evaluation Results: 8/8
  Training Logs:      4/4

✅ All files backed up successfully!

📍 Location: s3://snowml-model-ready/checkpoints/simran_thesis/20260930/

To verify backup:
  aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive --human-readable

To download later:
  aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ ./downloaded_checkpoints/

Finished: 2026-09-30 14:55:00
================================================================================
```

---

## Verify the Backup

### Check files were uploaded:
```bash
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive --human-readable
```

### Count total files:
```bash
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive | wc -l
# Should show ~526 files (480 checkpoints + results + logs)
```

### Check total size:
```bash
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive --summarize --human-readable
```

---

## Download Checkpoints Later

### Download all checkpoints:
```bash
# Download to local machine
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ ./downloaded_checkpoints/
```

### Download specific checkpoint:
```bash
# Download single file
aws s3 cp s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/Exp1B_Base_3e-4_epoch14.pth ./
```

### Download only Exp1B checkpoints:
```bash
# Download filtered set
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/ ./ \
  --exclude "*" \
  --include "Exp1B_*"
```

---

## Alternative: Use AWS CLI Directly (Without Script)

### Backup using aws s3 sync:
```bash
# Backup checkpoints directory
aws s3 sync /home/sagemaker-user/checkpoints/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/ \
  --region us-west-2

# Backup Exp1A results
aws s3 sync /home/sagemaker-user/exp1a_results/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1a_results/ \
  --region us-west-2

# Backup Exp1B results
aws s3 sync /home/sagemaker-user/exp1b_corrected_results/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1b_corrected_results/ \
  --region us-west-2

# Backup Exp2 results
aws s3 sync /home/sagemaker-user/exp2_finetune_results/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp2_finetune_results/ \
  --region us-west-2

# Backup logs
aws s3 cp /home/sagemaker-user/exp1a_training.log \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/logs/ \
  --region us-west-2

aws s3 cp /home/sagemaker-user/exp1b_training.log \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/logs/ \
  --region us-west-2

aws s3 cp /home/sagemaker-user/finetune.log \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/logs/ \
  --region us-west-2
```

**Advantages:**
- Simpler (one command)
- Skips already uploaded files automatically
- Shows progress

**Disadvantages:**
- Less organized output
- No error tracking
- No file counting

---

## Advanced: Organize by Experiment

If you want better organization, use this structure:

```bash
# Upload Exp1A checkpoints
aws s3 sync /home/sagemaker-user/checkpoints/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/exp1a/ \
  --exclude "*" \
  --include "Exp1A_*" \
  --region us-west-2

# Upload Exp1B checkpoints
aws s3 sync /home/sagemaker-user/checkpoints/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/exp1b/ \
  --exclude "*" \
  --include "Exp1B_*" \
  --region us-west-2

# Upload Exp2 results
aws s3 sync /home/sagemaker-user/exp2_finetune_results/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/exp2/ \
  --region us-west-2
```

**Result:**
```
s3://snowml-model-ready/checkpoints/simran_thesis/
├── exp1a/
│   ├── Exp1A_Base_1e-3_epoch*.pth
│   ├── Exp1A_Base_3e-4_epoch*.pth
│   └── ... (all 240 Exp1A files)
├── exp1b/
│   ├── Exp1B_Base_1e-3_epoch*.pth
│   ├── Exp1B_Base_3e-4_epoch*.pth
│   └── ... (all 240 Exp1B files)
└── exp2/
    ├── FineTune_Base_epoch*.pth
    └── finetune_summary.json
```

---

## Backup Training Logs and Results

### Backup logs:
```bash
aws s3 cp /home/sagemaker-user/exp1a_training.log \
  s3://snowml-model-ready/checkpoints/simran_thesis/logs/ \
  --region us-west-2

aws s3 cp /home/sagemaker-user/exp1b_training.log \
  s3://snowml-model-ready/checkpoints/simran_thesis/logs/ \
  --region us-west-2

aws s3 cp /home/sagemaker-user/finetune.log \
  s3://snowml-model-ready/checkpoints/simran_thesis/logs/ \
  --region us-west-2
```

### Backup evaluation results:
```bash
aws s3 sync /home/sagemaker-user/evaluation_results/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/evaluation_results/ \
  --region us-west-2
```

---

## Clean Up SageMaker After Backup

**Once backup is verified, you can delete local checkpoints to save space:**

```bash
# ⚠️ ONLY DO THIS AFTER VERIFYING S3 BACKUP!

# Check S3 backup first
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive | wc -l
# Should show 492 files (or your total)

# Then delete local checkpoints
rm -rf /home/sagemaker-user/checkpoints/*
rm -rf /home/sagemaker-user/exp2_finetune_results/*.pth

# Keep metadata files
# Keep training logs
```

**This frees up ~50-100 GB on SageMaker EFS, saving ~$15-30/month!**

---

## Move to Glacier for Long-Term Storage

**After thesis defense, move to Glacier Deep Archive:**

```bash
# Move to Glacier (cheapest long-term storage)
aws s3 cp s3://snowml-model-ready/checkpoints/simran_thesis/ \
  s3://snowml-model-ready/checkpoints/simran_thesis_glacier/ \
  --recursive \
  --storage-class DEEP_ARCHIVE \
  --region us-west-2

# Then delete from standard tier
aws s3 rm s3://snowml-model-ready/checkpoints/simran_thesis/ --recursive
```

**Cost savings:**
- Before: $2.30/month (S3 Standard for 100 GB)
- After: $0.40/month (Glacier Deep Archive for 100 GB)
- **Saves $1.90/month ($23/year)**

**Note:** Glacier retrieval takes 12-48 hours, so only use for archival.

---

## Share Checkpoints with Professor/Collaborators

### Generate presigned URL (temporary access):
```bash
# Create download link valid for 7 days
aws s3 presign s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/Exp1B_Base_3e-4_epoch14.pth \
  --expires-in 604800 \
  --region us-west-2
```

**Returns URL like:**
```
https://snowml-model-ready.s3.us-west-2.amazonaws.com/checkpoints/simran_thesis/20260930/checkpoints/Exp1B_Base_3e-4_epoch14.pth?X-Amz-Algorithm=...
```

Send this URL to professor/collaborators. No AWS account needed to download.

---

## Troubleshooting

### Error: "Access Denied"
**Problem:** No S3 write permissions

**Fix:**
```bash
# Check your IAM permissions
aws sts get-caller-identity

# Ask professor to add S3 write permissions to your IAM user
# Or use different bucket you have access to
```

### Error: "Bucket does not exist"
**Problem:** Bucket name wrong or doesn't exist

**Fix:**
```bash
# List available buckets
aws s3 ls

# Use correct bucket name in script
S3_BUCKET = "your-actual-bucket-name"
```

### Upload is very slow
**Problem:** Large files, slow connection

**Fix:**
```bash
# Use multipart upload for large files (automatic in script)
# Or run in screen/tmux to keep running if disconnected

screen -S s3_backup
python backup_checkpoints_to_s3.py
# Ctrl+A, D to detach
# screen -r s3_backup to reattach
```

### Upload interrupted
**Problem:** Connection lost mid-upload

**Fix:**
```bash
# Run aws s3 sync again - it will skip already uploaded files
aws s3 sync /home/sagemaker-user/checkpoints/ \
  s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/ \
  --region us-west-2
```

---

## Best Practices

1. **Backup immediately after training completes**
   - Don't wait - accidents happen

2. **Verify backup before deleting local files**
   - Check file count matches
   - Spot-check a few files can be downloaded

3. **Use descriptive folder names**
   - Include date: `20260930`
   - Include student name: `simran_thesis`
   - Include experiment: `exp1a`, `exp1b`, `exp2`

4. **Keep metadata files**
   - JSON summaries
   - Training logs
   - README files

5. **Document S3 locations in thesis**
   - Include S3 paths in reproducibility section
   - Make it easy for reviewers to access

6. **Set lifecycle policies**
   - Auto-transition to Glacier after 90 days
   - Auto-delete after 7 years (after graduation)

---

## Cost Estimate

**Assumptions:**
- 480 checkpoints (.pth files) ~52 GB
- 16 Exp1A results (.csv files) ~45 MB
- 10 Exp1B results (.csv files) ~29 MB
- 12 Exp2 results (.pth + .json) ~1.1 GB
- 8 evaluation results (.csv) ~15 MB
- 4 training logs ~11 MB
- **Total: ~53.2 GB (526 files)**

**Storage Costs:**
| Period | Storage Type | Monthly Cost | Annual Cost |
|--------|--------------|--------------|-------------|
| During thesis (6 months) | S3 Standard | $1.24 | $14.88 |
| After thesis (5 years) | Glacier Deep Archive | $0.22 | $2.64 |
| **Total 5.5 years** | | | **$28** |

**Compared to keeping on SageMaker EFS:**
- SageMaker: $16.20/month × 66 months = **$1,069**
- **Savings: $1,041** 💰

---

## Summary

✅ **Run this command on SageMaker:**
```bash
python backup_checkpoints_to_s3.py
```

✅ **Verify backup:**
```bash
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive --summarize
```

✅ **Result:**
- All checkpoints safely stored in S3
- Can delete from SageMaker to save money
- Can download anytime
- Can share with collaborators
- Cost: ~$1.24/month vs $16/month on SageMaker

---

**Questions? Check AWS S3 documentation or ask professor for bucket access.**
