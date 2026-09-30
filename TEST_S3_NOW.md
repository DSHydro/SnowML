# Test S3 Access on SageMaker - Quick Guide

**Date:** July 17, 2026  
**Status:** Professor approved S3 permissions - ready to test!  
**Goal:** Verify automatic S3 data loading works in SageMaker

---

## What Changed

✅ **Professor fixed S3 permissions** on your SageMaker role  
✅ **Created new notebook:** `Pilot_Test_S3_Fixed.ipynb`  
✅ **Fixed file format:** Now uses CSV (not parquet)  
✅ **Fixed naming:** Uses `model_ready_huc{ID}.csv` format

---

## Quick Start (10 Steps)

### 1. Start SageMaker Notebook (from your laptop)

```bash
cd /Users/simran/Desktop/SnowML

# Start the notebook instance
aws sagemaker start-notebook-instance \
  --notebook-instance-name st-gnn-T4x1 \
  --region us-west-2

echo "⏳ Waiting for notebook to start (takes ~2 minutes)..."
```

### 2. Wait for Instance to Start

```bash
# Check status (run this every 30 seconds)
aws sagemaker describe-notebook-instance \
  --notebook-instance-name st-gnn-T4x1 \
  --region us-west-2 \
  --query 'NotebookInstanceStatus' \
  --output text

# Wait until it says: "InService"
```

### 3. Get Access URL

```bash
# Get presigned URL (valid for 5 minutes)
aws sagemaker create-presigned-notebook-instance-url \
  --notebook-instance-name st-gnn-T4x1 \
  --region us-west-2 \
  --query 'AuthorizedUrl' \
  --output text
```

**Copy the URL and paste it in your browser**

---

### 4. Upload New Notebook

In the Jupyter interface:

1. Click **"Upload"** button (top right)
2. Select: `Pilot_Test_S3_Fixed.ipynb`
3. Click **"Upload"** again to confirm
4. Click on the notebook to open it

---

### 5. Select Kernel

When the notebook opens:
- Click **Kernel → Change Kernel**
- Select: **`conda_pytorch_p310`**
- Wait for "Kernel Ready" indicator

---

### 6. Run Cells in Order

**Cell 1: Check GPU**
- Run it (Shift+Enter)
- Should show: "GPU name: Tesla T4"

**Cell 2: Install SnowML**
- Run it (takes ~2 minutes)
- Installs dependencies

**Cell 3: Import Libraries**
- Run it
- Should show: "✅ All libraries imported!"

**Cell 4: 🔥 TEST S3 ACCESS 🔥**
- **This is the critical test!**
- Run it
- **Expected output:**
  ```
  Testing S3 access...
  
  [Test 1] Listing bucket contents...
     ✅ SUCCESS! Found files in bucket
        - model_ready_huc170103040105.csv
        ...
  
  [Test 2] Reading test file...
     ✅ SUCCESS! Loaded file
     📊 Shape: (3652, 13)
     📅 Columns: ['day', 'mean_pr', 'mean_tair', ...]
  
  ================================================================
  ✅ S3 PERMISSIONS WORKING!
  ================================================================
  ```

**If Cell 4 FAILS:**
- S3 permissions not applied yet
- Screenshot the error
- Contact professor with the error message
- **STOP HERE** - no point continuing

**If Cell 4 SUCCEEDS:**
- 🎉 Permissions working!
- Continue to next cells

---

### 7. Continue Testing (Cells 5-7)

**Cell 5: Configuration**
- Run it
- Sets up training parameters

**Cell 6: Define HUCs**
- Run it
- Lists the 5 HUCs for pilot test

**Cell 7: 🔥 LOAD DATA FROM S3 🔥**
- **This tests the full automatic pipeline!**
- Run it (takes ~1-2 minutes)
- **Expected output:**
  ```
  LOADING DATA FROM S3 - Testing Automatic Pipeline
  
  number of sub units for training is 5
  
  ================================================================
  ✅ DATA LOADED SUCCESSFULLY FROM S3!
  ================================================================
  
  📊 Loaded 5 HUCs
  📈 Normalization stats computed:
     Means: {...}
     Stds: {...}
  ```

**If Cell 7 succeeds:**
- **🎉 YOU'RE DONE!**
- S3 access fully working!
- Ready for full experiments!

---

### 8. Optional: Run Full Training (Cells 8-9)

If you want to test the complete pipeline:
- Cell 8: Initialize model (~5 seconds)
- Cell 9: Train for 5 epochs (~10 minutes)

**But if Cell 7 works, you already proved S3 access!**

---

### 9. CRITICAL: Stop the Notebook Instance

**From your laptop terminal:**

```bash
# Stop the instance to avoid charges!
aws sagemaker stop-notebook-instance \
  --notebook-instance-name st-gnn-T4x1 \
  --region us-west-2

# Verify it stopped
aws sagemaker describe-notebook-instance \
  --notebook-instance-name st-gnn-T4x1 \
  --region us-west-2 \
  --query 'NotebookInstanceStatus' \
  --output text

# Should show: "Stopping" or "Stopped"
```

**Cost:** ml.g4dn.xlarge = **$0.53/hour**  
If you forget to stop: **$12.72/day!** ⚠️

---

### 10. Report Back

Take screenshots of:
- ✅ Cell 4 output (S3 test)
- ✅ Cell 7 output (data loading)

---

## What Each Cell Tests

| Cell | What It Tests | Critical? |
|------|---------------|-----------|
| 1 | GPU access | ✅ Yes |
| 2 | Package installation | ✅ Yes |
| 3 | Imports | ✅ Yes |
| 4 | **S3 bucket access** | 🔥 **MOST CRITICAL** |
| 5 | Config | Optional |
| 6 | HUC lists | Optional |
| 7 | **Automatic S3 data loading** | 🔥 **CRITICAL** |
| 8 | Model init | Optional |
| 9 | Training | Optional |

**Minimum to prove S3 works:** Cells 1-4, 7

---

## Expected Timeline

| Task | Time |
|------|------|
| Start instance | 2 min |
| Upload notebook | 30 sec |
| Cell 1-3 (setup) | 3 min |
| Cell 4 (S3 test) | 30 sec |
| Cell 5-6 (config) | 10 sec |
| Cell 7 (load data) | 1-2 min |
| **TOTAL (minimum test)** | **~7 minutes** |
| Optional: Cells 8-9 (train) | +10 min |

---

## Troubleshooting

### Problem: Cell 4 fails with "AccessDenied" or "403"

**Cause:** S3 permissions not applied yet

**Solution:**
1. Screenshot the exact error
2. Check which role the notebook is using:
   ```bash
   aws sagemaker describe-notebook-instance \
     --notebook-instance-name st-gnn-T4x1 \
     --region us-west-2 \
     --query 'RoleArn'
   ```
3. Send the role ARN to professor
4. Ask: "Can you verify this role has s3:GetObject permission for snowml-model-ready bucket?"

---

### Problem: Cell 4 succeeds but Cell 7 fails

**Cause:** Different issue (likely code/data format)

**Look for:**
- Missing HUC files
- Wrong column names
- Date parsing errors

**Solution:**
- Read the error message carefully
- Check if specific HUC IDs exist in S3
- May need to adjust HUC list in Cell 6

---

### Problem: "Kernel not ready"

**Cause:** Wrong kernel selected

**Solution:**
- Click **Kernel → Change Kernel**
- Select: `conda_pytorch_p310`
- Wait for green "Ready" indicator

---

### Problem: Can't get presigned URL

**Cause:** URL expired (5 min timeout)

**Solution:**
```bash
# Just run the command again
aws sagemaker create-presigned-notebook-instance-url \
  --notebook-instance-name st-gnn-T4x1 \
  --region us-west-2 \
  --query 'AuthorizedUrl' \
  --output text
```

---

## Success Checklist

- [ ] Started notebook instance
- [ ] Got presigned URL and opened browser
- [ ] Uploaded `Pilot_Test_S3_Fixed.ipynb`
- [ ] Selected `conda_pytorch_p310` kernel
- [ ] Cell 1: GPU detected ✅
- [ ] Cell 2: Packages installed ✅
- [ ] Cell 3: Imports successful ✅
- [ ] Cell 4: **S3 ACCESS WORKING** ✅ 🎉
- [ ] Cell 7: **DATA LOADED FROM S3** ✅ 🎉
- [ ] **STOPPED INSTANCE** ✅ (most important!)

---

## If Everything Works

**You'll see:**
```
================================================================
✅ DATA LOADED SUCCESSFULLY FROM S3!
================================================================
```

**This means:**
1. ✅ S3 permissions fixed
2. ✅ Automatic data loading works
3. ✅ Ready for full experiments
4. ✅ No more manual file downloads!

**Next step:**
- Create full training notebook for Experiment 1B
- Use same S3 loading approach
- Scale to 231 training HUCs

---

## Quick Commands Summary

```bash
# 1. Start notebook
aws sagemaker start-notebook-instance --notebook-instance-name st-gnn-T4x1 --region us-west-2

# 2. Check status
aws sagemaker describe-notebook-instance --notebook-instance-name st-gnn-T4x1 --region us-west-2 --query 'NotebookInstanceStatus'

# 3. Get URL
aws sagemaker create-presigned-notebook-instance-url --notebook-instance-name st-gnn-T4x1 --region us-west-2 --query 'AuthorizedUrl' --output text

# 4. After testing - STOP IT!
aws sagemaker stop-notebook-instance --notebook-instance-name st-gnn-T4x1 --region us-west-2
```

---

**Remember:** The S3 test (Cell 4 & 7) is all you really need to verify permissions! The rest is optional.

**Good luck! 🚀**
