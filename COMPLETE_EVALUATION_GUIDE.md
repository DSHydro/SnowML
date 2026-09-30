# Complete Evaluation Guide - Start to Finish

**Date:** August 30, 2026  
**Goal:** Evaluate Wind_3e-4 and Humidity_3e-4 on Test Sets A & B using original Exp3 methodology

**Estimated Total Time:** 2 hours (mostly unattended)  
**Estimated Cost:** $0.60-1.00

---

## 📋 WHAT YOU'LL DO

1. **Prepare files on laptop** (10 minutes)
2. **Start AWS SageMaker Studio** (5 minutes)
3. **Upload files to SageMaker** (10 minutes)
4. **Run evaluation** (60-90 minutes - unattended)
5. **Download results** (5 minutes)
6. **Stop SageMaker** (2 minutes - CRITICAL!)

---

## PART 1: PREPARE FILES ON LAPTOP (10 minutes)

### Step 1.1: Extract Checkpoints

```bash
cd /Users/simran/Desktop/SnowML

# Extract checkpoint files (creates 'checkpoints/' folder)
tar -xzf exp1b_final_checkpoints.tar.gz

# Verify extraction
ls -lh checkpoints/
# Should see 60 .pth files (2 models × 30 epochs)

# Count files
ls checkpoints/*.pth | wc -l
# Should show: 60

# Check the best epoch files exist
ls -lh checkpoints/Exp1B_Wind_3e-4_epoch6.pth
ls -lh checkpoints/Exp1B_Humidity_3e-4_epoch7.pth
```

**Expected output:**
```
checkpoints/Exp1B_Wind_3e-4_epoch6.pth     (~219 KB)
checkpoints/Exp1B_Humidity_3e-4_epoch7.pth (~219 KB)
```

✅ **You now have a `checkpoints/` folder with 60 .pth files!**

### Step 1.2: Verify Test Sets

```bash
# Check test sets folder
ls -lh correct_test_sets/

# Should see:
# test_a_hucs.txt (55 HUCs)
# test_b_hucs.txt (81 HUCs)
# train_hucs.txt (162 HUCs - reference)
# val_hucs.txt (54 HUCs - reference)

# Verify counts
wc -l correct_test_sets/*.txt
```

**Expected output:**
```
  55 correct_test_sets/test_a_hucs.txt
  81 correct_test_sets/test_b_hucs.txt
 162 correct_test_sets/train_hucs.txt
  54 correct_test_sets/val_hucs.txt
```

### Step 1.3: Check Evaluation Script

```bash
# Verify evaluation script exists
ls -lh evaluate_from_checkpoints.py

# Should be ~300 lines, ~10-12 KB
```

**You should now have:**
- ✅ `checkpoints/` folder (extracted, 60 .pth files)
- ✅ `correct_test_sets/` folder (4 .txt files)
- ✅ `evaluate_from_checkpoints.py` script

---

## PART 2: START AWS SAGEMAKER STUDIO (5 minutes)

### Option A: Using AWS Console (Recommended - Easier)

1. **Open AWS Console:**
   - Go to https://console.aws.amazon.com/
   - Login with your credentials

2. **Navigate to SageMaker:**
   - Search for "SageMaker" in the search bar
   - Click "Amazon SageMaker"

3. **Open Studio:**
   - In left sidebar, click "Studio"
   - Find your domain: `d-spd2senoxi9j`
   - Find your space: `snowML-ConvLSTM-khaja`
   - Click "Open Studio"

4. **Wait for Studio to start:**
   - First time: ~3-5 minutes
   - Subsequent times: ~1-2 minutes
   - You'll see "JupyterLab" interface open in browser

### Option B: Using AWS CLI

```bash
# On your laptop terminal
aws sagemaker create-app \
  --domain-id d-spd2senoxi9j \
  --space-name snowML-ConvLSTM-khaja \
  --app-type JupyterLab \
  --app-name default \
  --region us-west-2

# Wait 2-3 minutes, then check status
aws sagemaker list-apps \
  --domain-id d-spd2senoxi9j \
  --region us-west-2

# Look for Status: "InService"
```

**When Studio is ready:**
- You'll see JupyterLab interface
- File browser on left
- Terminal and notebook options

---

## PART 3: UPLOAD FILES TO SAGEMAKER (10 minutes)

### Step 3.1: Open Terminal in Studio

In SageMaker Studio JupyterLab:
1. Click "File" → "New" → "Terminal"
2. You should see a terminal prompt: `sagemaker-user@...`

### Step 3.2: Check Existing Files

```bash
# In Studio terminal
cd /home/sagemaker-user

# Check if src/ folder exists (from training)
ls -lh src/

# Should see src/snowML/ folder
ls -lh src/snowML/LSTM/
ls -lh src/snowML/Scripts/
ls -lh src/snowML/datapipe/huc_lists/

# Check if hucs_data.json exists (this is what we used for training)
ls -lh src/snowML/datapipe/huc_lists/hucs_data.json
```

**If src/ folder doesn't exist:**
You'll need to upload it or extract from your training package. But you mentioned you already have it, so it should be there!

### Step 3.3: Create Directories

```bash
# In Studio terminal
cd /home/sagemaker-user

# Create directories for evaluation
mkdir -p evaluation_files
mkdir -p checkpoints_best
mkdir -p test_sets
```

### Step 3.4: Upload Files via Studio UI

**Method 1: Drag & Drop (Easiest)**

In Studio file browser (left sidebar):

1. **Upload checkpoints (2 files):**
   - Navigate to `checkpoints_best/` folder in Studio
   - Drag from Finder/Explorer on your laptop:
     - `checkpoints/Exp1B_Wind_3e-4_epoch6.pth`
     - `checkpoints/Exp1B_Humidity_3e-4_epoch7.pth`
   - Wait for upload (~10-20 seconds for ~440 KB total - very small!)

2. **Upload test sets (2 files):**
   - Navigate to `test_sets/` folder
   - Drag from Finder/Explorer:
     - `correct_test_sets/test_a_hucs.txt`
     - `correct_test_sets/test_b_hucs.txt`
   - Upload is instant (small files)

3. **Upload evaluation script:**
   - Navigate to `evaluation_files/` folder
   - Drag: `evaluate_from_checkpoints.py`
   - Upload is instant

**Method 2: Upload via Terminal (Alternative)**

If drag & drop doesn't work, use the upload button:
1. Click the upload icon (↑) in file browser
2. Select files one by one
3. Wait for uploads to complete

### Step 3.5: Verify Uploads

```bash
# In Studio terminal
cd /home/sagemaker-user

# Check checkpoints
ls -lh checkpoints_best/
# Should show 2 .pth files (~219 KB each)

# Check test sets
ls -lh test_sets/
# Should show 2 .txt files

# Check evaluation script
ls -lh evaluation_files/evaluate_from_checkpoints.py

# Count HUCs in test sets
wc -l test_sets/*.txt
# Should show: 55 and 81
```

---

## PART 4: RUN EVALUATION (60-90 minutes - Unattended)

### Step 4.1: Update Evaluation Script Paths

```bash
# In Studio terminal
cd /home/sagemaker-user/evaluation_files

# Edit the evaluation script
nano evaluate_from_checkpoints.py
```

**Find lines 20-29 and update paths:**

```python
# CHANGE FROM:
CHECKPOINTS = {
    "Wind_3e-4": {
        "file": "/home/sagemaker-user/exp1b_final_checkpoints/Exp1B_Wind_3e-4_epoch6.pth",
        "val_kge": 0.8104
    },
    "Humidity_3e-4": {
        "file": "/home/sagemaker-user/exp1b_final_checkpoints/Exp1B_Humidity_3e-4_epoch7.pth",
        "val_kge": 0.6212
    }
}

# CHANGE TO:
CHECKPOINTS = {
    "Wind_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Wind_3e-4_epoch6.pth",
        "val_kge": 0.8104
    },
    "Humidity_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Humidity_3e-4_epoch7.pth",
        "val_kge": 0.6212
    }
}
```

**Find lines 32-36 and update paths:**

```python
# CHANGE FROM:
TEST_SETS = {
    "Test_A": "/home/sagemaker-user/data/exp1b_test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/data/exp1b_test_b_hucs.txt"
}

# CHANGE TO:
TEST_SETS = {
    "Test_A": "/home/sagemaker-user/test_sets/test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/test_sets/test_b_hucs.txt"
}
```

**Save and exit:** Ctrl+O, Enter, Ctrl+X

### Step 4.2: Test the Script First (Quick Check)

```bash
cd /home/sagemaker-user/evaluation_files

# Check Python and imports
python3 -c "import sys; print(sys.path)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Should show:
# PyTorch: 2.5.1 or similar
# CUDA: True (if GPU available)
```

### Step 4.3: Run Evaluation (Background Mode)

```bash
cd /home/sagemaker-user/evaluation_files

# Run in background (won't stop if terminal closes)
nohup python3 evaluate_from_checkpoints.py > evaluation.log 2>&1 &

# Save the process ID
echo $! > eval.pid

# Wait a few seconds
sleep 5

# Check if it started successfully
tail -30 evaluation.log
```

**You should see:**
```
================================================================================
EXPERIMENT 1B - TEST SET EVALUATION
Using Original Exp3 Evaluation Methodology
================================================================================
Started: 2026-08-30 XX:XX:XX

📋 Loading test sets...

  Test_A: 55 HUCs
  Test_B: 81 HUCs

================================================================================
MODEL: Wind_3e-4
================================================================================
Loading checkpoint: /home/sagemaker-user/checkpoints_best/Exp1B_Wind_3e-4_epoch6.pth
  Features: ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_vs']
  Train HUCs: 162 HUCs
  Val HUCs: 54 HUCs
  Learning rate: 0.0003
  Train dimension: huc
```

**If you see this, it's working!** You can close the browser - it will keep running.

### Step 4.4: Monitor Progress (Optional)

**To check progress later:**

```bash
cd /home/sagemaker-user/evaluation_files

# Watch live output
tail -f evaluation.log

# Or check last 50 lines
tail -50 evaluation.log

# Check if still running
ps aux | grep evaluate_from_checkpoints
```

**Progress indicators you'll see:**
```
================================================================================
Evaluating on: Test_A (55 HUCs)
================================================================================

✅ Using HUC-based normalization (same global stats as training)
✅ Loaded data for 55 HUCs

  Progress: 10/55 HUCs evaluated
  Progress: 20/55 HUCs evaluated
  Progress: 30/55 HUCs evaluated
  Progress: 40/55 HUCs evaluated
  Progress: 50/55 HUCs evaluated

✅ Completed: 52/55 HUCs

📊 Summary Statistics:
   Median KGE: 0.XXXX
   Mean KGE: 0.XXXX
   ...
```

**Expected Timeline:**
- Wind_3e-4 on Test A: ~15-20 minutes
- Wind_3e-4 on Test B: ~20-25 minutes
- Humidity_3e-4 on Test A: ~15-20 minutes
- Humidity_3e-4 on Test B: ~20-25 minutes
- **Total: 70-90 minutes**

### Step 4.5: Check When Complete

```bash
# Check end of log
tail -100 evaluation.log

# Look for this:
# ================================================================================
# ✅ EVALUATION COMPLETE!
# Finished: 2026-08-30 XX:XX:XX
# ================================================================================
```

---

## PART 5: DOWNLOAD RESULTS (5 minutes)

### Step 5.1: Check Results Files

```bash
cd /home/sagemaker-user

# List results
ls -lh exp1b_evaluation_results/

# Should see 5 CSV files:
# - Wind_3e-4_Test_A_metrics.csv
# - Wind_3e-4_Test_B_metrics.csv
# - Humidity_3e-4_Test_A_metrics.csv
# - Humidity_3e-4_Test_B_metrics.csv
# - validation_vs_test_comparison.csv ← MOST IMPORTANT
```

### Step 5.2: View Summary (Quick Look)

```bash
# View the comparison summary
cat exp1b_evaluation_results/validation_vs_test_comparison.csv

# Should show something like:
# Model,Validation_KGE,Test_A_KGE_median,Test_A_KGE_mean,Test_A_n_hucs,Test_A_drop_pct,Test_B_KGE_median,Test_B_KGE_mean,Test_B_n_hucs,Test_B_drop_pct
# Wind_3e-4,0.8104,0.XXXX,0.XXXX,52,XX.X,0.XXXX,0.XXXX,78,XX.X
# Humidity_3e-4,0.6212,0.XXXX,0.XXXX,52,XX.X,0.XXXX,0.XXXX,78,XX.X
```

### Step 5.3: Download via Studio UI

**Method 1: File Browser (Easiest)**

1. In Studio file browser, navigate to `exp1b_evaluation_results/`
2. Right-click on the folder → "Download"
3. Or download individual CSV files one by one
4. Save to `/Users/simran/Desktop/SnowML/results/exp1b_evaluation/`

**Method 2: Via S3 (Recommended for Backup)**

```bash
# In Studio terminal - upload to S3 first
aws s3 sync /home/sagemaker-user/exp1b_evaluation_results/ \
  s3://snowml-results/simran/exp1b_eval_aug30/

# On your laptop - download from S3
cd /Users/simran/Desktop/SnowML
mkdir -p results/exp1b_evaluation

aws s3 sync s3://snowml-results/simran/exp1b_eval_aug30/ \
  results/exp1b_evaluation/

# Check downloaded files
ls -lh results/exp1b_evaluation/
```

---

## PART 6: STOP SAGEMAKER (2 MINUTES - CRITICAL!)

**⚠️ CRITICAL: You MUST stop Studio to avoid charges ($0.53/hour)**

### Option A: AWS Console (Recommended)

1. Go to AWS Console → SageMaker → Studio
2. Find your domain: `d-spd2senoxi9j`
3. Find your space: `snowML-ConvLSTM-khaja`
4. Find the running app (JupyterLab)
5. Click "Delete" or "Stop"
6. Confirm deletion

### Option B: AWS CLI

```bash
# On your laptop terminal (NOT Studio terminal!)
aws sagemaker delete-app \
  --domain-id d-spd2senoxi9j \
  --space-name snowML-ConvLSTM-khaja \
  --app-type JupyterLab \
  --app-name default \
  --region us-west-2

# Verify it's stopped
aws sagemaker list-apps \
  --domain-id d-spd2senoxi9j \
  --region us-west-2

# Should show Status: "Deleted" or no apps listed
```

---

## PART 7: ANALYZE RESULTS (On Laptop)

### Step 7.1: View Comparison Table

```bash
cd /Users/simran/Desktop/SnowML/results/exp1b_evaluation

# View the summary
cat validation_vs_test_comparison.csv

# Or open in Excel/Numbers for better viewing
open validation_vs_test_comparison.csv
```

### Step 7.2: Interpret Results

**EXCELLENT Generalization (✅ Success):**
```
Wind_3e-4:
  Validation KGE: 0.8104
  Test A KGE: 0.75-0.80 (7-15% drop)
  Test B KGE: 0.70-0.75 (12-20% drop)
```

**GOOD Generalization (✅ Acceptable):**
```
Wind_3e-4:
  Validation KGE: 0.8104
  Test A KGE: 0.65-0.75 (15-25% drop)
  Test B KGE: 0.60-0.70 (20-30% drop)
```

**POOR Generalization (❌ Problem):**
```
Wind_3e-4:
  Validation KGE: 0.8104
  Test A KGE: <0.60 (>30% drop)
  Test B KGE: <0.55 (>35% drop)
```

### Step 7.3: Compare to Original Exp3

**Original Exp3 (from main branch):**
- Validation KGE: ~0.82
- Test Set A KGE: ~0.72
- Drop: ~12% (good generalization)

**Your Wind_3e-4:**
- Validation KGE: 0.8104 (matches Exp3!)
- Test Set A KGE: [YOUR RESULT]
- Drop: [YOUR DROP %]

**Interpretation:**
- If drop < 15%: ✅ **Matches or beats Exp3**
- If drop 15-25%: ✅ **Similar to Exp3, acceptable**
- If drop > 25%: ⚠️ **Investigate why**

### Step 7.4: Create Summary for Professor

```bash
cd /Users/simran/Desktop/SnowML/results/exp1b_evaluation

# Create a simple summary
cat > EVALUATION_SUMMARY.txt << 'EOF'
Experiment 1B Evaluation Results
Date: August 30, 2026
Models: Wind_3e-4 (epoch 6), Humidity_3e-4 (epoch 7)

WIND_3E-4 RESULTS:
  Validation KGE: 0.8104 (54 HUCs)
  Test Set A KGE: [FROM CSV] (55 HUCs)
  Test Set B KGE: [FROM CSV] (81 HUCs)
  
  Generalization Drop:
    Test A: [XX]%
    Test B: [XX]%
  
  Status: [EXCELLENT/GOOD/POOR]

HUMIDITY_3E-4 RESULTS:
  Validation KGE: 0.6212 (54 HUCs)
  Test Set A KGE: [FROM CSV] (55 HUCs)
  Test Set B KGE: [FROM CSV] (81 HUCs)
  
  Generalization Drop:
    Test A: [XX]%
    Test B: [XX]%
  
  Status: [CONSISTENT/UNDERPERFORMING]

COMPARISON TO ORIGINAL EXP3:
  Original: Val 0.82 → Test 0.72 (12% drop)
  Wind_3e-4: Val 0.81 → Test [XX] ([XX]% drop)
  
  Conclusion: [MATCHES/BEATS/WORSE than Exp3]

METHODOLOGY VERIFICATION:
  ✅ Used correct test HUCs (55 from hucs_data.json)
  ✅ Used original LSTM_evaluate.py functions
  ✅ Same normalization as training (HUC-based, global stats)
  ✅ Same metrics calculation
  
FILES GENERATED:
  - Wind_3e-4_Test_A_metrics.csv
  - Wind_3e-4_Test_B_metrics.csv
  - Humidity_3e-4_Test_A_metrics.csv
  - Humidity_3e-4_Test_B_metrics.csv
  - validation_vs_test_comparison.csv
EOF

# Fill in the [FROM CSV] values from the actual results
nano EVALUATION_SUMMARY.txt
```

---

## 📊 QUICK REFERENCE COMMANDS

### Check if Studio is Running:
```bash
aws sagemaker list-apps --domain-id d-spd2senoxi9j --region us-west-2
```

### Monitor Evaluation Progress:
```bash
# In Studio terminal
tail -f /home/sagemaker-user/evaluation_files/evaluation.log
```

### Check Results:
```bash
# In Studio terminal
ls -lh /home/sagemaker-user/exp1b_evaluation_results/
cat /home/sagemaker-user/exp1b_evaluation_results/validation_vs_test_comparison.csv
```

### Download from S3:
```bash
# On laptop
aws s3 sync s3://snowml-results/simran/exp1b_eval_aug30/ results/exp1b_evaluation/
```

### Stop Studio:
```bash
aws sagemaker delete-app \
  --domain-id d-spd2senoxi9j \
  --space-name snowML-ConvLSTM-khaja \
  --app-type JupyterLab \
  --app-name default \
  --region us-west-2
```

---

## 🐛 TROUBLESHOOTING

### "Module not found: snowML"
```bash
# Check if src/ exists
ls -lh /home/sagemaker-user/src/

# If not, upload from training or extract package
```

### "Checkpoint file not found"
```bash
# Check uploaded files
ls -lh /home/sagemaker-user/checkpoints_best/

# Re-upload if missing
```

### "S3 access denied"
```bash
# Check credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://snowml-model-ready/pnw_swe_data/ | head -5
```

### Evaluation running too slow
```bash
# Check if using GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# If False: using CPU (slower but okay, ~2 hours total)
# If True: using GPU (faster, ~1 hour total)
```

### Can't download results
```bash
# Upload to S3 first, then download to laptop
# See "Download via S3" section above
```

---

## ✅ COMPLETE CHECKLIST

### Before Starting:
- [ ] Extracted `exp1b_final_checkpoints/` on laptop
- [ ] Have `correct_test_sets/` folder ready
- [ ] Have `evaluate_from_checkpoints.py` ready
- [ ] AWS credentials configured

### On SageMaker:
- [ ] Started Studio
- [ ] Uploaded 2 checkpoint files
- [ ] Uploaded 2 test set files
- [ ] Uploaded evaluation script
- [ ] Updated script paths
- [ ] Verified src/ folder exists
- [ ] Started evaluation with nohup

### During Evaluation:
- [ ] Checked log shows it started successfully
- [ ] Can close browser (running in background)
- [ ] Check progress occasionally if desired

### After Completion:
- [ ] Checked evaluation.log shows "COMPLETE"
- [ ] Downloaded all 5 CSV files
- [ ] Backed up to S3
- [ ] **STOPPED Studio app** ✅ CRITICAL
- [ ] Verified app is deleted (AWS console or CLI)

### Analysis:
- [ ] Viewed validation_vs_test_comparison.csv
- [ ] Calculated drop percentages
- [ ] Compared to original Exp3
- [ ] Created summary for professor

---

## 💰 COST ESTIMATE

**SageMaker Studio:**
- Instance: ml.g4dn.xlarge @ $0.53/hour
- Evaluation time: 1-1.5 hours
- Cost: **$0.53-0.80**

**S3 Storage:**
- Results files: <10 MB
- Cost: **~$0.00** (negligible)

**Total: $0.60-1.00**

---

## ⏱️ TIME ESTIMATE

**Your Active Time:**
- Prepare files: 10 min
- Start Studio: 5 min
- Upload files: 10 min
- Run evaluation: 5 min
- Download results: 5 min
- Stop Studio: 2 min
- **Total: ~35-40 minutes**

**Wall Clock Time:**
- Evaluation running: 70-90 min (unattended)
- **Total: ~2 hours**

---

**Last Updated:** August 30, 2026  
**Ready to Start:** YES  
**Next:** Follow Part 1 to begin!
