# AWS Evaluation Instructions - All 6 Variations

**Date:** September 3, 2026  
**Goal:** Evaluate all 6 variations on Test Sets A & B using AWS SageMaker

---

## 📦 FILES READY TO UPLOAD TO AWS

### 1. **Checkpoints Archive (20 MB)**
- **File:** `exp1b_all_6_variations.tar.gz`
- **Contains:** 178 checkpoint files for all 6 variations
  - Base_1e-3: 30 epochs
  - Base_3e-4: 30 epochs
  - Srad_1e-3: 30 epochs
  - Srad_3e-4: 28 epochs
  - Wind_3e-4: 30 epochs
  - Humidity_3e-4: 30 epochs

### 2. **Evaluation Package (7.3 KB)**
- **File:** `exp1b_evaluation_package.tar.gz`
- **Contains:**
  - `evaluate_all_6_variations.py` - Evaluation script
  - `correct_test_sets/` folder with:
    - test_a_hucs.txt (55 HUCs)
    - test_b_hucs.txt (81 HUCs)
    - train_hucs.txt (162 HUCs for normalization)
    - val_hucs.txt (54 HUCs for normalization)

---

## 🚀 STEP-BY-STEP INSTRUCTIONS

### **Step 1: Start SageMaker Studio (5 minutes)**

```bash
# Option A: Use AWS Console
# 1. Go to AWS Console → SageMaker → Studio
# 2. Domain: d-spd2senoxi9j
# 3. Space: snowML-ConvLSTM-khaja
# 4. Click "Open Studio"
# 5. Wait for JupyterLab to load

# Option B: Use AWS CLI
aws sagemaker create-app \
  --domain-id d-spd2senoxi9j \
  --space-name snowML-ConvLSTM-khaja \
  --app-type JupyterLab \
  --app-name default \
  --region us-west-2
```

### **Step 2: Upload Files to SageMaker (5 minutes)**

In SageMaker Studio JupyterLab:

1. **Open Terminal** (File → New → Terminal)

2. **Create working directory:**
```bash
cd /home/sagemaker-user
mkdir -p exp1b_all_6_eval
cd exp1b_all_6_eval
```

3. **Upload the 2 tar.gz files** via drag-and-drop to `exp1b_all_6_eval/` folder:
   - `exp1b_all_6_variations.tar.gz`
   - `exp1b_evaluation_package.tar.gz`

### **Step 3: Extract Files (2 minutes)**

In SageMaker Terminal:

```bash
cd /home/sagemaker-user/exp1b_all_6_eval

# Extract checkpoints
tar -xzf exp1b_all_6_variations.tar.gz
ls -lh Exp1B_*.pth | wc -l  # Should show 178

# Extract evaluation package
tar -xzf exp1b_evaluation_package.tar.gz
ls -la correct_test_sets/

# Verify
echo "Checkpoints: $(ls Exp1B_*.pth | wc -l)"
echo "Test A HUCs: $(wc -l < correct_test_sets/test_a_hucs.txt)"
echo "Test B HUCs: $(wc -l < correct_test_sets/test_b_hucs.txt)"
```

Expected output:
```
Checkpoints: 178
Test A HUCs: 55
Test B HUCs: 81
```

### **Step 4: Verify Python Environment (2 minutes)**

```bash
cd /home/sagemaker-user/exp1b_all_6_eval

# Check Python and packages
python3 --version  # Should be 3.10
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import sys; sys.path.insert(0, '/home/sagemaker-user/src'); from snowML.LSTM import LSTM_evaluate; print('SnowML imported OK')"

# Check GPU
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### **Step 5: Run Evaluation (60-90 minutes)** ✅ Paths Already Configured!

**Note:** The evaluation script already has all AWS SageMaker paths pre-configured to `/home/sagemaker-user/exp1b_all_6_eval/`. No manual editing needed!

```bash
cd /home/sagemaker-user/exp1b_all_6_eval

```bash
# Run in background so you can close browser
nohup python3 evaluate_all_6_variations.py > evaluation_all_6.log 2>&1 &

# Save the process ID
echo $! > evaluation.pid

# Monitor progress
tail -f evaluation_all_6.log

# Or check periodically
tail -100 evaluation_all_6.log
```

**Estimated time:**
- Finding best epochs: ~5 minutes
- Each variation × each test set: ~8-10 minutes
- Total: 6 variations × 2 test sets = **60-90 minutes**

**You can close your browser** - it will keep running!

### **Step 6: Check Progress (Anytime)**

Reconnect to SageMaker and check:

```bash
cd /home/sagemaker-user/exp1b_all_6_eval

# Check if still running
ps aux | grep evaluate_all_6

# Check progress
tail -50 evaluation_all_6.log

# Check results generated so far
ls -lh results/all_6_variations_evaluation/
```

### **Step 7: When Complete - Verify Results**

```bash
cd /home/sagemaker-user/exp1b_all_6_eval

# Check completion
tail -100 evaluation_all_6.log | grep "COMPLETE"

# List all results
ls -lh results/all_6_variations_evaluation/

# Should see:
# - 12 individual CSV files (6 variations × 2 test sets)
# - 1 complete results CSV
# - 1 summary CSV
```

### **Step 8: Download Results (5 minutes)**

**Option A: Via Studio File Browser**
1. Navigate to `exp1b_all_6_eval/results/all_6_variations_evaluation/`
2. Right-click folder → Download
3. Save to your laptop

**Option B: Via S3 (Recommended for backup)**

In SageMaker terminal:
```bash
cd /home/sagemaker-user/exp1b_all_6_eval

# Upload to S3
aws s3 sync results/all_6_variations_evaluation/ \
  s3://snowml-results/simran/exp1b_all_6_variations_eval_$(date +%Y%m%d)/
```

Then on your laptop:
```bash
# Download from S3
aws s3 sync s3://snowml-results/simran/exp1b_all_6_variations_eval_YYYYMMDD/ \
  ~/Desktop/SnowML/results/all_6_variations_eval/
```

### **Step 9: STOP STUDIO APP! ⚠️ CRITICAL**

```bash
# From your laptop terminal
aws sagemaker delete-app \
  --domain-id d-spd2senoxi9j \
  --space-name snowML-ConvLSTM-khaja \
  --app-type JupyterLab \
  --app-name default \
  --region us-west-2

# Verify stopped
aws sagemaker list-apps \
  --domain-id d-spd2senoxi9j \
  --region us-west-2
```

**If you forget: $0.53/hour = $12.72/day = $381/month!** 💸

---

## 📊 EXPECTED RESULTS

After evaluation completes, you should have:

### **Individual Results (12 files):**
- `Base_1e-3_Test_A_metrics.csv`
- `Base_1e-3_Test_B_metrics.csv`
- `Base_3e-4_Test_A_metrics.csv`
- `Base_3e-4_Test_B_metrics.csv`
- `Srad_1e-3_Test_A_metrics.csv`
- `Srad_1e-3_Test_B_metrics.csv`
- `Srad_3e-4_Test_A_metrics.csv`
- `Srad_3e-4_Test_B_metrics.csv`
- `Wind_3e-4_Test_A_metrics.csv`
- `Wind_3e-4_Test_B_metrics.csv`
- `Humidity_3e-4_Test_A_metrics.csv`
- `Humidity_3e-4_Test_B_metrics.csv`

### **Summary Files (2 files):**
- `all_6_variations_complete_YYYYMMDD_HHMMSS.csv` - All HUCs, all variations
- `summary_all_variations_YYYYMMDD_HHMMSS.csv` - Aggregated stats

### **What Each Variation Contains:**

| Variation | Features | Learning Rate | Description |
|-----------|----------|---------------|-------------|
| **Base_1e-3** | Temp + Precip + Elevation | 0.001 | Baseline with 3 features, high LR |
| **Base_3e-4** | Temp + Precip + Elevation | 0.0003 | Baseline with 3 features, optimal LR |
| **Srad_1e-3** | Base + Solar Radiation | 0.001 | Base + solar, high LR |
| **Srad_3e-4** | Base + Solar Radiation | 0.0003 | Base + solar, optimal LR |
| **Wind_3e-4** | Base + Wind Speed | 0.0003 | Base + wind, optimal LR |
| **Humidity_3e-4** | Base + Humidity | 0.0003 | Base + humidity, optimal LR |

**Feature Details:**
- **Temp:** `mean_tair` - Mean air temperature
- **Precip:** `mean_pr` - Mean precipitation
- **Elevation:** `Mean Elevation` - Mean elevation of the HUC
- **Solar Radiation:** `mean_srad` - Mean solar radiation
- **Wind Speed:** `mean_vs` - Mean wind speed
- **Humidity:** `mean_rh` - Mean relative humidity

### **Expected Performance:**
Based on training log analysis:

| Variation | LR | Expected Test KGE | Status |
|-----------|-----|-------------------|---------|
| Base_1e-3 | 0.001 | 0.04-0.35 | ❌ Poor (LR too high) |
| Base_3e-4 | 0.0003 | 0.70-0.84 | ✅ Good baseline |
| Srad_1e-3 | 0.001 | -0.47-0.73 | ❌ Unstable (LR too high) |
| Srad_3e-4 | 0.0003 | 0.70-0.85 | ✅ Good with solar |
| Wind_3e-4 | 0.0003 | 0.81-0.84 | ✅ Excellent (validated) |
| Humidity_3e-4 | 0.0003 | 0.79-0.81 | ✅ Excellent (validated) |

---

## 💰 COST ESTIMATE

- **Instance:** ml.g4dn.xlarge @ $0.53/hour
- **Time:** ~1.5 hours
- **Cost:** ~$0.80-1.00

**Total AWS cost for complete evaluation: < $1.00** ✅

---

## 🐛 TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'snowML'"

```bash
# Check if src folder exists
ls /home/sagemaker-user/src/

# If not, upload snowml-package-gpu-fixed.tar.gz and extract
cd /home/sagemaker-user
tar -xzf snowml-package-gpu-fixed.tar.gz
```

### Issue: "FileNotFoundError: test_a_hucs.txt"

```bash
# Check paths in script match actual locations
ls /home/sagemaker-user/exp1b_all_6_eval/correct_test_sets/
pwd
# Edit script to use absolute paths
```

### Issue: "CUDA out of memory"

The script evaluates one variation at a time, so this shouldn't happen. If it does:
```python
# In script, force CPU:
device = 'cpu'  # instead of 'cuda'
```

### Issue: Evaluation taking too long

Check if it's actually running:
```bash
nvidia-smi  # Should show Python process using GPU
top  # Should show python3 process
tail -f evaluation_all_6.log  # Should show progress
```

---

## ✅ CHECKLIST

**Before starting:**
- [ ] Files ready: `exp1b_all_6_variations.tar.gz` (20 MB)
- [ ] Files ready: `exp1b_evaluation_package.tar.gz` (7.3 KB)
- [ ] AWS credentials configured

**During evaluation:**
- [ ] SageMaker Studio started
- [ ] Files uploaded and extracted (178 checkpoints)
- [ ] Script paths updated for SageMaker
- [ ] Evaluation running in background (check log)
- [ ] Can close browser safely

**After evaluation:**
- [ ] Results generated (14 CSV files)
- [ ] Results downloaded to laptop
- [ ] Results backed up to S3
- [ ] **STOP STUDIO APP** ⚠️

---

**Ready to start!** 🚀

Follow steps 1-10 above. The evaluation will run automatically and generate all results.
