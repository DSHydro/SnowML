# Exp1A Training Script - Ready to Run!

**Date:** September 14, 2026  
**Script:** `run_exp1a_8_variations.py`  
**Status:** ✅ Ready for AWS SageMaker

---

## ✅ **WHAT'S BEEN CREATED**

### New Training Script: `run_exp1a_8_variations.py`

**Based on your successful Exp1B code with these modifications:**

1. ✅ **15 epochs** (not 30) - saves 50% time
2. ✅ **Loads Exp1A splits** from files (272 train / 90 val / 92 test)
3. ✅ **All 8 variations** (4 features × 2 learning rates)
4. ✅ **Spatial splits** (`train_size_dimension="huc"`)
5. ✅ **Includes ephemeral** HUCs (that's what Exp1A is for!)

---

## 📊 **WHAT IT WILL RUN**

### 8 Variations Total:

**Learning Rate = 0.001:**
1. Base_1e-3: Temp + Precip + Elevation
2. Srad_1e-3: Base + Solar
3. Wind_1e-3: Base + Wind
4. Humidity_1e-3: Base + Humidity

**Learning Rate = 0.0003:**
5. Base_3e-4: Temp + Precip + Elevation
6. Srad_3e-4: Base + Solar
7. Wind_3e-4: Base + Wind (BEST in Exp1B)
8. Humidity_3e-4: Base + Humidity

**⚠️ Note:** Based on Exp1B, LR=0.001 variations may fail/underperform

---

## ⏱️ **TIME & COST ESTIMATE**

### With 15 Epochs:

**Time:**
- Per variation: ~10-12 hours
- Total: 8 × 12 = **~96-100 GPU hours**

**Cost:**
- Spot instances ($0.16/hr): **~$16-17**
- On-demand ($0.526/hr): **~$50-53**

**Calendar Time:**
- Sequential (1 GPU): 4+ days
- Parallel (2 GPUs): 2 days
- Parallel (4 GPUs): 1 day

---

## 🚀 **HOW TO RUN ON SAGEMAKER**

### Step 1: Upload Script to SageMaker

```bash
# From your local machine
scp run_exp1a_8_variations.py sagemaker-user@<your-instance>:/home/sagemaker-user/
```

Or use SageMaker Studio file upload.

### Step 2: Verify Exp1A Split Files Exist

SSH into SageMaker and check:
```bash
ls -la /home/sagemaker-user/src/data/exp1a_*.txt
wc -l /home/sagemaker-user/src/data/exp1a_*.txt
# Should show: 272 train, 90 val, 92 test
```

**If files don't exist on SageMaker:**
```bash
# Copy from your local machine
scp data/exp1a_*.txt sagemaker-user@<your-instance>:/home/sagemaker-user/src/data/
```

### Step 3: Run the Script

```bash
cd /home/sagemaker-user
python3 run_exp1a_8_variations.py
```

**Or run in background:**
```bash
nohup python3 run_exp1a_8_variations.py > exp1a_training.log 2>&1 &
# Monitor with: tail -f exp1a_training.log
```

---

## 📋 **KEY DIFFERENCES FROM EXP1B**

| Aspect | Exp1B (Your Successful Run) | Exp1A (New Script) |
|--------|----------------------------|-------------------|
| **Train HUCs** | 162 (deep only) | **272 (deep + ephemeral)** |
| **Val HUCs** | 54 | **90** |
| **Test A HUCs** | 55 | **92** |
| **Total HUCs** | 270 | **535** |
| **Epochs** | 30 | **15** (optimized) |
| **Variations** | 2 (only LR=0.0003) | **8 (both learning rates)** |
| **Time** | ~36 hours | **~100 hours** |
| **Cost** | ~$17-19 | **~$16-17 (Spot)** |

---

## ✅ **VERIFICATION CHECKLIST**

Before running, verify:

- [ ] Script uploaded to SageMaker: `run_exp1a_8_variations.py`
- [ ] Exp1A split files exist in `/home/sagemaker-user/src/data/`
  - [ ] exp1a_train_hucs.txt (272 HUCs)
  - [ ] exp1a_validation_hucs.txt (90 HUCs)
  - [ ] exp1a_test_a_hucs.txt (92 HUCs)
- [ ] GPU instance running (g4dn.xlarge or similar)
- [ ] Enough disk space (~50GB recommended)
- [ ] MLflow server accessible

---

## 🎯 **AFTER TRAINING COMPLETES**

### 1. Check MLflow Results

The script will log to MLflow experiment: `Exp1A_<variation_name>`

Look for:
- Median validation KGE per epoch
- Best epoch for each variation
- **Select BEST overall variation** (highest median validation KGE)

### 2. Compare Exp1A vs Exp1B

**Key Question:** Does including ephemeral basins help or hurt?

Compare:
- Best Exp1A variation
- Best Exp1B variation (Wind_3e-4, KGE 0.8375)

### 3. Prepare for Experiment 2

**Select:**
- Best Exp1A model → for fine-tuning on Yakima/Naches
- Best Exp1B model (Wind_3e-4) → for fine-tuning comparison

---

## ⚠️ **POTENTIAL ISSUES & SOLUTIONS**

### Issue 1: LR=0.001 Variations Fail
**Expected!** Based on Exp1B, these will likely fail.
- Continue anyway - you need the data to show they failed
- The LR=0.0003 variations will work

### Issue 2: Out of Memory
**Solution:** 
- Check `num_workers` parameter (set to 4)
- Reduce `batch_size` from 32 to 16 if needed

### Issue 3: Script Crashes Mid-Run
**Solution:**
- MLflow saves checkpoint every epoch
- Check which variations completed
- Restart script, it will skip completed ones (if you modify to check MLflow first)

---

## 📊 **EXPECTED OUTCOMES**

### Based on Your Exp1B Results:

**Likely to work well (LR=0.0003):**
- Wind_3e-4: Test KGE ~0.80-0.85 (maybe lower due to ephemeral)
- Humidity_3e-4: Test KGE ~0.75-0.82
- Base_3e-4: Test KGE ~0.70-0.78
- Srad_3e-4: Test KGE ~0.65-0.75

**Likely to fail (LR=0.001):**
- All 0.001 variations: KGE 0.04-0.40 (too high learning rate)

---

## 🎯 **SUCCESS CRITERIA**

**Training is successful if:**
- ✅ At least 1-2 LR=0.0003 variations converge well
- ✅ Best Exp1A model has validation KGE > 0.70
- ✅ Can identify BEST model to use for Exp2 fine-tuning

**It's OK if:**
- ❌ LR=0.001 variations fail (expected from Exp1B)
- ⚠️ Performance is slightly lower than Exp1B (ephemeral adds noise)

---

## 📁 **FILES SUMMARY**

**Created for you:**
- ✅ `run_exp1a_8_variations.py` - Training script (ready to run)
- ✅ This document - Instructions

**Already exist:**
- ✅ `data/exp1a_train_hucs.txt` - 272 train HUCs
- ✅ `data/exp1a_validation_hucs.txt` - 90 validation HUCs  
- ✅ `data/exp1a_test_a_hucs.txt` - 92 test HUCs
- ✅ `data/exp1a_test_b_hucs.txt` - 81 Yakima/Naches HUCs

**Next to create (after Exp1A):**
- ⏳ Exp2 fine-tuning workflow script

---

## 🚀 **YOU'RE READY!**

Everything is set up. Just:

1. Upload `run_exp1a_8_variations.py` to SageMaker
2. Verify split files exist
3. Run: `python3 run_exp1a_8_variations.py`
4. Wait ~4 days (or 1-2 days with parallel GPUs)
5. Check MLflow for results
6. Select best model

**Good luck! 🎉**
