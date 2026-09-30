# Exp1A Training - 4 Variations (LR=0.0003 Only)

**Created:** September 17, 2026  
**Script:** `run_exp1a_4_variations.py`  
**Status:** ✅ Ready to run on AWS SageMaker

---

## 🎯 **What Changed from Sept 14th Script**

### Old Script: `run_exp1a_8_variations.py`
- 8 variations (4 features × 2 learning rates)
- Included LR=0.001 (expected to fail based on Exp1B)
- ~100 GPU hours
- ~$16-17 cost

### New Script: `run_exp1a_4_variations.py` ✅
- **4 variations (4 features × 1 learning rate = 0.0003 ONLY)**
- Skips LR=0.001 (proven to fail in Exp1B)
- **~48-50 GPU hours (50% time savings!)**
- **~$8-9 cost (50% cost savings!)**

---

## 📊 **The 4 Variations**

All with **Learning Rate = 0.0003** (proven to work in Exp1B):

1. **Base_3e-4**: Temperature + Precipitation + Elevation
2. **Srad_3e-4**: Base + Solar Radiation
3. **Wind_3e-4**: Base + Wind Speed (BEST in Exp1B with KGE 0.8375!)
4. **Humidity_3e-4**: Base + Humidity

---

## ⏱️ **Time & Cost Estimate**

### With 4 Variations (LR=0.0003 only):

**Time:**
- Per variation: ~10-12 hours
- Total: **~48-50 GPU hours**

**Cost:**
- **Spot instances ($0.16/hr): ~$8-9** ✅ RECOMMENDED
- On-demand ($0.526/hr): ~$25-26

**Calendar Time:**
- Sequential (1 GPU): **2 days**
- Parallel (2 GPUs): 1 day
- Parallel (4 GPUs): 12 hours

---

## 🚀 **How to Run**

### Step 1: Upload to SageMaker

**Option A: Using SageMaker Studio File Upload**
1. Open SageMaker Studio
2. Click "Upload Files"
3. Select `run_exp1a_4_variations.py`

**Option B: Using SCP (if you have SSH access)**
```bash
scp run_exp1a_4_variations.py sagemaker-user@<your-instance>:/home/sagemaker-user/
```

---

### Step 2: Verify Split Files Exist

SSH or open terminal in SageMaker Studio:
```bash
cd /home/sagemaker-user/src/data
ls -la exp1a_*.txt
wc -l exp1a_*.txt
```

**Expected output:**
```
  272 exp1a_train_hucs.txt
   90 exp1a_validation_hucs.txt
   92 exp1a_test_a_hucs.txt
   81 exp1a_test_b_hucs.txt
```

**If files missing:** Upload from your laptop:
```bash
scp data/exp1a_*.txt sagemaker-user@<instance>:/home/sagemaker-user/src/data/
```

---

### Step 3: Start Training

**Run in foreground (to monitor):**
```bash
cd /home/sagemaker-user
python3 run_exp1a_4_variations.py
```

**Run in background (recommended for long jobs):**
```bash
cd /home/sagemaker-user
nohup python3 run_exp1a_4_variations.py > exp1a_training.log 2>&1 &

# Monitor progress
tail -f exp1a_training.log

# Check if still running
ps aux | grep run_exp1a
```

---

## 📋 **Pre-Flight Checklist**

Before starting training:

- [ ] Script uploaded: `run_exp1a_4_variations.py`
- [ ] Split files exist: `data/exp1a_*.txt` (4 files)
- [ ] GPU instance running (g4dn.xlarge or similar)
- [ ] MLflow server accessible
- [ ] Enough disk space (~30GB recommended)
- [ ] AWS budget approved (~$9 for training)

---

## 📊 **Expected Results**

### Based on Your Exp1B Performance:

| Variation | Expected Val KGE | Expected Test A KGE | Notes |
|-----------|-----------------|---------------------|-------|
| **Wind_3e-4** | **~0.75-0.80** | **~0.78-0.83** | Best in Exp1B (0.84) |
| **Humidity_3e-4** | ~0.70-0.78 | ~0.75-0.82 | Was 0.81 in Exp1B |
| **Base_3e-4** | ~0.68-0.75 | ~0.70-0.78 | Baseline |
| **Srad_3e-4** | ~0.65-0.73 | ~0.68-0.76 | New, untested |

**Note:** Performance likely slightly lower than Exp1B due to ephemeral basins adding noise.

---

## 🔍 **How to Monitor Progress**

### Option 1: Log File
```bash
tail -f exp1a_training.log
```

Look for:
- `VARIATION X/4: <name>` - Which variation is training
- `Epoch X/15` - Progress within variation
- `✅ <name> COMPLETED!` - Variation finished
- `⏱️ Total elapsed: X hours` - Time tracking

### Option 2: MLflow UI
1. Open MLflow server: `arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML`
2. Look for experiments:
   - `Exp1A_Base_3e-4`
   - `Exp1A_Srad_3e-4`
   - `Exp1A_Wind_3e-4`
   - `Exp1A_Humidity_3e-4`
3. Check metrics in real-time

### Option 3: GPU Monitoring
```bash
watch -n 5 nvidia-smi
```
- GPU usage should be >80% during training
- Memory usage should be ~10-14GB (out of 16GB)

---

## ⚠️ **Troubleshooting**

### Issue 1: File Not Found Error
**Error:** `FileNotFoundError: data/exp1a_train_hucs.txt`

**Fix:**
```bash
# Check if files exist
ls /home/sagemaker-user/src/data/exp1a_*.txt

# If missing, upload them
scp data/exp1a_*.txt sagemaker-user@<instance>:/home/sagemaker-user/src/data/
```

### Issue 2: CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Fix:**
- Reduce `batch_size` from 32 to 16 in the script
- Reduce `num_workers` from 4 to 2
- Restart and run again

### Issue 3: Script Crashes Mid-Run
**Solution:**
1. Check which variations completed in MLflow
2. Edit script to skip completed ones (comment out in `variations` list)
3. Restart script

### Issue 4: MLflow Connection Error
**Error:** `Could not connect to MLflow server`

**Fix:**
```bash
# Test MLflow connection
python3 -c "import mlflow; mlflow.set_tracking_uri('arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML'); print(mlflow.get_tracking_uri())"
```

---

## ✅ **After Training Completes (~2 days)**

### Step 1: Check Results in MLflow

1. Open MLflow UI
2. For each experiment (`Exp1A_<name>`):
   - Check final epoch metrics
   - Look for `median_test_kge` (validation KGE)
   - Identify best epoch

### Step 2: Find Best Model

Sort by median validation KGE:
- Expected winner: **Wind_3e-4** (was best in Exp1B)
- Runner-up: Humidity_3e-4

### Step 3: Compare Exp1A vs Exp1B

| Metric | Exp1B (270 HUCs, deep only) | Exp1A (535 HUCs, with ephemeral) |
|--------|----------------------------|----------------------------------|
| **Train HUCs** | 162 | 272 |
| **Best Model** | Wind_3e-4 (KGE 0.8375) | TBD (likely Wind_3e-4) |
| **Includes Ephemeral** | No | Yes |
| **Expected Performance** | Higher (specialist) | Lower (generalist) |

**Key Question:** Does including ephemeral hurt or help?

### Step 4: Evaluate on Test Sets

Create evaluation script to test best Exp1A model on:
- Test Set A: 92 random HUCs
- Test Set B: 81 Yakima/Naches HUCs

### Step 5: Download Important Files

```bash
# Checkpoints
scp -r sagemaker-user@<instance>:/home/sagemaker-user/checkpoints/Exp1A_* ./exp1a_checkpoints/

# Training log
scp sagemaker-user@<instance>:/home/sagemaker-user/exp1a_training.log ./
```

### Step 6: ⚠️ STOP INSTANCE!

**CRITICAL:** Stop Studio to avoid charges!

```bash
aws sagemaker stop-notebook-instance \
  --notebook-instance-name <your-instance-name> \
  --region us-west-2
```

Or delete Studio app from console.

---

## 📊 **Comparison Table: Exp1A vs Exp1B**

| Aspect | Exp1B (Complete) | Exp1A (This Run) |
|--------|------------------|------------------|
| **Status** | ✅ Complete | ⏳ Ready to run |
| **Train HUCs** | 162 (deep only) | 272 (deep + ephemeral) |
| **Val HUCs** | 54 | 90 |
| **Test A HUCs** | 55 | 92 |
| **Epochs** | 30 | 15 |
| **Variations** | 2 (Wind, Humidity @ 0.0003) | 4 (Base, Srad, Wind, Humidity @ 0.0003) |
| **Best Result** | Wind_3e-4: KGE 0.8375 | TBD (likely 0.75-0.82) |
| **Training Time** | ~36 hours | ~48-50 hours |
| **Cost** | ~$17-19 | ~$8-9 (Spot) |

---

## 🎯 **Success Criteria**

Training is successful if:
- ✅ All 4 variations complete without crashing
- ✅ At least 1-2 models have validation KGE > 0.70
- ✅ Can identify BEST model for Experiment 2
- ✅ Results logged to MLflow correctly

Performance expectations:
- ✅ Validation KGE 0.70-0.82 = SUCCESS
- ⚠️ Validation KGE 0.60-0.70 = OK (ephemeral adds noise)
- ❌ Validation KGE < 0.60 = Investigate

---

## 🚀 **Next Steps After Exp1A**

1. **Evaluate best Exp1A model on Test Sets A & B**
2. **Compare Exp1A vs Exp1B:**
   - Which performs better on deep snow?
   - Which performs better on ephemeral?
   - Does multi-snow-type training hurt specialists?
3. **Design Experiment 2:**
   - Fine-tune best Exp1A on 81 Yakima/Naches HUCs
   - Fine-tune best Exp1B on 81 Yakima/Naches HUCs
   - Compare pre-trained-only vs fine-tuned
4. **Write thesis sections**

---

## 💡 **Key Decisions Made**

1. **Skip LR=0.001 variations:**
   - Rationale: Failed in Exp1B (KGE 0.04-0.35)
   - Saves: 50% time and cost
   - Risk: None (proven to fail)

2. **Keep 15 epochs:**
   - Rationale: Exp1B converged at epochs 6-7
   - Safety: 2× margin is sufficient
   - Saves: 50% vs original 30 epochs

3. **Train all 4 feature sets:**
   - Rationale: Need complete comparison
   - Benefit: Systematic feature evaluation
   - Expected winner: Wind_3e-4 (was best in Exp1B)

---

## 📁 **Files Summary**

**Created today:**
- ✅ `run_exp1a_4_variations.py` - Training script (4 vars, LR=0.0003)
- ✅ `EXP1A_4_VARIATIONS_GUIDE.md` - This guide

**Already exist:**
- ✅ `data/exp1a_train_hucs.txt` - 272 train HUCs
- ✅ `data/exp1a_validation_hucs.txt` - 90 validation HUCs
- ✅ `data/exp1a_test_a_hucs.txt` - 92 test HUCs
- ✅ `data/exp1a_test_b_hucs.txt` - 81 Yakima/Naches HUCs

**Previous (Sept 14):**
- 📄 `run_exp1a_8_variations.py` - Original 8-variation script (not using)
- 📄 `EXP1A_READY_TO_RUN.md` - Original guide for 8 variations

---

## ✅ **You're Ready to Run!**

**Quick checklist:**
1. Upload `run_exp1a_4_variations.py` to SageMaker
2. Verify split files exist in `src/data/`
3. Run: `nohup python3 run_exp1a_4_variations.py > exp1a_training.log 2>&1 &`
4. Monitor: `tail -f exp1a_training.log`
5. Wait ~2 days
6. Check MLflow for results
7. **Stop instance!**

**Good luck! 🎉**

---

**Questions before running? Let me know!**
