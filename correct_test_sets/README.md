# Correct Test Sets for Exp1B Evaluation

**Created:** August 30, 2026  
**Source:** Extracted from `src/snowML/datapipe/huc_lists/hucs_data.json`  
**Purpose:** Evaluation of Wind_3e-4 and Humidity_3e-4 models

---

## ✅ Files in This Folder

### For Evaluation (Use These):

1. **`test_a_hucs.txt`** (55 HUCs)
   - Held-out deep snow HUCs from the training split
   - Source: `hucs_data.json["test_hucs"]`
   - Purpose: Test spatial generalization to similar (unseen) watersheds
   - **This is the Test Set A for evaluation**

2. **`test_b_hucs.txt`** (81 HUCs)
   - Yakima and Naches watersheds (completely separate region)
   - Source: `data/test_set_b_hucs.txt`
   - Purpose: Test spatial transfer to a new geographic area
   - **This is Test Set B for evaluation**

### For Reference (Training Context):

3. **`train_hucs.txt`** (162 HUCs)
   - Training HUCs used for Wind_3e-4 and Humidity_3e-4
   - Source: `hucs_data.json["train_hucs"]`

4. **`val_hucs.txt`** (54 HUCs)
   - Validation HUCs used during training
   - Source: `hucs_data.json["val_hucs"]`

---

## 📊 HUC Split Summary

| Split | HUCs | Percentage | Purpose |
|-------|------|------------|---------|
| Train | 162 | 60% | Model training |
| Val | 54 | 20% | Hyperparameter tuning |
| Test A | 55 | 20% | Spatial generalization test |
| **Total** | **271** | **100%** | **Deep snow HUCs only** |
| Test B | 81 | Separate | New region transfer test |

---

## ✅ Verification

### No Overlaps Between Splits:
- Train ∩ Val = 0 HUCs ✅
- Train ∩ Test A = 0 HUCs ✅
- Val ∩ Test A = 0 HUCs ✅

### Clean Spatial Separation:
All splits are based on entire HUC time series (HUC-based splits), not time-based splits.

---

## 🚀 How to Use on SageMaker

### 1. Upload to SageMaker:

Upload this entire `correct_test_sets/` folder to SageMaker:

```bash
# On SageMaker Studio, create folder
mkdir -p /home/sagemaker-user/correct_test_sets

# Upload via Studio file browser or SCP
# All 4 .txt files from this folder
```

### 2. Update Evaluation Script:

Use these paths in your evaluation script:

```python
TEST_SETS = {
    "Test_A": "/home/sagemaker-user/correct_test_sets/test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/correct_test_sets/test_b_hucs.txt"
}
```

### 3. Expected Results:

**Wind_3e-4:**
- Validation KGE: 0.8104 (54 HUCs)
- Test A KGE: 0.75-0.80 expected (55 HUCs)
- Test B KGE: 0.70-0.75 expected (81 HUCs)

**Humidity_3e-4:**
- Validation KGE: 0.6212 (54 HUCs)
- Test A KGE: 0.55-0.60 expected (55 HUCs)
- Test B KGE: 0.50-0.55 expected (81 HUCs)

---

## 🔬 Comparison to Original Exp3

The original Exp3 in the main branch used these EXACT same splits:
- Same 162 train HUCs
- Same 54 val HUCs
- Same 55 test HUCs

**Original Exp3 Results:**
- Validation KGE: ~0.82
- Test KGE: ~0.72
- Drop: ~12% (good generalization)

**Your Wind_3e-4 should match or beat this!**

---

## ⚠️ DO NOT USE

**DO NOT use the old files in `data/` folder:**
- ❌ `data/exp1b_train_hucs.txt` (231 HUCs - WRONG)
- ❌ `data/exp1b_validation_hucs.txt` (77 HUCs - WRONG)
- ❌ `data/exp1b_test_a_hucs.txt` (78 HUCs - WRONG)

These are from the INCORRECT initial training with time-based splits and do not match your corrected HUC-based training.

---

## 📁 Files Summary

```
correct_test_sets/
├── README.md              (this file)
├── test_a_hucs.txt        (55 HUCs) ← Use for Test Set A evaluation
├── test_b_hucs.txt        (81 HUCs) ← Use for Test Set B evaluation
├── train_hucs.txt         (162 HUCs) - Reference only
└── val_hucs.txt           (54 HUCs) - Reference only
```

---

**Last Updated:** August 30, 2026  
**Matches Training:** Yes - Wind_3e-4 and Humidity_3e-4 (Aug 24-26, 2026)  
**Ready for Evaluation:** Yes
