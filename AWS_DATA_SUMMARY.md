# AWS Data Summary - What We Found
**Date:** June 30, 2026  
**Status:** ✅ AWS Access Confirmed

---

## 🎉 **SUCCESS - Your AWS is Working!**

### **What You Can Access:**

**S3 Bucket:** `snowml-model-ready` (Region: us-west-2)

**Contents:**
- 📊 **3,681 HUC-12 time series data files** (model_ready_huc*.csv)
- 📋 **1 classification file** (huc12_snow_classification_master.csv)
- 📁 **2 additional directories** (scf_time_series_partitioned, snow_cover_era5_land)

---

## 📊 **HUC Classification Breakdown**

We downloaded and analyzed `huc12_snow_classification_master.csv`:

### **Total HUCs Available:**
- **460 "deep" snow HUCs** (persistent snowpack - this is what we need!)
- **75 "shallow" snow HUCs** (ephemeral snow)
- **Total: 535 HUCs**

### **What These Mean:**

| Classification | Original Snow Types | Count | Use in Experiments |
|----------------|-------------------|-------|-------------------|
| **deep** | Maritime + Montane Forest + some Prairie/Boreal | 460 | ✅ Use these! |
| **shallow** | Ephemeral | 75 | ⚠️ Exclude in Exp 1B |

---

## 🔍 **Key Insight for Your Experiments**

**Professor's document mentioned:**
- Maritime: 155 HUCs
- Montane Forest: 187 HUCs
- Ephemeral: 180 HUCs
- Prairie: 11 HUCs
- Boreal Forest: 1 HUC
- **Total: 534 HUCs** (in theory)

**What we actually found:**
- Deep snow: 460 HUCs
- Shallow snow: 75 HUCs
- **Total: 535 HUCs**

**What this means:**
- ✅ The classification system has been updated (as you discovered in simran-training-experiments!)
- ✅ "deep" ≈ Maritime + Montane Forest + persistent Prairie/Boreal
- ✅ "shallow" ≈ Ephemeral
- ✅ We have 535 HUCs available (close to professor's 533-534)

---

## 📁 **Data File Structure**

### **Classification File (Downloaded ✅)**
**Location:** `/Users/simran/Desktop/SnowML/data/huc12_snow_classification_master.csv`

**Columns:**
```csv
huc_id,mean_peak_swe,snow_class,mean_peak_swe_meters,is_shallow

Example rows:
170200090101,1081.1,deep,1.081,0       ← High peak SWE, deep snow
170200100502,23.9,shallow,0.024,1      ← Low peak SWE, shallow/ephemeral
```

**Key Fields:**
- `huc_id`: 12-digit HUC identifier
- `mean_peak_swe`: Average peak snow water equivalent (mm)
- `snow_class`: "deep" or "shallow"
- `is_shallow`: 1 = shallow, 0 = deep (easy filtering!)

### **Time Series Files (In S3)**
**Pattern:** `model_ready_huc{HUC_ID}.csv`

**Example:** `model_ready_huc160401010101.csv`

**What's inside each file** (we'll download samples next):
- Daily data from ~1983-2022 (~14,000 rows)
- Columns: date, precipitation, temperature, wind speed, solar radiation, humidity, SWE, elevation, snow type, forest cover

**Size:** ~2 MB per file (1,948 KB to 2,302 KB from what we saw)

---

## 🎯 **What We Need for Experiments**

### **Experiment 1A: All HUCs (including shallow/ephemeral)**
**Data needed:**
- All 535 HUCs (both deep + shallow)
- Total download: ~535 files × 2 MB = ~1.1 GB

### **Experiment 1B: Deep snow only (no ephemeral)**
**Data needed:**
- 460 deep snow HUCs only
- Total download: ~460 files × 2 MB = ~920 MB

### **Experiment 2: Yakima + Naches HUC-12s**
**Data needed:**
- Need to identify which HUC-12 basins are in:
  - Upper Yakima HUC-8: 17030001
  - Naches HUC-8: 17030002
- Estimate: ~20-40 HUC-12 basins
- Total download: ~40-80 MB

---

## 📋 **Next Steps - What We'll Do Together**

### **Step 1: Identify Study Region HUCs** ✅ (Partially done)
- [x] Download classification file
- [x] Count by snow type
- [ ] Filter for Pacific Northwest region
- [ ] Identify which HUCs are in the 533/535 study area

### **Step 2: Create Experiment Splits**
Need to create:
- `exp1a_hucs.json` - All 533-535 HUCs split 60/20/20
- `exp1b_hucs.json` - 460 deep snow HUCs split 60/20/20
- `yakima_naches_hucs.txt` - List of HUC-12s in Test Set B

### **Step 3: Download Sample Data**
- Download 5-10 sample HUCs to test scripts
- Verify data format
- Test training pipeline

### **Step 4: Download Full Dataset**
- Download all needed HUCs for experiments
- Organize in data/ folder

---

## 💻 **Commands We Used (Reference)**

```bash
# List S3 contents
aws s3 ls s3://snowml-model-ready/ --region us-west-2

# Count total files
aws s3 ls s3://snowml-model-ready/ --region us-west-2 | wc -l
# Result: 3,684 files

# Count HUC data files
aws s3 ls s3://snowml-model-ready/ --region us-west-2 | grep "model_ready_huc" | wc -l
# Result: 3,681 HUC files

# Download classification file
aws s3 cp s3://snowml-model-ready/huc12_snow_classification_master.csv ./data/ --region us-west-2

# Analyze snow classes
cut -d',' -f3 data/huc12_snow_classification_master.csv | sort | uniq -c
# Result: 460 deep, 75 shallow
```

---

## 🔑 **Key Files Created**

```
/Users/simran/Desktop/SnowML/
├── data/
│   └── huc12_snow_classification_master.csv  ✅ Downloaded (30 KB)
│
├── COMPLETE_SETUP_GUIDE.md  ✅ Setup instructions
├── CLEAR_ROADMAP.md  ✅ Experiment overview
└── AWS_DATA_SUMMARY.md  ✅ This file
```

---

## ✅ **Status Check**

**Completed:**
- [x] AWS CLI installed
- [x] AWS credentials configured
- [x] S3 bucket access verified
- [x] Classification file downloaded
- [x] Data structure understood
- [x] Snow class counts verified

**Next (Ready to do now):**
- [ ] Identify Pacific NW HUCs (533-535 study region)
- [ ] Create train/val/test splits
- [ ] Download sample data
- [ ] Test training scripts
- [ ] Set up conda environment (if not done yet)

---

## 📊 **Visual Summary**

```
AWS S3: snowml-model-ready
│
├── 3,681 HUC time series files
│   │
│   ├── 460 "deep" snow HUCs
│   │   ├── Maritime-like
│   │   ├── Montane Forest-like
│   │   └── Some persistent Prairie/Boreal
│   │
│   └── 75 "shallow" snow HUCs
│       └── Ephemeral snow
│
└── Classification file ✅ Downloaded
    └── Maps HUC → snow type
```

---

## 🎯 **For Your Reference**

**Professor's Requirements:**
- Exp 1A: Train on ~533 HUCs (all types) → We have 535 available ✅
- Exp 1B: Train on ~270 HUCs (no ephemeral) → We have 460 deep available ✅
- Exp 2: Test on Yakima + Naches → Need to identify these HUC-12s next

**Data Access:**
- ✅ You have full read access to S3
- ✅ You can download any HUC data file
- ✅ Classification data is ready
- ✅ AWS MLflow server available (not tested yet)

---

**Ready for next step?** Let me know and we'll:
1. Create Python script to identify Pacific NW HUCs
2. Generate train/val/test splits
3. Download sample data to test

**Or if conda environment isn't set up yet, we should do that first!**
