# 🚀 Ready for Next Session - Quick Reference

**Last Updated:** July 20, 2026  
**Status:** ✅ GPU fixes complete, pilot notebook ready  
**Next:** Test GPU training on SageMaker

---

## 📁 What You Need to Upload

Upload these 2 files to SageMaker:

1. **`snowml-package-gpu-fixed.tar.gz`** - SnowML with GPU fixes (127 KB)
2. **`Pilot_Test_GPU_Fixed.ipynb`** - Ready-to-run notebook (16 KB)

---

## ⚡ Quick Start (3 Steps)

### 1. Start SageMaker (2 minutes)

```bash
# Start instance
aws sagemaker start-notebook-instance --notebook-instance-name st-gnn-T4x1 --region us-west-2

# Wait 2 minutes, then get URL
aws sagemaker create-presigned-notebook-instance-url --notebook-instance-name st-gnn-T4x1 --region us-west-2 --query 'AuthorizedUrl' --output text
```

### 2. Upload Files (1 minute)

In Jupyter:
- Click **"Upload"**
- Select both files above
- Click "Upload"

### 3. Run Notebook (10 minutes)

Open `Pilot_Test_GPU_Fixed.ipynb` and run cells **in order**:
- Step 1: Check GPU ✅
- Step 2: Install dependencies (3 min)
- Step 3: Install SnowML → **RESTART KERNEL**
- Step 4-12: Run training on GPU (2-3 min)

**That's it!** The notebook has everything.

---

## 🎯 What to Expect

### Success Indicators:
✅ No "device mismatch" errors  
✅ Training completes in ~2-3 minutes (not 5-10 like CPU)  
✅ Validation KGE: 0.6-0.8  
✅ Checkpoints saved  

### If Something Fails:
- Check `GPU_FIX_SUMMARY.md` for troubleshooting
- Check `COMPLETE_SAGEMAKER_TRAINING_GUIDE.md` for full details

---

## 📊 After Pilot Works

### For Full Training:

1. **Upload HUC lists** (from `data/` folder):
   - `exp1b_train_hucs.txt` (231 HUCs)
   - `exp1b_validation_hucs.txt` (77 HUCs)

2. **Modify pilot notebook:**
   ```python
   # Load full lists instead of pilot lists
   with open('exp1b_train_hucs.txt') as f:
       train_hucs = [line.strip() for line in f]
   
   with open('exp1b_validation_hucs.txt') as f:
       val_hucs = [line.strip() for line in f]
   
   # Change epochs
   n_epochs = 30  # Instead of 5
   ```

3. **Run:** 4-6 hours, ~$3

---

## 📖 Documentation Files

| File | Purpose | When to Use |
|------|---------|-------------|
| **`README_NEXT_SESSION.md`** | This file - quick start | Starting next session |
| **`Pilot_Test_GPU_Fixed.ipynb`** | Ready-to-run notebook | Upload to SageMaker |
| **`NEXT_SESSION_QUICK_START.md`** | Step-by-step commands | If you need more detail |
| **`GPU_FIX_SUMMARY.md`** | Technical details | Troubleshooting |
| **`COMPLETE_SAGEMAKER_TRAINING_GUIDE.md`** | Full reference | Everything from today |

---

## 🛑 Don't Forget!

**STOP THE INSTANCE** when done:
```bash
aws sagemaker stop-notebook-instance --notebook-instance-name st-gnn-T4x1 --region us-west-2
```

Or you'll keep paying $0.53/hour!

---

## ✅ What's Already Done

- [x] S3 permissions working
- [x] SnowML GPU fixes applied
- [x] Checkpoint saving added
- [x] Pilot notebook created
- [x] Updated package ready
- [x] All documentation complete

---

## 💡 Key Files Locations

**On your laptop:**
```
/Users/simran/Desktop/SnowML/
├── snowml-package-gpu-fixed.tar.gz  ← Upload this
├── Pilot_Test_GPU_Fixed.ipynb       ← Upload this
├── data/
│   ├── exp1b_train_hucs.txt         ← Upload for full training
│   ├── exp1b_validation_hucs.txt    ← Upload for full training
│   ├── exp1b_test_a_hucs.txt        ← Upload later
│   └── exp1b_test_b_hucs.txt        ← Upload later
└── Documentation (read as needed)
```

---

**You're all set! Just upload the 2 files and run the notebook.** 🎉

Good luck! 🚀
