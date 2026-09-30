# MLflow Analysis Summary

## Files Analyzed
- mlflow_1.csv: 255 runs
- mlflow_2.csv: 444 runs  
- Total: 699 runs from previous students

## Key Findings

### Multi_All Experiments (Closest to What We Need)
- **Multi_All-2**: 31 runs, 22 finished
- Average HUCs tested: **39 HUCs only**
- This is FAR LESS than needed (270 or 535)
- These appear to be PARTIAL/TEST runs, not full experiments

### Other Experiments Found
**Maritime (Mar_) experiments:** 
- Mar_Mixed_Loss: 16 runs
- Mar_Multi: 9 runs
- Mar_Mixed_Loss_Stop: 12 runs

**Montane (Mon_) experiments:**
- Mon_Mixed_Loss: 7 runs
- Mon_Multi: 5 runs  
- Mon_Mixed_Loss_Stop: 17 runs

**Special experiments:**
- Ephem-MixedLoss: 3 runs (ephemeral snow!)
- Data_Integration: 7 runs
- Forest Cover: 5 runs
- 90 Day Lookback: 5 runs

### Critical Missing Information
❌ No experiments with ~270 HUCs (full Exp1B)
❌ No experiments with ~535 HUCs (Exp1A with ephemeral)
❌ Most experiments focus on individual snow types (Maritime, Montane)
❌ "Multi_All" experiments only tested 39 HUCs on average

## Conclusion
⚠️ **NO USABLE EXP1A OR FULL EXP1B FOUND IN MLFLOW**

Previous students ran:
- Individual snow type experiments (Maritime, Montane, Ephemeral separately)
- Small multi-HUC tests (~39 HUCs)
- Data integration and feature experiments

They did NOT run:
- Full Exp1B (270 deep snow HUCs trained together)
- Exp1A (535 HUCs including ephemeral)

## Recommendation
**You CANNOT reuse previous experiments** because:
1. Multi_All experiments only used ~39 HUCs (too small)
2. Snow type experiments were separate, not joint training
3. No evidence of 270-HUC or 535-HUC experiments

**You must run:**
- Exp1A from scratch (if professor wants it)
- Your Exp1B results (162/54/55 split) are already complete and valid!
