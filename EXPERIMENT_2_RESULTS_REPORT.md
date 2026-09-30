# Experiment 2: Fine-Tuning Results Report

**Date:** September 29, 2026  
**Student:** Simran Dhankar  
**Experiment:** Fine-tuning Exp1B Base model on Yakima/Naches region (Test Set B)

---

## Executive Summary

Fine-tuning the Exp1B Base model on the Yakima/Naches region resulted in **degraded performance** compared to the pre-trained model. The fine-tuned model achieved a Test B KGE of 0.6110, representing a **19.6% decrease** from the pre-trained baseline of 0.7603.

**Recommendation:** Use the original Exp1B Base model without fine-tuning, or explore alternative fine-tuning strategies with more conservative hyperparameters.

---

## 1. Experiment Design

### Objective
Improve performance on Yakima/Naches basins (Test Set B - 81 HUCs) through fine-tuning the pre-trained Exp1B Base model.

### Methodology

**Pre-trained Model:**
- Source: Exp1B Base (trained on 270 deep snow HUCs)
- Checkpoint: Epoch 14 (Val KGE: 0.8084)
- Baseline Test B Performance: KGE = 0.7603

**Fine-tuning Configuration:**
- Training HUCs: 56 (70% of 81 Yakima/Naches HUCs)
- Validation HUCs: 25 (30% of 81 Yakima/Naches HUCs)
- Epochs: 10
- Learning rate: 0.0001 (reduced from 0.0003)
- Dropout: 0.2 (reduced from 0.5)
- Features: Base set (precipitation, temperature, elevation)

**Hardware:**
- Platform: AWS SageMaker Studio
- Instance: ml.g4dn.xlarge (Tesla T4 GPU)
- Training time: ~2 hours

---

## 2. Results

### Performance Comparison

| Metric | Pre-trained (Exp1B Base) | Fine-tuned (Exp2) | Change |
|--------|-------------------------|-------------------|---------|
| **Test B KGE** | **0.7603** | **0.6110** | **-0.1493 (-19.6%)** |
| Test B MSE | 0.0034 | 0.0066 | +0.0032 (+94.1%) |
| Best Val KGE | 0.8084 | 0.9738 | +0.1654 (+20.4%) |

### Training Progress

| Epoch | Validation KGE | Status |
|-------|---------------|---------|
| 1 | 0.5606 | Initial |
| **2** | **0.9738** | **Best** |
| 3 | 0.9116 | - |
| 4 | 0.7501 | - |
| 5 | 0.9123 | - |
| 6 | 0.9125 | - |
| 7 | 0.9049 | - |
| 8 | 0.6238 | - |
| 9 | 0.6285 | - |
| 10 | 0.9327 | Final |

**Observation:** Highly unstable validation performance with large fluctuations between epochs (0.56 → 0.97 → 0.62), suggesting overfitting to the small validation set.

---

## 3. Analysis

### Why Fine-Tuning Failed

**1. Small Dataset Overfitting**
- Only 56 training HUCs and 25 validation HUCs
- Model over-specialized to the specific training subset
- Validation KGE of 0.9738 was misleadingly high due to small validation set
- Poor generalization when evaluated on all 81 Test B HUCs together

**2. Catastrophic Forgetting**
- Pre-trained model learned from 270 diverse deep snow HUCs across multiple regions
- Fine-tuning on a single region (Yakima/Naches) caused the model to "forget" broader spatial patterns
- Lost the ability to generalize across different basin characteristics

**3. Distribution Mismatch**
- The 70/30 split created non-representative subsets
- 25 validation HUCs insufficient to represent full Yakima/Naches variability
- Model optimized for the specific split but failed on the complete region

**4. Yakima/Naches Complexity**
- This region may have unique characteristics not well-represented in the broader Exp1B training data
- Fine-tuning with limited local data insufficient to capture regional complexity

### Training Instability

The large fluctuations in validation KGE across epochs indicate:
- High variance due to small dataset size
- Possible need for stronger regularization
- Potential issues with the train/val split creating unrepresentative subsets

---

## 4. Comparison with Literature

Transfer learning typically improves performance when:
- Fine-tuning dataset is large enough (typically >1000 samples)
- Target domain is similar to source domain
- Regularization prevents overfitting

In this case:
- Dataset too small (56 HUCs vs. 270 in pre-training)
- Single region vs. multi-region pre-training reduced diversity
- Regularization (dropout 0.2) may have been insufficient

---

## 5. Recommendations

### Option 1: Use Pre-trained Model (Recommended)
**Action:** Deploy Exp1B Base model without fine-tuning  
**Test B KGE:** 0.7603  
**Rationale:** 
- Proven performance on unseen watersheds
- Better generalization than fine-tuned version
- Lower risk of overfitting

### Option 2: Try Conservative Fine-Tuning
If fine-tuning is still desired, test these modifications:

**Hyperparameter Adjustments:**
- Learning rate: 5e-5 (half of current)
- Dropout: 0.4 (double current, stronger regularization)
- Epochs: 5 (reduce overfitting risk)
- Weight decay: 1e-4 (add L2 regularization)

**Training Strategy:**
- Freeze encoder layers, only fine-tune final prediction layer
- Use early stopping with patience=2 epochs
- Increase validation set size (use 80/20 split instead of 70/30)

### Option 3: Ensemble Approach
- Combine predictions from Exp1B Base (0.7603) and best fine-tuned epoch
- May capture benefits of both approaches
- Requires additional validation

### Option 4: Regional Data Augmentation
- Collect more Yakima/Naches specific data if available
- Augment with similar climate regions (e.g., neighboring basins)
- Expand training set size before fine-tuning

---

## 6. Lessons Learned

### Technical Insights
1. **Transfer learning requires sufficient target data** - 56 HUCs insufficient for stable fine-tuning
2. **Validation set size matters** - 25 HUCs too small to reliably estimate performance
3. **Regional specialization has trade-offs** - Gains on subset may hurt overall performance

### For Future Experiments
1. Run pilot tests with different train/val splits before full training
2. Consider k-fold cross-validation for small datasets
3. Monitor both per-epoch validation and full-region evaluation
4. Implement more aggressive regularization for small datasets

---

## 7. Next Steps

### Immediate Actions
1. **Discuss results with committee** - Determine if 0.7603 KGE acceptable for Yakima/Naches
2. **Investigate regional patterns** - Analyze which specific HUCs perform poorly and why
3. **Consider alternative approaches** - Explore other methods if fine-tuning deemed necessary

### If Pursuing Alternative Fine-Tuning
1. Implement Option 2 (conservative hyperparameters)
2. Test with k-fold cross-validation (5-fold on 81 HUCs)
3. Track per-HUC performance to identify systematic issues
4. Compare ensemble approach (Option 3)

### For Thesis Documentation
- Document both pre-trained and fine-tuned results
- Present as exploration of transfer learning limitations
- Discuss small-dataset challenges in hydrologic modeling
- Position negative result as valuable scientific finding

---

## 8. Conclusion

Experiment 2 demonstrates that fine-tuning on a small regional dataset (81 HUCs) degraded the pre-trained model's performance. The **original Exp1B Base model (Test B KGE = 0.7603) should be used** for Yakima/Naches predictions unless alternative fine-tuning strategies can be shown to improve performance through rigorous cross-validation.

This result provides valuable insight into the limitations of transfer learning in hydrologic modeling when target datasets are small and represents an important finding for the thesis.

---

## Appendix A: Detailed Metrics

### Per-Epoch Performance Summary

```
Epoch  Val_KGE  Val_MSE   Time(s)  Best
-----  -------  --------  -------  ----
1      0.5606   0.000370  666      ✓
2      0.9738   0.001534  670      ✓
3      0.9116   0.001665  662      
4      0.7501   0.002240  658      
5      0.9123   0.001266  657      
6      0.9125   0.003193  682      
7      0.9049   0.000334  681      
8      0.6238   0.045479  681      
9      0.6285   0.004630  680      
10     0.9327   0.004123  677      
```

### Final Evaluation Details

**Model:** Best checkpoint (Epoch 2, Val KGE 0.9738)  
**Test Set:** All 81 Yakima/Naches HUCs  
**Final Test B KGE:** 0.6110  
**Final Test B MSE:** 0.006640

**Comparison:**
- Pre-trained Test B KGE: 0.7603
- Fine-tuned Test B KGE: 0.6110
- Performance loss: 19.6%

---

## Appendix B: Files Generated

**Training Outputs:**
- Script: `/home/sagemaker-user/finetune_exp1b_base_CORRECT.py`
- Log: `/Users/simran/Desktop/SnowML/finetune.log`
- Checkpoints: `/home/sagemaker-user/exp2_finetune_results/FineTune_Base_epoch*.pth`

**Results Files:**
- Summary: `exp2_finetune_results/finetune_summary.json`
- Splits: `exp2_finetune_results/finetune_splits.json`
- MLflow Run: Experiment 707, Run ID 73855c685e094cdb8aff0f27f036d25e

**This Report:**
- Location: `EXPERIMENT_2_RESULTS_REPORT.md`
- Date: September 29, 2026

---

**Report prepared by:** Simran Dhankar  
**Supervised by:** [Professor Name]  
**Institution:** University of Washington  
**Program:** MS Thesis - Snow Water Equivalent Prediction using LSTM
