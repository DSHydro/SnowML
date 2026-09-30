#!/usr/bin/env python3
"""
Quick fix: Add this to evaluate_from_checkpoints_FIXED.py

Find the evaluate_model_on_test_set function and add device handling
"""

# In the evaluate_model_on_test_set function, after loading train/val HUCs,
# add this code before calling eval_from_saved_model:

# ============ ADD THIS CODE ============

# Move model to GPU if available (same as training)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f"✅ Model moved to: {device}")

# Make sure params has device set
params['device'] = device

# ============ END FIX ============
