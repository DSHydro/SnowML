#!/bin/bash
# Quick script to restart training on AWS SageMaker
# Run this from your laptop to get the Jupyter URL

echo "======================================================================"
echo "RESTART TRAINING - QUICK START"
echo "======================================================================"
echo ""

echo "Step 1: Getting SageMaker Jupyter URL..."
echo ""

JUPYTER_URL=$(aws sagemaker create-presigned-notebook-instance-url \
  --notebook-instance-name st-gnn-T4x1 \
  --region us-west-2 \
  --query 'AuthorizedUrl' \
  --output text)

if [ $? -eq 0 ]; then
    echo "✅ Got URL successfully!"
    echo ""
    echo "📋 COPY THIS URL AND OPEN IN BROWSER:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$JUPYTER_URL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Try to open automatically
    if command -v open &> /dev/null; then
        echo "🌐 Opening in browser..."
        open "$JUPYTER_URL"
    fi
else
    echo "❌ Failed to get URL. Check:"
    echo "   1. AWS credentials configured"
    echo "   2. Instance name is correct (st-gnn-T4x1)"
    echo "   3. Instance is running"
    exit 1
fi

echo ""
echo "======================================================================"
echo "NEXT STEPS IN JUPYTER:"
echo "======================================================================"
echo ""
echo "1. Click 'New' → 'Terminal' (top right)"
echo ""
echo "2. In the terminal, run:"
echo "   cd ~"
echo "   nohup python train_exp1b_background.py > training.log 2>&1 &"
echo ""
echo "3. Check it started:"
echo "   tail -f training.log"
echo ""
echo "4. Press Ctrl+C to stop watching (training keeps going!)"
echo ""
echo "5. Close laptop and go home! 🏠"
echo ""
echo "======================================================================"
echo "CHECK PROGRESS LATER:"
echo "======================================================================"
echo ""
echo "Run this script again, then in terminal:"
echo "   tail -f training.log"
echo ""
echo "======================================================================"
