#!/bin/bash
# Setup Exp1B evaluation on SageMaker (uses existing checkpoints in S3)

set -e

INSTANCE_NAME="st-gnn-T4x1"
REGION="us-west-2"

echo "=============================================="
echo "Exp1B Test Set Evaluation - Setup"
echo "=============================================="
echo ""

# Check instance status
echo "📊 Checking SageMaker instance..."
STATUS=$(aws sagemaker describe-notebook-instance \
    --notebook-instance-name $INSTANCE_NAME \
    --region $REGION \
    --query 'NotebookInstanceStatus' \
    --output text)

echo "   Status: $STATUS"

if [ "$STATUS" = "Stopped" ]; then
    echo ""
    echo "🚀 Starting instance (takes 2-3 minutes)..."
    aws sagemaker start-notebook-instance \
        --notebook-instance-name $INSTANCE_NAME \
        --region $REGION

    aws sagemaker wait notebook-instance-in-service \
        --notebook-instance-name $INSTANCE_NAME \
        --region $REGION
    echo "   ✅ Instance started!"
elif [ "$STATUS" = "InService" ]; then
    echo "   ✅ Already running"
fi

echo ""
echo "📦 Uploading evaluation files to S3..."

# Upload evaluation script
aws s3 cp evaluate_exp1b_sagemaker.py \
    s3://snowml-results/simran/exp1b_evaluation/ \
    --quiet

# Upload test set files
aws s3 cp data/exp1b_test_a_hucs.txt \
    s3://snowml-results/simran/exp1b_evaluation/ \
    --quiet
aws s3 cp data/exp1b_test_b_hucs.txt \
    s3://snowml-results/simran/exp1b_evaluation/ \
    --quiet

echo "   ✅ Files uploaded!"
echo ""
echo "📍 Checkpoints already in S3:"
echo "   s3://snowml-results/simran/exp1b_checkpoints/20260805_230759/"

echo ""
echo "🔗 Getting Jupyter URL..."
JUPYTER_URL=$(aws sagemaker create-presigned-notebook-instance-url \
    --notebook-instance-name $INSTANCE_NAME \
    --region $REGION \
    --query 'AuthorizedUrl' \
    --output text)

echo ""
echo "=============================================="
echo "✅ Ready to Run Evaluation!"
echo "=============================================="
echo ""
echo "1. Open Jupyter:"
echo "   $JUPYTER_URL"
echo ""
echo "2. Open Terminal (New → Terminal) and run:"
echo ""
echo "   cd /home/ec2-user/SageMaker"
echo ""
echo "   # Download evaluation script and test sets"
echo "   aws s3 sync s3://snowml-results/simran/exp1b_evaluation/ ./exp1b_evaluation/ --exclude 'results/*'"
echo ""
echo "   # Download checkpoints (already in S3)"
echo "   aws s3 sync s3://snowml-results/simran/exp1b_checkpoints/20260805_230759/ ./exp1b_evaluation/models/"
echo ""
echo "   # Run evaluation (in background)"
echo "   cd exp1b_evaluation"
echo "   nohup python evaluate_exp1b_sagemaker.py > evaluation.log 2>&1 &"
echo ""
echo "   # Monitor progress"
echo "   tail -f evaluation.log"
echo ""
echo "   # When done, upload results back to S3"
echo "   aws s3 sync results/ s3://snowml-results/simran/exp1b_evaluation/results/"
echo ""
echo "=============================================="
