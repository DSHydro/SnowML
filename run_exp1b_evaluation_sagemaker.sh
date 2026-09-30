#!/bin/bash
# Upload evaluation script and models to SageMaker, then run evaluation

set -e  # Exit on error

INSTANCE_NAME="st-gnn-T4x1"
REGION="us-west-2"

echo "=============================================="
echo "Exp1B Evaluation on SageMaker"
echo "=============================================="
echo ""

# Check if instance is running
echo "📊 Checking instance status..."
STATUS=$(aws sagemaker describe-notebook-instance \
    --notebook-instance-name $INSTANCE_NAME \
    --region $REGION \
    --query 'NotebookInstanceStatus' \
    --output text)

echo "   Current status: $STATUS"

if [ "$STATUS" = "Stopped" ]; then
    echo ""
    echo "🚀 Starting SageMaker instance..."
    aws sagemaker start-notebook-instance \
        --notebook-instance-name $INSTANCE_NAME \
        --region $REGION

    echo "   Waiting for instance to start (this takes 2-3 minutes)..."
    aws sagemaker wait notebook-instance-in-service \
        --notebook-instance-name $INSTANCE_NAME \
        --region $REGION
    echo "   ✅ Instance is running!"
elif [ "$STATUS" = "InService" ]; then
    echo "   ✅ Instance already running"
else
    echo "   ⚠️  Instance is in '$STATUS' state. Waiting..."
    aws sagemaker wait notebook-instance-in-service \
        --notebook-instance-name $INSTANCE_NAME \
        --region $REGION
fi

echo ""
echo "📦 Uploading files to S3..."

# Upload evaluation script
aws s3 cp evaluate_exp1b_sagemaker.py \
    s3://snowml-results/simran/exp1b_evaluation/

# Upload test set files
aws s3 cp data/exp1b_test_a_hucs.txt \
    s3://snowml-results/simran/exp1b_evaluation/
aws s3 cp data/exp1b_test_b_hucs.txt \
    s3://snowml-results/simran/exp1b_evaluation/

# Upload model checkpoints
aws s3 sync models/exp1b/ \
    s3://snowml-results/simran/exp1b_evaluation/models/ \
    --exclude "*" --include "*.pt"

echo "   ✅ Files uploaded to S3"

echo ""
echo "🔗 Getting Jupyter URL..."
JUPYTER_URL=$(aws sagemaker create-presigned-notebook-instance-url \
    --notebook-instance-name $INSTANCE_NAME \
    --region $REGION \
    --query 'AuthorizedUrl' \
    --output text)

echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Open Jupyter in your browser:"
echo "   $JUPYTER_URL"
echo ""
echo "2. Open a Terminal: New → Terminal"
echo ""
echo "3. Download files from S3:"
echo "   cd /home/ec2-user/SageMaker"
echo "   aws s3 sync s3://snowml-results/simran/exp1b_evaluation/ ./exp1b_evaluation/"
echo ""
echo "4. Run evaluation:"
echo "   cd exp1b_evaluation"
echo "   nohup python evaluate_exp1b_sagemaker.py > evaluation.log 2>&1 &"
echo ""
echo "5. Monitor progress:"
echo "   tail -f evaluation.log"
echo ""
echo "6. When done, download results:"
echo "   aws s3 sync results/ s3://snowml-results/simran/exp1b_evaluation/results/"
echo ""
echo "=============================================="
