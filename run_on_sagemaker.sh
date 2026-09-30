#!/bin/bash
# Launch training on SageMaker Studio with g4dn.2xlarge

set -e

echo "=========================================="
echo "SAGEMAKER PILOT TEST - g4dn.2xlarge"
echo "=========================================="

# Package code for SageMaker
echo "Packaging code..."
tar -czf sourcedir.tar.gz train_pilot_g4dn.py data/ src/

# Upload to S3
echo "Uploading to S3..."
aws s3 cp sourcedir.tar.gz s3://snowml-model-ready/code/

# Create training script wrapper
cat > sagemaker_train.py <<'EOF'
import subprocess
import sys

# Extract source code
subprocess.run(['tar', '-xzf', '/opt/ml/input/data/code/sourcedir.tar.gz', '-C', '/opt/ml/code/'])

# Run training
sys.path.insert(0, '/opt/ml/code')
exec(open('/opt/ml/code/train_pilot_g4dn.py').read())
EOF

aws s3 cp sagemaker_train.py s3://snowml-model-ready/code/

JOB_NAME="snowml-pilot-$(date +%s)"

echo "Launching SageMaker training job: $JOB_NAME"

aws sagemaker create-training-job \
  --training-job-name "$JOB_NAME" \
  --role-arn "arn:aws:iam::677276086662:role/service-role/AmazonSageMaker-ExecutionRole-20250214T075112" \
  --algorithm-specification TrainingImage=763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker,TrainingInputMode=File \
  --resource-config InstanceType=ml.g4dn.2xlarge,InstanceCount=1,VolumeSizeInGB=100 \
  --input-data-config "[{\"ChannelName\":\"code\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"s3://snowml-model-ready/code/\",\"S3DataDistributionType\":\"FullyReplicated\"}}}]" \
  --output-data-config S3OutputPath=s3://snowml-model-ready/output/ \
  --stopping-condition MaxRuntimeInSeconds=7200 \
  --region us-west-2

echo ""
echo "✅ Job launched: $JOB_NAME"
echo ""
echo "Monitor with:"
echo "  aws sagemaker describe-training-job --training-job-name $JOB_NAME --region us-west-2"
