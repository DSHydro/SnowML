#!/bin/bash
# Launch p4d.24xlarge instance and start Experiment 1B training
# This script automates the entire setup process

set -e  # Exit on error

echo "========================================="
echo "Launch p4d.24xlarge for Experiment 1B"
echo "========================================="
echo ""

# Configuration
INSTANCE_TYPE="p4d.24xlarge"
AMI_ID="ami-0edb31eae202f52ce"  # Deep Learning AMI GPU PyTorch 2.9
REGION="us-west-2"
KEY_NAME="snowml-training"
KEY_FILE="snowml-training.pem"
SECURITY_GROUP="default"

echo "Instance Configuration:"
echo "  Type: $INSTANCE_TYPE (8× NVIDIA A100 GPUs)"
echo "  Region: $REGION"
echo "  AMI: Deep Learning AMI GPU PyTorch 2.9"
echo "  Cost: ~\$32.77/hour"
echo "  Estimated training time: ~1 hour"
echo "  Estimated cost: ~\$33"
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "❌ ERROR: SSH key file not found: $KEY_FILE"
    echo "Please make sure $KEY_FILE is in the current directory"
    exit 1
fi

# Confirm launch
read -p "Launch instance? This will cost ~\$33 for 1 hour. (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Launching EC2 instance..."

# Launch instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --region $REGION \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=snowml-exp1b-p4d}]" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ ERROR: Failed to launch instance"
    exit 1
fi

echo "✅ Instance launched: $INSTANCE_ID"
echo ""

# Wait for instance to be running
echo "⏳ Waiting for instance to start (this takes ~2 minutes)..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ Instance running at: $PUBLIC_IP"
echo ""

# Save instance info
cat > instance_info.txt <<EOF
# EC2 Instance Information for Experiment 1B

Instance ID: $INSTANCE_ID
Instance Type: $INSTANCE_TYPE (8× NVIDIA A100 GPUs)
Public IP: $PUBLIC_IP
Region: $REGION
SSH Key: $KEY_FILE

# SSH Command:
ssh -i $KEY_FILE ubuntu@$PUBLIC_IP

# Started: $(date)
# Expected completion: ~1 hour
# Expected cost: ~\$33

# IMPORTANT: Terminate when done to stop charges!
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
EOF

echo "✅ Instance info saved to: instance_info.txt"
echo ""

# Wait for SSH to be ready
echo "⏳ Waiting for SSH to be ready (this takes ~30 seconds)..."
sleep 30

# Test SSH connection
echo "🔌 Testing SSH connection..."
ssh -i $KEY_FILE -o StrictHostKeyChecking=no -o ConnectTimeout=5 ubuntu@$PUBLIC_IP "echo 'SSH connected successfully'" || {
    echo "⏳ SSH not ready yet, waiting another 30 seconds..."
    sleep 30
}

echo ""
echo "========================================="
echo "Setting up training environment"
echo "========================================="
echo ""

# Upload files to instance
echo "📤 Uploading training files..."
scp -i $KEY_FILE -o StrictHostKeyChecking=no \
    train_exp1b_p4d_parallel.py \
    data/exp1b_train_hucs.txt \
    data/exp1b_validation_hucs.txt \
    ubuntu@$PUBLIC_IP:~/

echo "✅ Files uploaded"
echo ""

# Setup and start training
echo "🎯 Starting training on EC2 instance..."
ssh -i $KEY_FILE ubuntu@$PUBLIC_IP <<'ENDSSH'
    set -e

    echo "Setting up environment..."

    # Verify GPUs
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    # Create conda environment (if not exists)
    if ! conda env list | grep -q "^snowml "; then
        echo "Creating conda environment..."
        conda create -n snowml python=3.10 -y
    fi

    # Activate environment and install packages
    source /opt/conda/etc/profile.d/conda.sh
    conda activate snowml

    echo "Installing packages..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install mlflow pandas pyarrow s3fs boto3 numpy scikit-learn

    # Install SnowML package
    if [ ! -d "SnowML" ]; then
        echo "Cloning SnowML repository..."
        git clone https://github.com/DSHydro/SnowML.git
        cd SnowML
        git checkout simran-unified-experiments
        pip install -e .
    fi

    # Create data directory and move split files
    mkdir -p ~/SnowML/data
    mv ~/exp1b_train_hucs.txt ~/SnowML/data/
    mv ~/exp1b_validation_hucs.txt ~/SnowML/data/

    # Move training script
    mv ~/train_exp1b_p4d_parallel.py ~/SnowML/

    # Start training in tmux
    cd ~/SnowML
    tmux new-session -d -s training "conda activate snowml && python -u train_exp1b_p4d_parallel.py 2>&1 | tee training.log"

    echo ""
    echo "✅ Training started in tmux session 'training'"
    echo "To view live progress: tmux attach -t training"
    echo "To detach: Ctrl+B then D"
    echo ""
ENDSSH

echo ""
echo "========================================="
echo "🎉 SUCCESS - Training Started!"
echo "========================================="
echo ""
echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo ""
echo "Training is now running on 8 GPUs in parallel!"
echo "Expected completion: ~1 hour from now"
echo ""
echo "To monitor progress:"
echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "  tmux attach -t training"
echo ""
echo "To check status later:"
echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "  tail -f ~/SnowML/training.log"
echo ""
echo "When training completes:"
echo "  1. Download results:"
echo "     scp -i $KEY_FILE ubuntu@$PUBLIC_IP:~/SnowML/mlflow.db ./"
echo "     scp -i $KEY_FILE ubuntu@$PUBLIC_IP:~/SnowML/exp1b_best_model.json ./"
echo ""
echo "  2. TERMINATE INSTANCE (critical!):"
echo "     aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "Instance info saved to: instance_info.txt"
echo ""
echo "⚠️  IMPORTANT: Don't forget to terminate the instance when done!"
echo "    Cost: \$32.77/hour - will keep charging until terminated"
echo ""
