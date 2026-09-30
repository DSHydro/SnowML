#!/bin/bash
# Launch g5.xlarge instance for Experiment 1B (sequential training)
# Uses 1× NVIDIA A10G GPU instead of 8× A100

set -e

echo "========================================="
echo "Launch g5.xlarge for Experiment 1B"
echo "========================================="
echo ""

# Configuration
INSTANCE_TYPE="g5.xlarge"
AMI_ID="ami-0edb31eae202f52ce"
REGION="us-west-2"
KEY_NAME="snowml-training"
KEY_FILE="snowml-training.pem"

echo "Instance Configuration:"
echo "  Type: $INSTANCE_TYPE (1× NVIDIA A10G GPU)"
echo "  Region: $REGION"
echo "  AMI: Deep Learning AMI GPU PyTorch 2.9"
echo "  Cost: ~\$1.01/hour"
echo "  Estimated training time: ~6-8 hours (sequential)"
echo "  Estimated cost: ~\$6-8"
echo ""
echo "NOTE: Using sequential training (8 models one after another)"
echo "      p4d.24xlarge quota not yet active - contact professor if urgent"
echo ""

if [ ! -f "$KEY_FILE" ]; then
    echo "❌ ERROR: SSH key file not found: $KEY_FILE"
    exit 1
fi

read -p "Launch instance? This will cost ~\$6-8 for 6-8 hours. (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Launching EC2 instance..."

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --region $REGION \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=snowml-exp1b-g5}]" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100,\"VolumeType\":\"gp3\"}}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ ERROR: Failed to launch instance"
    exit 1
fi

echo "✅ Instance launched: $INSTANCE_ID"
echo ""

echo "⏳ Waiting for instance to start (this takes ~2 minutes)..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "✅ Instance running at: $PUBLIC_IP"
echo ""

cat > instance_info.txt <<EOF
# EC2 Instance Information for Experiment 1B

Instance ID: $INSTANCE_ID
Instance Type: $INSTANCE_TYPE (1× NVIDIA A10G GPU)
Public IP: $PUBLIC_IP
Region: $REGION
SSH Key: $KEY_FILE

# SSH Command:
ssh -i $KEY_FILE ubuntu@$PUBLIC_IP

# Started: $(date)
# Expected completion: ~6-8 hours
# Expected cost: ~\$6-8

# IMPORTANT: Terminate when done!
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
EOF

echo "✅ Instance info saved to: instance_info.txt"
echo ""

echo "⏳ Waiting for SSH to be ready..."
sleep 30

echo "📤 Uploading training files..."
scp -i $KEY_FILE -o StrictHostKeyChecking=no \
    train_exp1b_full.py \
    data/exp1b_train_hucs.txt \
    data/exp1b_validation_hucs.txt \
    ubuntu@$PUBLIC_IP:~/

echo "✅ Files uploaded"
echo ""

echo "🎯 Starting training on EC2 instance..."
ssh -i $KEY_FILE ubuntu@$PUBLIC_IP <<'ENDSSH'
    set -e

    echo "Setting up environment..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch || conda create -n pytorch python=3.10 -y && conda activate pytorch

    echo "Installing packages..."
    pip install -q mlflow pandas pyarrow s3fs boto3 numpy scikit-learn 2>/dev/null || true

    if [ ! -d "SnowML" ]; then
        echo "Cloning SnowML repository..."
        git clone -q https://github.com/DSHydro/SnowML.git
        cd SnowML
        git checkout simran-unified-experiments
        pip install -q -e .
    fi

    mkdir -p ~/SnowML/data
    mv ~/exp1b_train_hucs.txt ~/SnowML/data/
    mv ~/exp1b_validation_hucs.txt ~/SnowML/data/
    mv ~/train_exp1b_full.py ~/SnowML/

    cd ~/SnowML
    tmux new-session -d -s training "conda activate pytorch && python -u train_exp1b_full.py 2>&1 | tee training.log"

    echo ""
    echo "✅ Training started in tmux session 'training'"
ENDSSH

echo ""
echo "========================================="
echo "🎉 SUCCESS - Training Started!"
echo "========================================="
echo ""
echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo ""
echo "Training will take ~6-8 hours (sequential on 1 GPU)"
echo ""
echo "To monitor:"
echo "  ssh -i $KEY_FILE ubuntu@$PUBLIC_IP"
echo "  tmux attach -t training"
echo ""
echo "When complete (in ~6-8 hours):"
echo "  ./download_and_terminate.sh"
echo ""
echo "⚠️  IMPORTANT: Set reminder for 6-8 hours from now!"
echo "    Don't forget to terminate the instance"
echo ""
