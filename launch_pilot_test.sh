#!/bin/bash
# PILOT TEST - Launch g4dn.2xlarge for testing
# Cost: ~$0.75/hour (~30-45 min = ~$0.50 total)

set -e

echo "=========================================="
echo "PILOT TEST - g4dn.2xlarge"
echo "Testing GPU access and basic functionality"
echo "=========================================="

INSTANCE_TYPE="g4dn.2xlarge"
AMI_ID="ami-00a38ad69f8c44451"  # Ubuntu 22.04 LTS
KEY_NAME="snowml-training"
REGION="us-west-2"
SECURITY_GROUP="default"

echo ""
echo "Configuration:"
echo "  Instance: $INSTANCE_TYPE (1× NVIDIA T4 GPU)"
echo "  Cost: ~\$0.75/hour"
echo "  Duration: ~30-45 minutes"
echo "  Total cost: ~\$0.50"
echo ""
read -p "Press Enter to launch pilot test..."

echo ""
echo "Step 1: Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id $AMI_ID \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --region $REGION \
  --security-groups $SECURITY_GROUP \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=snowml-pilot-test}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

if [ -z "$INSTANCE_ID" ]; then
    echo "❌ Failed to launch instance!"
    exit 1
fi

echo "✅ Instance launched: $INSTANCE_ID"
echo "   Waiting for instance to start..."

aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids $INSTANCE_ID \
  --region $REGION \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "✅ Instance running!"
echo "   Public IP: $PUBLIC_IP"

# Save instance info
cat > instance_info.txt <<EOF
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Instance Type: $INSTANCE_TYPE
Region: $REGION
Launched: $(date)
Purpose: Pilot test
EOF

echo ""
echo "Step 2: Waiting for SSH to be ready (2 minutes)..."
sleep 120

echo ""
echo "Step 3: Uploading files..."
scp -i snowml-training.pem -o StrictHostKeyChecking=no \
  train_pilot_g4dn.py \
  ubuntu@$PUBLIC_IP:~/

scp -i snowml-training.pem -r data ubuntu@$PUBLIC_IP:~/

echo "✅ Files uploaded"

echo ""
echo "Step 4: Setting up environment (this takes ~5 minutes)..."
ssh -i snowml-training.pem ubuntu@$PUBLIC_IP <<'ENDSSH'
set -e

echo "Installing NVIDIA drivers and CUDA..."
sudo apt-get update -qq
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu
echo "Waiting for NVIDIA driver to load..."
sleep 10

echo "Installing system dependencies..."
sudo apt-get install -y python3-pip git

echo "Cloning SnowML repository..."
cd ~
if [ ! -d "SnowML" ]; then
    git clone https://github.com/DSHydro/SnowML.git
fi
cd SnowML

echo "Installing Python packages..."
pip3 install -q --upgrade pip
pip3 install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -q mlflow s3fs pyarrow pandas numpy scikit-learn
pip3 install -q -e .

echo "Moving pilot script..."
mv ~/train_pilot_g4dn.py .
mv ~/data .

echo "Verifying GPU..."
nvidia-smi

echo "✅ Setup complete!"
ENDSSH

echo ""
echo "Step 5: Starting pilot test in tmux..."
ssh -i snowml-training.pem ubuntu@$PUBLIC_IP <<'ENDSSH'
tmux new-session -d -s pilot "cd ~/SnowML && python3 -u train_pilot_g4dn.py 2>&1 | tee pilot_test.log"
echo "✅ Pilot test started in tmux session 'pilot'"
ENDSSH

echo ""
echo "=========================================="
echo "✅ PILOT TEST LAUNCHED!"
echo "=========================================="
echo ""
echo "Instance Info:"
echo "  ID: $INSTANCE_ID"
echo "  IP: $PUBLIC_IP"
echo "  Type: $INSTANCE_TYPE"
echo ""
echo "Monitor progress:"
echo "  ssh -i snowml-training.pem ubuntu@$PUBLIC_IP"
echo "  tail -f ~/SnowML/pilot_test.log"
echo ""
echo "Or attach to live session:"
echo "  ssh -i snowml-training.pem ubuntu@$PUBLIC_IP"
echo "  tmux attach -t pilot"
echo "  (Detach with: Ctrl+B then D)"
echo ""
echo "Expected duration: 30-45 minutes"
echo "Set a timer and check back!"
echo ""
echo "When done, download results and terminate:"
echo "  ./download_pilot_results.sh"
echo ""
