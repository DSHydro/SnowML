#!/bin/bash
# Launch AWS EC2 instance for SnowML training
# Instance type: g4dn.xlarge (1 NVIDIA T4 GPU, 4 vCPU, 16GB RAM)
# Cost: ~$0.526/hour

set -e

echo "=========================================="
echo "Launching EC2 Instance for SnowML Training"
echo "=========================================="
echo ""

# Configuration
INSTANCE_TYPE="g4dn.xlarge"
AMI_ID="ami-0f6c7c96b23f4dbd6"  # Deep Learning AMI with PyTorch 2.12
KEY_NAME="snowml-training"
REGION="us-west-2"
SUBNET_ID=""  # Will use default VPC

echo "📋 Configuration:"
echo "   Instance Type: $INSTANCE_TYPE"
echo "   AMI: Deep Learning OSS PyTorch 2.12"
echo "   Region: $REGION"
echo "   Key: $KEY_NAME"
echo ""

# Get default VPC
echo "🔍 Finding default VPC..."
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region $REGION)
echo "   VPC: $VPC_ID"

# Create security group if it doesn't exist
echo ""
echo "🔒 Checking security group..."
SG_NAME="snowml-training-sg"
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" --query 'SecurityGroups[0].GroupId' --output text --region $REGION 2>/dev/null || echo "None")

if [ "$SG_ID" == "None" ]; then
    echo "   Creating new security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name $SG_NAME \
        --description "Security group for SnowML training instances" \
        --vpc-id $VPC_ID \
        --region $REGION \
        --query 'GroupId' \
        --output text)

    # Allow SSH from anywhere (you can restrict this to your IP for better security)
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region $REGION

    echo "   ✅ Created security group: $SG_ID"
else
    echo "   ✅ Using existing security group: $SG_ID"
fi

# Launch instance
echo ""
echo "🚀 Launching instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=snowml-training},{Key=Project,Value=SnowML},{Key=Purpose,Value=Exp1B-Training}]' \
    --region $REGION \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "   ✅ Instance launched: $INSTANCE_ID"

# Wait for instance to be running
echo ""
echo "⏳ Waiting for instance to start (this takes ~30 seconds)..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text \
    --region $REGION)

echo ""
echo "=========================================="
echo "✅ INSTANCE READY!"
echo "=========================================="
echo ""
echo "📍 Instance ID: $INSTANCE_ID"
echo "🌐 Public IP: $PUBLIC_IP"
echo "💰 Cost: ~$0.526/hour"
echo ""
echo "🔑 SSH Connection:"
echo "   ssh -i snowml-training.pem ubuntu@$PUBLIC_IP"
echo ""
echo "⚠️  IMPORTANT: First SSH connection may take 1-2 minutes"
echo "   The instance needs time to initialize"
echo ""
echo "📝 Save this info:"
echo "   Instance ID: $INSTANCE_ID" > instance_info.txt
echo "   Public IP: $PUBLIC_IP" >> instance_info.txt
echo "   SSH Command: ssh -i snowml-training.pem ubuntu@$PUBLIC_IP" >> instance_info.txt
echo "   ✅ Saved to: instance_info.txt"
echo ""
echo "🚀 Next steps:"
echo "   1. Wait 1-2 minutes for instance to fully initialize"
echo "   2. SSH in: ssh -i snowml-training.pem ubuntu@$PUBLIC_IP"
echo "   3. I'll help you set up training once connected"
echo ""
echo "🛑 To stop instance later:"
echo "   aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-west-2"
echo ""
