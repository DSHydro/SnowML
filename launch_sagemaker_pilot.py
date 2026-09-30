#!/usr/bin/env python3
"""
Launch pilot test on SageMaker with g4dn.2xlarge
"""

import boto3
import time
from datetime import datetime

sagemaker = boto3.client('sagemaker', region_name='us-west-2')

# Training job configuration
job_name = f"snowml-pilot-{int(time.time())}"

training_params = {
    'TrainingJobName': job_name,
    'RoleArn': 'arn:aws:iam::677276086662:role/service-role/AmazonSageMaker-ExecutionRole-20250214T075111',
    'AlgorithmSpecification': {
        'TrainingImage': '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker',
        'TrainingInputMode': 'File'
    },
    'ResourceConfig': {
        'InstanceType': 'ml.g4dn.2xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 100
    },
    'StoppingCondition': {
        'MaxRuntimeInSeconds': 7200  # 2 hours max
    },
    'InputDataConfig': [
        {
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://snowml-model-ready/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }
    ],
    'OutputDataConfig': {
        'S3OutputPath': 's3://snowml-model-ready/output/'
    }
}

print("=" * 80)
print("LAUNCHING SAGEMAKER PILOT TEST")
print("=" * 80)
print(f"Job name: {job_name}")
print(f"Instance: ml.g4dn.2xlarge")
print(f"Starting at: {datetime.now()}")
print()

try:
    response = sagemaker.create_training_job(**training_params)
    print("✅ Training job launched!")
    print(f"Job ARN: {response['TrainingJobArn']}")
    print()
    print("Monitor with:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name} --region us-west-2")

except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Trying to get IAM role...")
    iam = boto3.client('iam', region_name='us-west-2')
    roles = iam.list_roles(PathPrefix='/service-role/')
    print("Available SageMaker roles:")
    for role in roles['Roles']:
        if 'SageMaker' in role['RoleName']:
            print(f"  - {role['RoleName']}")
            print(f"    ARN: {role['Arn']}")
