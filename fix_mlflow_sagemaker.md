# MLflow SageMaker Issue - Solution

## The Problem
Newer MLflow (3.15.1) doesn't recognize SageMaker ARN format.

## The Solution

**On SageMaker, run:**

```bash
# Downgrade to MLflow version that worked with SageMaker ARNs
pip uninstall mlflow -y
pip install mlflow==2.8.0
```

Then the ARN format will work:
```python
params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
```

## Why This Works
- MLflow 2.x had built-in SageMaker support
- MLflow 3.x removed it, expecting users to use plugins
- Previous students used MLflow 2.x
- Downgrading matches their environment

## After Downgrade
Run the test again:
```bash
PYTHONPATH=/home/ec2-user/SageMaker/src:$PYTHONPATH python test_pipeline_small.py
```
