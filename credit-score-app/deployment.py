import sagemaker
from sagemaker.sklearn.model import SKLearnModel

import boto3
import os

# Get absolute path to inference.py
current_dir = os.path.dirname(os.path.abspath(__file__))
inference_path = os.path.join(current_dir, "inference.py")

print(boto3.Session().region_name)  # Should output 'us-east-1'

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
# role = sagemaker.get_execution_role()  # IAM role with permissions
role = "arn:aws:iam::611838720246:role/service-role/AmazonSageMaker-ExecutionRole-20240528T222901"

# Configure the model
model = SKLearnModel(
    model_data="s3://credit-score-classification-project/model/model.tar.gz",
    role=role,
    entry_point=inference_path,
    framework_version="1.6.1",  # Match your sklearn version
    py_version="py3",
    name="credit-score-model"
)

# Deploy to a real-time endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  # Instance type
    endpoint_name="credit-score-endpoint",
    wait=True  # Wait for deployment completion
)

print(f"Endpoint ARN: {predictor.endpoint_arn}")
print(f"Endpoint Status: {predictor.endpoint_status}")

import json

# Sample input matching your model's requirements
sample_input = {
    "features": [35, 75000, 6250, 3, 2, 15.5, 1, 5, 2, 0.5, 
                 3, 1, 15000, 25.5, 120, 1, 500, 1000, 3, 25000]
}

# Make prediction
response = predictor.predict(sample_input["features"])
print(f"Prediction: {response}")