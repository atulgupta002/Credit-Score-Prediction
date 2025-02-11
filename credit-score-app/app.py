# app.py
from flask import Flask, request, jsonify, render_template
import requests
from requests_aws4auth import AWS4Auth
import os
import boto3
import json

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

def get_aws_secret(name,region="us-east-1"):
    session = boto3.Session()
    client = session.client(service_name="secretsmanager",region_name=region)
    response = client.get_secret_value(SecretId=name)
    return response["SecretString"]

AWS_ACCESS_KEY = get_aws_secret("AWS_ACCESS_KEY")
AWS_SECRET_KEY = get_aws_secret("AWS_SECRET_KEY")

REGION = 'us-east-1'  # Update with your region
SAGEMAKER_ENDPOINT = 'your-sagemaker-endpoint-name'

# Initialize AWS Auth
awsauth = AWS4Auth(AWS_ACCESS_KEY, AWS_SECRET_KEY, REGION, 'execute-api')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert multi-select checkboxes to list
        loan_types = request.form.getlist('loan_types[]')
        
        # Prepare features in the order expected by your model
        features = [
            float(data.get('age', 0)),
            float(data.get('annual_income', 0)),
            float(data.get('monthly_salary', 0)),
            float(data.get('bank_accounts', 0)),
            float(data.get('credit_cards', 0)),
            float(data.get('interest_rate', 0)),
            float(data.get('num_loans', 0)),
            float(data.get('avg_delay', 0)),
            float(data.get('avg_delayed_payments', 0)),
            float(data.get('credit_limit_change', 0)),
            float(data.get('credit_inquiries', 0)),
            1 if data.get('credit_mix') == 'Good' else 0,  # Example encoding
            float(data.get('outstanding_debt', 0)),
            float(data.get('credit_utilization', 0)),
            float(data.get('credit_history_age', 0)),
            1 if data.get('pay_minimum') == 'Yes' else 0,
            float(data.get('total_emi', 0)),
            float(data.get('amount_invested', 0)),
            # Add payment behavior encoding
            len(loan_types),
            float(data.get('avg_monthly_balance', 0))
        ]

        # Create SageMaker request payload
        payload = {
            "features": features
        }

        # SageMaker endpoint URL
        endpoint_url = f'https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{SAGEMAKER_ENDPOINT}/invocations'

        # Send request to SageMaker
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            endpoint_url,
            auth=awsauth,
            json=payload,
            headers=headers
        )

        # Parse response
        if response.status_code == 200:
            prediction = response.json().get('result', 'Unknown')
            return jsonify({'result': prediction})
        else:
            return jsonify({'error': 'Prediction failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)