# app.py
from flask import Flask, request, jsonify, render_template
import requests
import os
import boto3
import json
from normalize import normalize

application = Flask(__name__)
application.config['CORS_HEADERS'] = 'Content-Type'

@application.route('/')
def index():
    return render_template('index2.html')

@application.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        LOAN_TYPE_OPTIONS = [
            "Auto Loan",
            "Other",
            "Payday Loan",
            "Debt Consolidation Loan",
            "Credit-Builder Loan",
            "Personal Loan",
            "Home Equity Loan",
            "Student Loan",
            "Mortgage Loan"
            ]
        
        data = request.get_json()
        loan_types = data.get('loan_types', [])
        loan_encoding = [1 if loan in loan_types else 0 for loan in LOAN_TYPE_OPTIONS]
        # print(len(data) + len(loan_encoding)-1)

        credit_mix_map = {
            'Good': 2,
            'Standard': 1,
            'Bad': 0
        }

        payment_behavior_map ={
            'High_spent_Large_value_payments' : 3,
            'High_spent_Medium_value_payments': 2,
            'High_spent_Small_value_payments': 1,
            'Unknown': 0,
            'Low_spent_Large_value_payments': -1,
            'Low_spent_Medium_value_payments': -2,
            'Low_spent_Small_value_payments': -3
        }
        
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
            credit_mix_map.get(data.get('credit_mix','Bad')),
            float(data.get('outstanding_debt', 0)),
            float(data.get('credit_utilization', 0)),
            float(data.get('credit_history_age', 0)),
            1 if data.get('pay_minimum') == 'Yes' else 0,
            float(data.get('total_emi', 0)),
            float(data.get('amount_invested', 0)),
            payment_behavior_map.get(data.get('payment_behavior','Unknown')),
            loan_encoding,
            float(data.get('avg_monthly_balance', 0))
        ]

        features = normalize(features)

        # Create SageMaker request payload
        payload = {
            "features": features
        }

        # SageMaker endpoint URL
        endpoint_url = 'https://u25fgqo4h9.execute-api.us-east-1.amazonaws.com/Prod/predict'

        # Send request to SageMaker
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            endpoint_url,
            # auth=awsauth,
            json=payload,
            headers=headers
        )

        # Parse response
        if response.status_code == 200:
            body = response.json().get('body')
            res = json.loads(body)
            prediction = res.get("prediction", 'Unknown')
            return jsonify({'result': prediction})
        else:
            return jsonify({'error': 'Prediction failed'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    application.run(host="0.0.0.0",port=8000)