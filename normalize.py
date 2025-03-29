scaler_min = [ 1.80000000e+01,  7.00603500e+03,  3.03645417e+02,  0.00000000e+00,
              0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              0.00000000e+00, -6.49000000e+00,  0.00000000e+00,  2.30000000e-01,
              2.00000000e+01,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
              8.86278653e-02]

scaler_max = [1.00000000e+02, 2.41980620e+07, 1.52046333e+04, 1.10000000e+01,
                1.10000000e+01, 3.40000000e+01, 9.00000000e+00, 5.50000000e+01,
                2.80000000e+01, 2.80000000e+01, 1.50000000e+01, 4.99807000e+03,
                4.95645193e+01, 4.04000000e+02, 3.49919748e+02, 1.00000000e+04,
                1.60204052e+03]

scaler_range = [8.20000000e+01, 2.41910560e+07, 1.49009879e+04, 1.10000000e+01,
                1.10000000e+01, 3.30000000e+01, 9.00000000e+00, 5.50000000e+01,
                2.80000000e+01, 3.44900000e+01, 1.50000000e+01, 4.99784000e+03,
                2.95645193e+01, 4.03000000e+02, 3.49919748e+02, 1.00000000e+04,
                1.60195189e+03]


features = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan',
                  'Delay_from_due_date','Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt',
                 'Credit_Utilization_Ratio','Credit_History_Age','Total_EMI_per_month','Amount_invested_monthly','Monthly_Balance']

order = [
    "age","annual_income","monthly_salary","bank_accounts","credit_cards","interest_rate","num_loans","avg_delay","avg_delayed_payments",
    "credit_limit_change","credit_inquiries","credit_mix","outstanding_debt","credit_utilization","credit_history_age","pay_minimum","total_emi",
    "amount_invested","loan_encoding","avg_monthly_balance"]

indices_from_data_to_scale = [0,1,2,3,4,5,6,7,8,9,10,12,13,14,16,17,19]

# Function to normalize user input data using the scaler that was used to normalize the training data.
def normalize(data):

    normalized_data = []
    scale_index = 0

    for i, value in enumerate(data):
        # Our data has a nested list. Here we are simply merging that list with the rest of the data.
        if isinstance(value, list):
            normalized_data.extend(value) 
        elif i in indices_from_data_to_scale:
            # Normalize value
            normalized_value = round(
                (value - scaler_min[scale_index]) /
                (scaler_max[scale_index] - scaler_min[scale_index]), 2
            )
            normalized_data.append(normalized_value)
            scale_index += 1
        else:
            normalized_data.append(value)

    return normalized_data
