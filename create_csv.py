import pandas as pd
import numpy as np
import random

print("=" * 60)
print(" 🏭 GENERATING REALISTIC INDIAN SMB CUSTOMER DATASET...")
print("=" * 60)

np.random.seed(42)
random.seed(42)

num_customers = 1000

# Indian Cities & Business Types
cities = ['Pune', 'Mumbai', 'Bangalore', 'Delhi', 'Hyderabad', 'Ahmedabad']
gym_plans = ['Month-to-month', 'One year', 'Two year']
payment_methods = ['Electronic check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check']

customer_ids = [f"PUNE_GYM_{1000 + i}" for i in range(num_customers)]
gender = np.random.choice(['Male', 'Female'], num_customers)
senior = np.random.choice([0, 1], num_customers, p=[0.85, 0.15])
partner = np.random.choice(['Yes', 'No'], num_customers)
dependents = np.random.choice(['Yes', 'No'], num_customers)

# Tenure in months (1 to 60 months)
tenure = np.random.randint(1, 61, num_customers)

# Service features
phone_service = np.random.choice(['Yes', 'No'], num_customers, p=[0.9, 0.1])
multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], num_customers)
internet_service = np.random.choice(['Fiber optic', 'DSL', 'No'], num_customers, p=[0.45, 0.40, 0.15])
online_security = np.random.choice(['Yes', 'No', 'No internet service'], num_customers)
online_backup = np.random.choice(['Yes', 'No', 'No internet service'], num_customers)
device_protection = np.random.choice(['Yes', 'No', 'No internet service'], num_customers)
tech_support = np.random.choice(['Yes', 'No', 'No internet service'], num_customers)
streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], num_customers)
streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], num_customers)

contract = np.random.choice(gym_plans, num_customers, p=[0.55, 0.30, 0.15])
paperless = np.random.choice(['Yes', 'No'], num_customers)
payment = np.random.choice(payment_methods, num_customers)

# Monthly charges in INR (₹600 to ₹3,500)
monthly_charges = np.round(np.random.uniform(600, 3500, num_customers), 2)
total_charges = np.round(tenure * monthly_charges + np.random.uniform(100, 500, num_customers), 2)

# Realistic Churn logic (High charges + Month-to-month + Short tenure = Higher Churn)
churn_prob = []
for i in range(num_customers):
    score = 0.2
    if contract[i] == 'Month-to-month':
        score += 0.3
    if tenure[i] <= 12:
        score += 0.25
    if monthly_charges[i] > 2200:
        score += 0.15
    if tech_support[i] == 'No':
        score += 0.1
    
    churn_prob.append(min(score, 0.85))

churn = ['Yes' if random.random() < p else 'No' for p in churn_prob]

# Create Dataframe
df = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': gender,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn
})

output_filename = "Indian_SMB_Gym_Customers.csv"
df.to_csv(output_filename, index=False)

print(f"[✔] SUCCESS! Created 1,000 Indian SMB Customer Records.")
print(f"[✔] File saved as: '{output_filename}' in your project folder.")
print("=" * 60)