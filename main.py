import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Creating the Dataset based on the project schema
data = {
    'Year': [2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025, 2025],
    'CustomerId': [15634602, 15647311, 15619304, 15701354, 15737888, 15574012, 15592531, 15656148, 15792365, 15592389],
    'Surname': ['Hargrave', 'Hill', 'Onio', 'Boni', 'Mitchell', 'Chu', 'Bartlett', 'Obinna', 'He', 'H?'],
    'CreditScore': [619, 608, 502, 699, 850, 645, 822, 376, 501, 684],
    'Geography': ['France', 'Spain', 'France', 'France', 'Spain', 'Spain', 'France', 'Germany', 'France', 'France'],
    'Gender': ['Female', 'Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male'],
    'Age': [42, 41, 42, 39, 43, 44, 50, 29, 44, 27],
    'Tenure': [2, 1, 8, 1, 2, 8, 7, 4, 4, 2],
    'Balance': [0.00, 83807.86, 159660.80, 0.00, 125510.80, 113755.80, 0.00, 115046.70, 142051.10, 134603.90],
    'NumOfProducts': [1, 1, 3, 2, 1, 2, 2, 4, 2, 1],
    'HasCrCard': [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    'IsActiveMember': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
    'EstimatedSalary': [101348.90, 112542.60, 113931.60, 93826.63, 79084.10, 149756.70, 10062.80, 119346.90, 74940.50, 71725.73],
    'Exited': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

print("--- Initial Dataset Sample ---")
print(df.head())

# 2. Data Preprocessing & Feature Engineering
X = df.drop(columns=['CustomerId', 'Surname', 'Exited'])
y = df['Exited']

# Encode categorical variables (Geography, Gender)
label_enc_geo = LabelEncoder()
label_enc_gen = LabelEncoder()

X['Geography'] = label_enc_geo.fit_transform(X['Geography'])
X['Gender'] = label_enc_gen.fit_transform(X['Gender'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training (Random Forest Classifier for Retention / Churn Prediction)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Predictions and Evaluation
y_pred = model.predict(X_test_scaled)

print("\n--- Model Evaluation Output ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance Analysis (Product Utilization & Engagement Insights)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n--- Feature Importance for Retention Strategy ---")
print(feature_importances.sort_values(ascending=False))
