import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("C:\\Users\\Nishitha reddy\\OneDrive\\Desktop\\project\\data.csv")

# Filter only useful transaction types
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT', 'DEBIT', 'PAYMENT'])]

# Encode transaction type as number
df['type'] = df['type'].map({'TRANSFER': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3})

# Select features and label
features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type']
X = df[features]
y = df['isFraud']

# Normalize numeric features
scaler = MinMaxScaler()
X[features[:-1]] = scaler.fit_transform(X[features[:-1]])

# Save scaler for later use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with class balance
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save trained model
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(model, f)
