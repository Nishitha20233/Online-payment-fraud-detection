from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[x]) for x in ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
    tx_type = int(request.form['type'])  # already encoded

    # Scale input
    scaled_input = scaler.transform([input_data])
    final_input = np.array([list(scaled_input[0]) + [tx_type]])

    # Predict probability
    prob = model.predict_proba(final_input)[0][1]
    prediction = 1 if prob > 0.5 else 0
    result = "FRAUD" if prediction == 1 else "NOT FRAUD"

    return render_template('index.html', prediction_text=f"Transaction is: {result}")

if __name__ == "__main__":
    app.run(debug=True)
