from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('churn_model.pkl')
contract_encoder = joblib.load('contract_encoder.pkl')
payment_encoder = joblib.load('payment_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get input values from form
            tenure = int(request.form['tenure'])
            monthly = float(request.form['monthly'])
            total = float(request.form['total'])
            support = int(request.form['support'])
            contract = contract_encoder.transform([request.form['contract']])[0]
            payment = payment_encoder.transform([request.form['payment']])[0]

            # Prepare input for model
            input_data = np.array([[tenure, monthly, total, support, contract, payment]])

            # Predict
            prediction = model.predict(input_data)[0]
            result = "Churn" if prediction == 1 else "No Churn"

            return render_template('index.html', prediction=result)

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {e}")

    # GET method or default
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
