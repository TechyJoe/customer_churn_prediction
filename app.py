from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("churn_prediction_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify({"Churn": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
