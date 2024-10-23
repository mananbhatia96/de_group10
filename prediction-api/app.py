import os
from flask import Flask, request
from heart_disease_predictor import HeartDiseasePredictor  # Import the HeartDiseasePredictor class

app = Flask(__name__)
app.config["DEBUG"] = True

# Instantiate the HeartDiseasePredictor class
hdp = HeartDiseasePredictor()

# Define the prediction API endpoint
@app.route('/heart_disease_predictor/', methods=['POST'])  # Path of the endpoint. Accept only HTTP POST requests.
def predict_str():
    # Get the prediction input data from the request body (JSON payload)
    prediction_input = request.get_json()
    
    # Call the predict_single_record method from HeartDiseasePredictor class
    return hdp.predict_single_record(prediction_input)

# The code within this conditional block will only run if this file is executed as a script.
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
