import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/checkheartdisease', methods=["GET", "POST"])
def check_heart_disease():
    if request.method == "GET":
        return render_template("input_form_page.html")

    elif request.method == "POST":
        # Collect input values from the form
        prediction_input = [
            {
                "age": int(request.form.get("age")),  # getting input with name = age from HTML form
                "trestbps": int(request.form.get("trestbps")),  # Resting blood pressure
                "chol": int(request.form.get("chol")),  # Cholesterol level
                "thalch": int(request.form.get("thalch"))  # Maximum heart rate achieved
            }
        ]

        logging.debug("Prediction input : %s", prediction_input)

        # Execute the prediction service API by sending an HTTP POST request to the API
        # Use the environment variable to get the value of the heart disease prediction API
        predictor_api_url = os.environ['PREDICTOR_API']
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        prediction_value = res.json()['result']
        logging.info("Prediction Output : %s", prediction_value)

        # Render the response page with the prediction result
        return render_template("response_page.html",
                               prediction_variable=prediction_value)

    else:
        return jsonify(message="Method Not Allowed"), 405  # Restrict any other HTTP methods (e.g., PUT, DELETE)

# The code within this conditional block will only run if this Python file is executed as a script
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
