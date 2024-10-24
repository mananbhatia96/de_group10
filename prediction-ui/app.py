import json
import os
import logging
import requests
from flask import Flask, request, render_template, jsonify

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/checkheartdisease', methods=['GET', 'POST'])
def check_heart_disease():
    if request.method == "GET":
        print("here1")
        return render_template("input_form_page.html")

    elif request.method == "POST":
        print("here2")

        # Extract form data (only the 6 selected features)
        form_data = {
            'age': int(request.form['age']),
            'cp': int(request.form['cp']),
            'oldpeak': float(request.form['oldpeak']),
            'thalachh': int(request.form['thalachh']),
            'caa': int(request.form['caa']),
            'thall': int(request.form['thall'])
        }

        logging.debug("Prediction input : %s", form_data)

        # Make an HTTP POST request to the prediction API
        # Use the environment variable PREDICTOR_API to get the API URL
        predictor_api_url = os.environ.get('PREDICTOR_API', 'http://127.0.0.1:5001/heart_attack_predictor/')
        print(predictor_api_url)

        # Send the data to the prediction API and get the response
        res = requests.post(predictor_api_url, json=[form_data])
        print(res)

        # Extract the prediction value from the API response
        prediction_value = int(float(res.json()['result']))

        logging.info("Prediction Output : %s", prediction_value)
        print(prediction_value)

        # Render the response page with the prediction result
        return render_template("response_page.html",
                               prediction_variable=prediction_value)

    else:
        return jsonify(message="Method Not Allowed"), 405  # Restrict any other HTTP methods (e.g., PUT, DELETE)

# The code within this conditional block will only run if this Python file is executed as a script
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
