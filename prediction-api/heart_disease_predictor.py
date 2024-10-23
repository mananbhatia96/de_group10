import json
import os
import pandas as pd
from flask import jsonify
from keras.models import load_model
import logging
from io import StringIO


class HeartDiseasePredictor:
    def __init__(self):
        self.model = None

    def predict_single_record(self, prediction_input):
        logging.debug(prediction_input)
        
        # Load the model if not already loaded
        if self.model is None:
            try:
                model_repo = os.environ['MODEL_REPO']
                file_path = os.path.join(model_repo, "heart_disease_model.h5")  # Update the model name
                self.model = load_model(file_path)
            except KeyError:
                logging.error("MODEL_REPO is undefined. Loading model from local 'heart_disease_model.h5'.")
                self.model = load_model('heart_disease_model.h5')

        # Convert JSON input to DataFrame
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')

        # Make prediction
        y_pred = self.model.predict(df)
        logging.info(y_pred[0])

        # Determine the prediction status (True/False)
        status = (y_pred[0] > 0.5)
        logging.info(type(status[0]))

        # Return the prediction outcome as a JSON response
        return jsonify({'result': str(status[0])}), 200
