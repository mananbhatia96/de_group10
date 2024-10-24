import json
import os
import pandas as pd
from flask import Flask, request
from resources import model_trainer  # Assuming model_trainer contains your training logic

app = Flask(__name__)
app.config["DEBUG"] = True

# Training API endpoint
@app.route('/training-api/model', methods=['POST'])
def train_models():
    # Get the training input data from the message body (JSON payload)
    training_input = request.get_json()
    
    # Convert JSON data into a pandas DataFrame
    df = pd.read_json(json.dumps(training_input), orient='records')

    # Call the train function from the model_trainer module, passing the training data
    resp = model_trainer.train(df.values)
    
    return resp

# The code within this conditional block will only run if this file is executed as a script
if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0', debug=True)
