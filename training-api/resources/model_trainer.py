import logging
import os
from flask import jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def train(dataset):
    # Ensure the dataset is in numpy array format (or DataFrame)
    # Split into input (X) and output (Y) variables
    X = dataset[:, 0:6]
    Y = dataset[:, 6]

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Define Random Forest model
    model = RandomForestClassifier(random_state=42)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    predictions = model.predict(X_test)

    # Evaluate the model on the test set
    accuracy = accuracy_score(y_test, predictions)
    text_out = {
        "accuracy": accuracy
    }
    logging.info(text_out)
    print(text_out)

    # Saving the model
    model_repo = os.environ.get('MODEL_REPO', '.')
    if not os.path.exists(model_repo):
        os.makedirs(model_repo)
    model_path = os.path.join(model_repo, 'heart_attack_model.pkl')

    # Save the model using joblib
    joblib.dump(model, model_path)

    return jsonify(text_out)
