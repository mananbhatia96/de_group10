import logging
import os

from flask import jsonify
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import json

def train(dataset):
    # Ensure the dataset is in numpy array format (or DataFrame)
    # Split into input (X) and output (Y) variables

    X = dataset[:, 0:4]
    Y = dataset[:, 4]
    # Define model
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))  # 4 input features
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10, verbose=0)

    # Evaluate the model
    scores = model.evaluate(X, Y, verbose=0)
    text_out = {
        "accuracy": scores[1],
        "loss": scores[0],
    }
    logging.info(text_out)
    print(text_out)

    # Saving the model
    model_repo = os.environ.get('MODEL_REPO', None)
    print("MODEL_REPO:", os.environ.get('MODEL_REPO'))
    if model_repo:
        file_path = os.path.join(model_repo, "heart_disease_model.h5")
        model.save(file_path)
        logging.info("Saved the model to the location: " + model_repo)
        return jsonify(text_out), 200
    else:
        model.save("heart_disease_model.h5")
        return jsonify({'message': 'The model was saved locally.'}), 200# -*- coding: utf-8 -*-

