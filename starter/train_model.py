# Script to train machine learning model.
import os
# import sys
import joblib

import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data

from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    performance_slice_on_feature,
    save_model,
    load_model
)


# paths to model and data directories
PATH_TO_MODEL = os.path.join(os.path.dirname(__file__), "..", "model")
PATH_TO_DATA = os.path.join(os.path.dirname(__file__), "..", "data")


# Add code to load in the data.
data = pd.read_csv(os.path.join(PATH_TO_DATA, "clean_data.csv"), sep=",")

label_name = "salary"


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label_name, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label=label_name, training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)


save_model(PATH_TO_MODEL, model, encoder, lb)

# Load model and check its performance
model, encoder, lb = load_model(PATH_TO_MODEL)

y_test_preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, y_test_preds)
print(f"Evaluation on Test: Precision {precision}, Recall {recall}, Fbeta {fbeta}")

with open("slice_output.txt", "w") as ftxt:
    result_slicing = performance_slice_on_feature(
        model, train, cat_features[0], cat_features, label_name, encoder, lb)
    ftxt.write("".join([f"{k}: Precision {v[0]}, Recall {v[1]}, Fbeta {v[2]}\n" for k, v in result_slicing.items()]))