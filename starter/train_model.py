# Script to train machine learning model.
# import os
# import sys

import pandas as pd

from typing import List, Tuple

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data

from ml.model import train_model, compute_model_metrics


# Add code to load in the data.
data = pd.read_csv("")

label_name = "salary"
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, stratify=label_name)

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
    test, categorical_features=cat_features, label=label_name, training=False
)

# Train and save a model.
train_model(X_train, y_train)