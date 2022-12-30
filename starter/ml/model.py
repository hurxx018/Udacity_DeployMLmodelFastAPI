import os
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd

import joblib

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, precision_score, recall_score

import pytest

from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> RandomForestClassifier:
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model: sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    """

    # model with grid search
    rfc = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 30],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    return cv_rfc.best_estimator_


def compute_model_metrics(
        y: np.ndarray,
        preds: np.ndarray
    ) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: RandomForestClassifier, X: np.ndarray) -> np.ndarray:
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(
        path_directory: str,
        model: RandomForestClassifier,
        encoder: OneHotEncoder,
        lb: LabelBinarizer
    ) -> None:
    """ Save model, encoder, and lb

    Inputs
    ------
    path_directory: str
        a path to a directory where model, encoder, and lb are stored.
    model: sklearn.ensemble.RandomForestClassifier
        ML model
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb: sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer

    Returns
    -------
    None

    """
    with open(os.path.join(path_directory, 'model.pkl'), "wb") as fmodel:
        joblib.dump(model, fmodel)
    with open(os.path.join(path_directory, "encoder.pkl"), "bw") as fencoder:
        joblib.dump(encoder, fencoder)
    with open(os.path.join(path_directory, "lb.pkl"), "bw") as flb:
        joblib.dump(lb, flb)


def load_model(path_directory):
    """ Load model

    Inputs
    ------
    path_directory: str
        a path to a directory where model, encoder, and lb are stored.

    Returns
    -------
    model: sklearn.ensemble.RandomForestClassifier
        ML model
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb: sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer

    """
    with open(os.path.join(path_directory, 'model.pkl'), "rb") as fmodel:
        model = joblib.load(fmodel)
    with open(os.path.join(path_directory, "encoder.pkl"), "rb") as fencoder:
        encoder = joblib.load(fencoder)
    with open(os.path.join(path_directory, "lb.pkl"), "rb") as flb:
        lb = joblib.load(flb)

    return model, encoder, lb


# Testings
@pytest.fixture(scope="session")
def data():
    """ Get data """
    path_data = os.path.join(os.path.dirname(__file__),
        "..", "..", "data", "clean_data.csv")
    df = pd.read_csv(path_data, sep=",")

    return df

@pytest.fixture(scope="session")
def required_columns():
    """ Dictionary of column names and their types"""

    column_names_types = {"age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
        "salary": pd.api.types.is_string_dtype,
    }

    return column_names_types


def test_data_shape(data):
    """ Check whether data does not have null values.
        Then this is a valid test.
    """
    assert data.shape == data.dropna().shape, "Dropping null chages shape"


def test_column_presence_and_type(data, required_columns):
    """ Check whetehr data includes required columns
    """
    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys())), \
        "Data misses columns"


def test_column_type(data, required_columns):
    """ Check whether data contains correct value types.
    """
    #
    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(data[col_name]), \
            f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):
    """ Check wheter each categorical column contains correct values.
    """

    known_classes = {
        "workclass": ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
            'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'],
        "education": ['Bachelors', 'HS-grad', '11th', 'Masters', '9th',
            'Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc',
            'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],
        "marital-status": ['Never-married', 'Married-civ-spouse', 'Divorced',
            'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
        "occupation": ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
            'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
            'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair',
            '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'],
        "relationship": ['Not-in-family', 'Husband', 'Wife', 'Own-child',
            'Unmarried', 'Other-relative'],
        "race": ['White', 'Black', 'Asian-Pac-Islander',
            'Amer-Indian-Eskimo', 'Other'],
        "sex": ['Male', 'Female'],
        "native-country": ['United-States', 'Cuba', 'Jamaica', 'India',
            'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England',
            'Canada', 'Germany', 'Iran', 'Philippines', 'Italy',
            'Poland', 'Columbia', 'Cambodia', 'Thailand', 'Ecuador',
            'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic',
            'El-Salvador', 'France', 'Guatemala', 'China', 'Japan',
            'Yugoslavia', 'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland',
            'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam',
            'Hong', 'Ireland', 'Hungary', 'Holand-Netherlands'],
        "salary": ['<=50K', '>50K']
    }

    for col_name, classes_tmp in known_classes.items():
        for i, v in enumerate(data[col_name].isin(classes_tmp)):
            if v == False:
                print(i, data[col_name].iloc[i])
        assert data[col_name].isin(classes_tmp).all(), \
            f"all values in column {col_name} are not contained in the list known_classes"


def test_column_ranges(data):
    """ Check wheter each continuous colum contains values
        in the range of (minimum, maximum)
    """

    range_values = {
        "age": (17, 90),
        "fnlgt": (10000, 1500000),
        "education-num": (0, 20),
        "capital-gain": (0, 100000),
        "capital-loss": (0, 10000),
        "hours-per-week": (0, 100)
    }

    for col_name, (minmum, maximum) in range_values.items():
        assert data[col_name].dropna().between(minmum, maximum).all(), \
            f"values in {col_name} are not in the range of ({minmum}, {maximum})"





# Data slicing
def get_performance(
        model: RandomForestClassifier,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float, float]:
    """ Calculate model metrics

    Inputs
    ------
    model: sklearn.ensemble.RandomForestClassifier
        ML model
    X: np.ndarray
        input features
    y: np.ndarray
        Known labels, binarized.

    Returns
    -------
    precision: float
    recall: float
    fbeta: float

    """
    pred = inference(model, X)
    return compute_model_metrics(y, preds=pred)


def performance_slice_on_feature(
        model,
        data: pd.DataFrame,
        feature: str,
        categorical_features: List[str],
        label: str,
        encoder: OneHotEncoder,
        lb: LabelBinarizer,
    ) -> Dict[str, Tuple[float, float, float]]:
    """ Calculate metrics on slices of data for a given feature

    Inputs
    ------
    model: sklearn.ensemble.RandomForestClassifier
        ML model
    data: pd.DataFrame
        Data on which data slicing is performe
    feature: str
        feature on which slicing is performed
    categorical_featurs: List[str]
        list of categorical feature
    label: str
        label name
    encoder: sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder
    lb: sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer


    Returns
    -------
    result: Dict[str, Tuple[float, float, float]]
        feature_value:(precision, recall, fbeta) for a given feature

    """


    result = {}
    for value in data[feature].unique():
        data_tmp = data[data[feature] == value]
        x_tmp, y_tmp, _, _ = process_data(data_tmp, categorical_features=categorical_features,
            label=label, training=False, encoder=encoder, lb=lb)
        result[value] = get_performance(model, x_tmp, y_tmp)

    return result


