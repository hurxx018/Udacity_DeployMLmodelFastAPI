import os

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from starter.ml.data import process_data
from starter.ml.model import load_model, inference


alias_features = {
    "age": "age",
    "workclass": "workclass",
    "fnlgt": "fnlgt",
    "education": "education",
    "education_num": "education-num",
    "marital_status": "marital-status",
    "occupation": "occupation",
    "relationship": "relationship",
    "race": "race",
    "sex": "sex",
    "capital_gain": "capital-gain",
    "capital_loss": "captial-loss",
    "hours_per_week": "hours-per-week",
    "native_country": "native-country",
}


class Value(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 25,
                "workclass": "State-gov",
                "fnlgt": 50000,
                "education": "Bachelors",
                "education_num": 17,
                "marital_status": "Never-married",
                "occupation": "Sales",
                "relationship": "Not-in-family",
                "race": "Other",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb

    model, encoder, lb = load_model(os.path.join(".", "model"))


@app.get("/")
async def greeting():
    return {"greeting": "Welcome"}


@app.post("/{path}")
async def get_inference(path: int, query: int, body: Value):

    tmp = {alias_features[key]: [value] for key, value in dict(body).items()}

    data = pd.DataFrame(tmp)

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

    x, _, _, _ = process_data(
        data, cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    pred = inference(model, x).tolist()

    return {"predictions": pred}
