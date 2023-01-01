import os
import sys
import json

import requests

from fastapi.testclient import TestClient

t=os.path.join(os.path.dirname(__file__), "..")
sys.path.append(t)
from main import app


client = TestClient(app)

# Test get
def test_client():
    r = client.get("/")
    tmp = dict(r.json())
    assert r.status_code == 200
    assert "greeting" in tmp



def test_post_0():
    path = 1
    query = 2
    data = {
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
        "hours_per_week": 60,
        "native_country": "United-States"
    }

    r = requests.post(
        f"http://127.0.0.1:8000/{path}?query={query}", data=json.dumps(data)
    )

    tmp = dict(r.json())
    print(tmp)
    assert r.status_code == 200
    assert "predictions" in tmp
    assert tmp["predictions"][0] == 0


def test_post_1():
    path = 1
    query = 2
    data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 1000000,
        "education": "Doctorate",
        "education_num": 17,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 10000,
        "capital_loss": 0,
        "hours_per_week": 60,
        "native_country": "United-States"
    }

    r = requests.post(
        f"http://127.0.0.1:8000/{path}?query={query}", data=json.dumps(data)
    )
    tmp = dict(r.json())
    print(tmp)

    assert r.status_code == 200
    assert "predictions" in tmp
    assert tmp["predictions"][0] == 1