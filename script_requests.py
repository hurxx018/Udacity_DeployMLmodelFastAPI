import requests
import json

path = 1
query = 2
data_0 = {
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


data_1 = {
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


if __name__ == "__main__":

    # path_default= f"http://127.0.0.1:8000/{path}?query={query}"

    path_heroku = f"https://udacity-fastapi-app20221231.herokuapp.com/{path}?query={query}"

    for data in [data_0, data_1]:
        r = requests.post(
            path_heroku, data=json.dumps(data)
        )
        print(f"Status Code: {r.status_code}, Result: {r.json()}")
