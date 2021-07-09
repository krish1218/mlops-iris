from fastapi.testclient import TestClient
from main import app
from datetime import datetime

# test to check the correct functioning of the /ping route
def test_ping():
    with TestClient(app) as client:
        response = client.get("/ping")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"ping": "pong","timestamp":datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}


# test to check if Iris Virginica is classified correctly
def test_pred_virginica():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 3,
        "sepal_width": 5,
        "petal_length": 3.2,
        "petal_width": 4.4
    }
    with TestClient(app) as client:
        response = client.post("/predict_flower", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Virginica","timestamp":datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}

#Task2 Writing test cases
def test_appstatus():
    with TestClient(app) as client:
        response = client.get("/appstatus")
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"app": "running successfully","timestamp":datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}

def test_pred_setosa():
    # defining a sample payload for the testcase
    payload = {
        "sepal_length": 5.7,
        "sepal_width": 3.8,
        "petal_length": 1.7,
        "petal_width": 0.3
    }
    with TestClient(app) as client:
        response = client.post("/pred_setosa", json=payload)
        # asserting the correct response is received
        assert response.status_code == 200
        assert response.json() == {"flower_class": "Iris Setosa","timestamp":datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}