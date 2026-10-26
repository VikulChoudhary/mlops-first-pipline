from fastapi.testclient import TestClient
#inport application object defined in main.py
from main import app
#reusable test client for all test functions
client = TestClient(app)

def test_health_endpoint():
    '''
    verify that the health endpoint returns a 200 status code and the expected response'''
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_iris_prediction():
    '''
    verify that the predict endpoint returns a 200 status code and the expected response'''
    payload= {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "class_name" in response.json()
    assert response.json()["class_name"] in ["setosa", "versicolor", "virginica"]
    assert response.json()["prediction"] in [0, 1, 2]