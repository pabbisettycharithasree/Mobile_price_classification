from fastapi.testclient import TestClient
from api.main import app
client=TestClient(app)
def test_health_endpoint_responds():
    response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"] in {"healthy","not_ready"}
def test_predict_rejects_missing_features():
    response=client.post("/predict",json={})
    assert response.status_code==422
