import time
from fastapi.testclient import TestClient
from api.main import app
client=TestClient(app)
def test_health_latency_smoke():
    start=time.perf_counter(); response=client.get("/health"); elapsed=time.perf_counter()-start
    assert response.status_code==200
    assert elapsed < 5.0
