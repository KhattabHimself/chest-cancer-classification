from fastapi.testclient import TestClient
from src.inference.api import app

# Initialize the test client with your FastAPI app
client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}