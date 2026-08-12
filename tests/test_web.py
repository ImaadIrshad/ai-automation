from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_index_serves_html() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Movie CRS" in response.text


def test_config_reports_active_approach() -> None:
    response = client.get("/config")
    assert response.status_code == 200
    body = response.json()
    assert "approach" in body and "model" in body
