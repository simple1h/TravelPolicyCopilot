from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_healthcheck_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_agent_message_returns_clarification_step():
    response = client.post(
        "/v1/agent/message",
        json={"employee_id": "EMP-001", "message": "Нужна командировка в СПБ"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_step"] == "ask_clarification"
    assert "origin_city" in payload["missing_fields"]
