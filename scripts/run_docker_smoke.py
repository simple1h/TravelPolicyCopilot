from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_docker_smoke() -> dict[str, object]:
    api = httpx.get("http://localhost:8000/health", timeout=5.0)
    ui = httpx.get("http://localhost:8501", timeout=5.0)
    mock = httpx.get("http://localhost:8081/health", timeout=5.0)
    qdrant = httpx.get("http://localhost:6333/collections", timeout=5.0)

    for name, response in (("api", api), ("ui", ui), ("mock", mock), ("qdrant", qdrant)):
        if response.status_code >= 400:
            raise RuntimeError(f"{name} smoke check failed with status={response.status_code}")

    message_response = None
    last_error = "unknown"
    for _ in range(3):
        try:
            message_response = httpx.post(
                "http://localhost:8000/v1/agent/message",
                timeout=25.0,
                json={
                    "employee_id": "EMP-001",
                    "message": "Мне нужно 18 сентября попасть из Казани в Москву и вернуться на следующий день.",
                },
            )
            break
        except httpx.TimeoutException as exc:
            last_error = str(exc)
            time.sleep(1.0)
    if message_response is None:
        raise RuntimeError(f"Agent API smoke request timed out: {last_error}")
    message_response.raise_for_status()
    payload = message_response.json()
    if payload.get("current_step") not in {"present_recommendations", "ask_clarification"}:
        raise RuntimeError(f"Unexpected agent step from API: {payload.get('current_step')}")

    return {
        "api_health": api.json(),
        "mock_health": mock.json(),
        "agent_step": payload.get("current_step"),
        "trace_id_present": bool(payload.get("trace_id")),
    }


def main() -> None:
    print(json.dumps(run_docker_smoke(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
