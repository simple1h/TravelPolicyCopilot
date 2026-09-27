from __future__ import annotations

from datetime import UTC, datetime
from time import sleep
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from src.mock_travel_api.schemas import TravelOptionsSearchRequest, TravelOptionsSearchResponse
from src.mock_travel_api.service import get_catalog_date_range, get_catalog_size, search_offers

app = FastAPI(title="TravelCopilot Mock Travel API", version="0.1.0")


@app.get("/health")
def healthcheck() -> dict[str, str | int]:
    date_from, date_to = get_catalog_date_range()
    payload: dict[str, str | int] = {
        "status": "ok",
        "service": "mock-travel-api",
        "catalog_size": get_catalog_size(),
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if date_from is not None:
        payload["catalog_date_from"] = date_from.isoformat()
    if date_to is not None:
        payload["catalog_date_to"] = date_to.isoformat()
    return payload


@app.post("/v1/options/search", response_model=TravelOptionsSearchResponse)
def search_options(payload: TravelOptionsSearchRequest) -> TravelOptionsSearchResponse:
    if payload.scenario_mode == "error_4xx":
        raise HTTPException(status_code=422, detail="Scenario forced validation-like error.")
    if payload.scenario_mode == "error_5xx":
        raise HTTPException(status_code=503, detail="Scenario forced upstream failure.")
    if payload.scenario_mode == "timeout":
        sleep(payload.timeout_seconds or 3.0)
        raise HTTPException(status_code=504, detail="Scenario forced timeout.")
    if payload.scenario_mode == "empty":
        return TravelOptionsSearchResponse(
            search_id=f"SEARCH-{uuid4().hex[:12]}",
            total_options=0,
            options=[],
        )

    options = search_offers(payload)
    return TravelOptionsSearchResponse(
        search_id=f"SEARCH-{uuid4().hex[:12]}",
        total_options=len(options),
        options=options,
    )
