from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


TransportType = Literal["flight", "train"]
FareClass = Literal["economy", "premium_economy", "business"]
ScenarioMode = Literal["normal", "empty", "error_4xx", "error_5xx", "timeout"]


class TravelOptionsSearchRequest(BaseModel):
    origin_city: str = Field(min_length=2)
    destination_city: str = Field(min_length=2)
    departure_date: date
    return_date: date
    meeting_datetime: datetime | None = None
    earliest_departure_hour: int | None = Field(default=None, ge=0, le=23)
    latest_arrival_hour: int | None = Field(default=None, ge=0, le=23)
    disallowed_time_windows: list[str] = Field(default_factory=list)
    transport_preferences: list[TransportType] = Field(default_factory=list)
    max_segments: int | None = Field(default=None, ge=1, le=4)
    allowed_fare_classes: list[FareClass] = Field(default_factory=list)
    hotel_required: bool = True
    accessibility_requirements: list[str] = Field(default_factory=list)
    scenario_mode: ScenarioMode = "normal"
    timeout_seconds: float | None = Field(default=None, ge=0.1, le=10.0)

    @model_validator(mode="after")
    def validate_dates(self) -> "TravelOptionsSearchRequest":
        if self.return_date < self.departure_date:
            raise ValueError("return_date must be greater than or equal to departure_date")
        return self


class TravelOffer(BaseModel):
    option_id: str
    transport_type: TransportType
    origin_city: str
    destination_city: str
    outbound_departure: datetime
    outbound_arrival: datetime
    inbound_departure: datetime
    inbound_arrival: datetime
    carrier: str
    fare_class: FareClass
    hotel_name: str | None = None
    hotel_nightly_price: int | None = None
    currency: Literal["RUB"] = "RUB"
    total_price: int
    segments: int
    refundable: bool = False


class TravelOptionsSearchResponse(BaseModel):
    search_id: str
    currency: Literal["RUB"] = "RUB"
    total_options: int
    options: list[TravelOffer]
