from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

from src.mock_travel_api.schemas import TravelOffer


def _combine(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime.combine(day, time(hour=hour, minute=minute), tzinfo=UTC)


DEFAULT_RETURN_OFFSETS = (1, 2, 3, 5, 7)


def build_catalog(
    base_date: date | None = None,
    *,
    horizon_days: int = 90,
    include_past: bool | None = None,
    return_offsets: tuple[int, ...] = DEFAULT_RETURN_OFFSETS,
) -> list[TravelOffer]:
    today = date.today()
    if base_date is not None:
        seed_date = base_date
        include_past = True if include_past is None else include_past
    else:
        seed_date = today
        include_past = False if include_past is None else include_past
    routes = [
        ("Казань", "Москва"),
        ("Казань", "Санкт-Петербург"),
        ("Екатеринбург", "Москва"),
        ("Новосибирск", "Санкт-Петербург"),
        ("Самара", "Москва"),
    ]
    fares = ["economy", "premium_economy", "business"]
    departure_slots = [6, 8, 10, 12, 15, 18, 22]
    carriers = ["Mock Air", "Volga Wings", "North Rail", "Business Sky"]
    hotels = [
        ("Mock Hotel Center", 5200),
        ("Business Stay", 7600),
        ("Budget Inn", 4300),
        ("Premium Suites", 11200),
    ]

    offers: list[TravelOffer] = []
    idx = 1

    for day_shift in range(horizon_days):
        day = seed_date + timedelta(days=day_shift)
        if not include_past and day < today:
            continue
        for origin_city, destination_city in routes:
            for slot in departure_slots:
                for fare in fares:
                    for return_offset in return_offsets:
                        transport_type = "train" if slot in (6, 22) else "flight"
                        segments = 1 if slot in (8, 10, 12, 15) else 2
                        outbound_departure = _combine(day, slot)
                        outbound_arrival = outbound_departure + timedelta(
                            hours=2 if transport_type == "flight" else 6
                        )
                        inbound_departure = _combine(
                            day + timedelta(days=return_offset),
                            max(6, slot - 1),
                        )
                        inbound_arrival = inbound_departure + timedelta(
                            hours=2 if transport_type == "flight" else 6
                        )

                        hotel_name, hotel_price = hotels[idx % len(hotels)]
                        carrier = carriers[idx % len(carriers)]
                        fare_extra = {"economy": 0, "premium_economy": 7000, "business": 18000}[fare]
                        base_price = 24000 if transport_type == "flight" else 17000
                        total_price = (
                            base_price
                            + fare_extra
                            + segments * 1200
                            + day_shift * 600
                            + max(0, return_offset - 1) * hotel_price
                        )

                        offers.append(
                            TravelOffer(
                                option_id=f"OPT-{idx:04d}",
                                transport_type=transport_type,
                                origin_city=origin_city,
                                destination_city=destination_city,
                                outbound_departure=outbound_departure,
                                outbound_arrival=outbound_arrival,
                                inbound_departure=inbound_departure,
                                inbound_arrival=inbound_arrival,
                                carrier=carrier,
                                fare_class=fare,
                                hotel_name=hotel_name,
                                hotel_nightly_price=hotel_price,
                                total_price=total_price,
                                segments=segments,
                                refundable=fare != "economy",
                            )
                        )
                        idx += 1

    return offers
