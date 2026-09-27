from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from src.agent.state import TravelAgentState


@dataclass
class SessionRecord:
    session_id: str
    employee_id: str
    state: TravelAgentState


class InMemorySessionStore:
    def __init__(self) -> None:
        self._items: dict[str, SessionRecord] = {}
        self._lock = Lock()

    def create(self, state: TravelAgentState) -> SessionRecord:
        session_id = str(state.get("session_id") or "")
        employee_id = str(state.get("employee_id") or "")
        record = SessionRecord(session_id=session_id, employee_id=employee_id, state=state)
        with self._lock:
            self._items[session_id] = record
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._items.get(session_id)

    def update_state(self, session_id: str, state: TravelAgentState) -> SessionRecord | None:
        with self._lock:
            record = self._items.get(session_id)
            if record is None:
                return None
            updated = SessionRecord(session_id=session_id, employee_id=record.employee_id, state=state)
            self._items[session_id] = updated
            return updated


session_store = InMemorySessionStore()
