from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import httpx
import streamlit as st

# Ensure `src` package imports work for Streamlit script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.domain.models import TripRequest

EMPLOYEES = {
    "EMP-001": "Обычный сотрудник",
    "EMP-002": "Руководитель",
    "EMP-003": "Региональный сотрудник",
    "EMP-004": "Сотрудник с ограничением мобильности",
    "EMP-005": "Особый бюджетный профиль",
}

DEMO_MESSAGE = (
    "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
    "на встречу к 15:00. Вернуться на следующий день."
)


def _init_session_state() -> None:
    defaults = {
        "messages": [],
        "agent_state": None,
        "session_id": None,
        "employee_id": "EMP-001",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _format_answer(state: dict) -> str:
    return str(state.get("answer") or "Запрос обработан.")


def _api_url(path: str) -> str:
    return f"{settings.agent_api_url.rstrip('/')}{path}"


def _create_session(employee_id: str, message: str) -> dict:
    response = httpx.post(
        _api_url("/v1/agent/sessions"),
        timeout=30.0,
        json={"employee_id": employee_id, "message": message},
    )
    response.raise_for_status()
    return response.json()


def _default_departure_date(trip: TripRequest) -> date:
    if trip.departure_date is not None:
        return trip.departure_date
    if trip.return_date is not None:
        return trip.return_date
    return date.today()


def _default_return_date(trip: TripRequest) -> date:
    if trip.return_date is not None:
        return trip.return_date
    if trip.departure_date is not None:
        return trip.departure_date + timedelta(days=1)
    return date.today() + timedelta(days=1)


def _apply_action(session_id: str, employee_id: str, payload: dict) -> dict:
    request_body = {"employee_id": employee_id, **payload}
    response = httpx.post(
        _api_url(f"/v1/agent/sessions/{session_id}/actions"),
        timeout=30.0,
        json=request_body,
    )
    if response.status_code >= 400:
        detail = response.text
        try:
            detail = str(response.json().get("detail") or detail)
        except ValueError:
            pass
        raise httpx.HTTPStatusError(
            f"{detail} (HTTP {response.status_code})",
            request=response.request,
            response=response,
        )
    return response.json()


def _render_trip_request(trip_payload: dict | TripRequest | None) -> None:
    trip = None
    if isinstance(trip_payload, TripRequest):
        trip = trip_payload
    elif isinstance(trip_payload, dict):
        trip = TripRequest.model_validate(trip_payload)

    if trip is None:
        st.info("Параметры поездки пока не извлечены.")
        return

    cols = st.columns(2)
    cols[0].write(f"**Откуда:** {trip.origin_city or '—'}")
    cols[1].write(f"**Куда:** {trip.destination_city or '—'}")
    cols = st.columns(2)
    cols[0].write(f"**Выезд:** {trip.departure_date or '—'}")
    cols[1].write(f"**Возврат:** {trip.return_date or '—'}")


def _render_recommendations(state: dict) -> None:
    recommendations = state.get("recommendations", [])
    if not recommendations:
        return

    st.subheader("Варианты поездки")
    for recommendation in recommendations:
        label = str(recommendation.get("option_id", "N/A"))
        if recommendation.get("is_recommended"):
            label += " · рекомендуем"

        with st.expander(label, expanded=bool(recommendation.get("is_recommended"))):
            st.write(recommendation.get("explanation", ""))
            st.write(f"**Score:** {recommendation.get('score_total')}")
            st.write(f"**Policy:** {recommendation.get('policy_summary', '')}")
            st.write(f"**Budget:** {recommendation.get('budget_summary', '')}")
            citations = recommendation.get("citations", [])
            if citations:
                st.caption("Нормативные источники:")
                for citation in citations:
                    st.write(
                        f"- [{citation.get('document_id')}:{citation.get('section_id')}] "
                        f"{citation.get('source_path')} (score={citation.get('score')})"
                    )


def _render_selection_panel(state: dict) -> None:
    if state.get("current_step") != "present_recommendations":
        return

    recommendations = state.get("recommendations", [])
    if not recommendations:
        return

    selected = state.get("selected_option_id")
    selected = selected or recommendations[0]["option_id"]
    option_ids = [item["option_id"] for item in recommendations]

    st.subheader("Выбор варианта")
    picked = st.radio(
        "Выберите вариант для согласования",
        options=option_ids,
        index=option_ids.index(selected),
        format_func=lambda option_id: option_id,
        key="selected_option_radio",
    )

    if st.button("Перейти к подтверждению", key="select_option_button"):
        result = _apply_action(
            st.session_state.session_id,
            st.session_state.employee_id,
            {"action": "select_option", "selected_option_id": picked},
        )
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()


def _render_confirmation_panel(state: dict) -> None:
    if state.get("current_step") != "request_confirmation":
        return

    selected_option_id = state.get("selected_option_id")
    if not selected_option_id:
        st.warning("Вариант не выбран. Вернитесь к рекомендациям.")
        return

    st.subheader("Подтверждение")
    st.write(f"Выбран вариант: `{selected_option_id}`")
    confirm_col, change_col, reject_col = st.columns(3)

    if confirm_col.button("Подтвердить", type="primary", key="confirm_button"):
        result = _apply_action(
            st.session_state.session_id,
            st.session_state.employee_id,
            {"action": "confirm"},
        )
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()

    if change_col.button("Изменить выбор", key="change_button"):
        result = _apply_action(
            st.session_state.session_id,
            st.session_state.employee_id,
            {"action": "change_selection"},
        )
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()

    if reject_col.button("Отказаться", key="reject_button"):
        result = _apply_action(
            st.session_state.session_id,
            st.session_state.employee_id,
            {"action": "reject"},
        )
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()


def _render_recovery_panel(state: dict) -> None:
    if state.get("current_step") != "propose_constraint_change":
        return
    st.subheader("Корректировка ограничений")
    st.warning(state.get("recovery_advice") or "Подходящие варианты не найдены.")
    rejected_options = state.get("rejected_options", [])
    if rejected_options:
        st.caption("Отбракованные варианты:")
        for item in rejected_options[:5]:
            st.write(f"- {item.get('option_id')}: {', '.join(item.get('reasons', []))}")

    trip_payload = state.get("trip_request")
    trip = TripRequest.model_validate(trip_payload) if trip_payload else TripRequest()
    default_departure = _default_departure_date(trip)
    default_return = _default_return_date(trip)
    with st.form("recovery_update_form"):
        departure_date = st.date_input("Дата выезда", value=default_departure)
        return_date = st.date_input(
            "Дата возврата",
            value=max(default_return, departure_date),
            min_value=departure_date,
        )
        earliest_departure_hour = st.number_input(
            "Раньше какого часа не вылетать",
            min_value=0,
            max_value=23,
            value=int(trip.earliest_departure_hour or 0),
        )
        latest_arrival_hour = st.number_input(
            "Не позднее какого часа прибыть",
            min_value=0,
            max_value=23,
            value=int(trip.latest_arrival_hour or 23),
        )
        max_segments = st.number_input(
            "Максимум пересадок (сегментов)",
            min_value=1,
            max_value=4,
            value=int(trip.max_segments or 2),
        )
        submitted = st.form_submit_button("Обновить ограничения и повторить")
    if submitted:
        if return_date < departure_date:
            st.error("Дата возврата не может быть раньше даты выезда.")
            return
        patch = {
            "departure_date": departure_date.isoformat(),
            "return_date": return_date.isoformat(),
            "earliest_departure_hour": int(earliest_departure_hour),
            "latest_arrival_hour": int(latest_arrival_hour),
            "max_segments": int(max_segments),
        }
        try:
            result = _apply_action(
                st.session_state.session_id,
                st.session_state.employee_id,
                {"action": "update_constraints", "trip_request_patch": patch},
            )
        except httpx.HTTPStatusError as exc:
            st.error(f"Не удалось обновить ограничения: {exc}")
            return
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()

    retry_col, cancel_col = st.columns(2)
    if retry_col.button("Повторить поиск без изменений", key="retry_search_button"):
        result = _apply_action(
            st.session_state.session_id,
            st.session_state.employee_id,
            {"action": "retry_search"},
        )
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()
    if cancel_col.button("Отменить поиск", key="cancel_search_button"):
        result = _apply_action(
            st.session_state.session_id,
            st.session_state.employee_id,
            {"action": "cancel"},
        )
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()


def _render_result_panel(state: dict) -> None:
    if state.get("current_step") != "show_result":
        return

    st.success(f"Черновик заявки создан: `{state.get('approval_request_id', '—')}`")
    st.write(f"Вариант: `{state.get('selected_option_id', '—')}`")
    if state.get("error") is not None:
        st.error(state["error"].message)


def _render_clarification_form(state: dict) -> None:
    missing_fields = state.get("missing_fields", [])
    if state.get("current_step") != "ask_clarification" or not missing_fields:
        return

    trip_payload = state.get("trip_request")
    trip = TripRequest.model_validate(trip_payload) if trip_payload else TripRequest()
    st.subheader("Уточнить параметры")

    origin_city = trip.origin_city or ""
    destination_city = trip.destination_city or ""
    departure_date = _default_departure_date(trip)
    return_date = _default_return_date(trip)

    with st.form("clarification_form"):
        if "origin_city" in missing_fields:
            origin_city = st.text_input("Город отправления", value=origin_city)
        if "destination_city" in missing_fields:
            destination_city = st.text_input("Город назначения", value=destination_city)
        if "departure_date" in missing_fields:
            departure_date = st.date_input("Дата выезда", value=departure_date)
        if "return_date" in missing_fields:
            min_return = departure_date if "departure_date" in missing_fields else (trip.departure_date or departure_date)
            return_date = st.date_input("Дата возврата", value=max(return_date, min_return), min_value=min_return)

        submitted = st.form_submit_button("Продолжить поиск")

    if submitted:
        patch: dict[str, object] = {}
        if "origin_city" in missing_fields:
            if not origin_city.strip():
                st.error("Укажите город отправления.")
                return
            patch["origin_city"] = origin_city.strip()
        if "destination_city" in missing_fields:
            if not destination_city.strip():
                st.error("Укажите город назначения.")
                return
            patch["destination_city"] = destination_city.strip()
        if "departure_date" in missing_fields:
            patch["departure_date"] = departure_date.isoformat()
        if "return_date" in missing_fields:
            patch["return_date"] = return_date.isoformat()

        effective_departure = departure_date if "departure_date" in missing_fields else (trip.departure_date or departure_date)
        effective_return = return_date if "return_date" in missing_fields else (trip.return_date or return_date)
        if effective_departure and effective_return and effective_return < effective_departure:
            st.error("Дата возврата не может быть раньше даты выезда.")
            return

        try:
            result = _apply_action(
                st.session_state.session_id,
                st.session_state.employee_id,
                {"action": "submit_clarification", "trip_request_patch": patch},
            )
        except httpx.HTTPStatusError as exc:
            st.error(f"Не удалось применить уточнение: {exc}")
            return
        answer = _format_answer(result)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.agent_state = result
        st.rerun()


def _render_debug_panel(state: dict | None) -> None:
    if state is None:
        return

    with st.sidebar.expander("Debug / Observability"):
        st.write(f"**Trace ID:** `{state.get('trace_id', '—')}`")
        st.write(f"**Session ID:** `{state.get('session_id', '—')}`")
        st.write(f"**Current node:** `{state.get('current_step', '—')}`")
        st.write(f"**Missing fields:** {state.get('missing_fields', [])}")

        audit_events = state.get("audit_events", [])
        if audit_events:
            st.caption("Audit events:")
            for event in audit_events[-5:]:
                st.write(f"- {event.get('event_type')} @ {event.get('timestamp')}")


def main() -> None:
    st.set_page_config(page_title="Travel Policy Copilot", page_icon="✈️", layout="wide")
    _init_session_state()

    st.title("Travel Policy Copilot")
    st.caption("MVP: планирование командировки, проверка политики и рекомендации")

    st.sidebar.header("Настройки")
    st.session_state.employee_id = st.sidebar.selectbox(
        "Сотрудник",
        options=list(EMPLOYEES.keys()),
        format_func=lambda employee_id: f"{employee_id} — {EMPLOYEES[employee_id]}",
    )

    if st.sidebar.button("Загрузить demo-сценарий"):
        st.session_state.messages = [{"role": "user", "content": DEMO_MESSAGE}]
        result = _create_session(st.session_state.employee_id, DEMO_MESSAGE)
        st.session_state.session_id = result.get("session_id")
        st.session_state.messages.append({"role": "assistant", "content": _format_answer(result)})
        st.session_state.agent_state = result
        st.rerun()

    if st.sidebar.button("Очистить чат"):
        st.session_state.messages = []
        st.session_state.agent_state = None
        st.session_state.session_id = None
        st.rerun()

    _render_debug_panel(st.session_state.agent_state)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    state = st.session_state.agent_state
    if state is not None:
        st.divider()
        st.subheader("Состояние агента")
        st.write(f"**Шаг:** `{state.get('current_step', '—')}`")
        _render_trip_request(state.get("trip_request"))
        _render_clarification_form(state)
        _render_recommendations(state)
        _render_selection_panel(state)
        _render_confirmation_panel(state)
        _render_recovery_panel(state)
        _render_result_panel(state)

    prompt = st.chat_input("Опишите командировку...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        result = _create_session(st.session_state.employee_id, prompt)
        st.session_state.session_id = result.get("session_id")
        answer = _format_answer(result)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.agent_state = result
        st.rerun()


if __name__ == "__main__":
    main()
