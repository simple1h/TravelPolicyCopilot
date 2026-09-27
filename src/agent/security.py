from __future__ import annotations

import re

# Direct prompt-injection attempts in user input (MVP heuristic guardrail).
_PROMPT_INJECTION_PATTERN = re.compile(
    r"(?i)"
    r"(?:"
    r"игнориру(?:й|йте)\s+(?:корпоративн|правил|инструкц|политик|ограничен)"
    r"|ignore\s+(?:all\s+)?(?:corporate\s+)?(?:rules|instructions|policy)"
    r"|обой(?:ди|дите)\s+(?:правил|политик|ограничен)"
    r"|bypass\s+(?:the\s+)?(?:rules|policy|restrictions)"
    r"|забудь\s+(?:все\s+)?(?:правил|инструкц|ограничен)"
    r"|forget\s+(?:all\s+)?(?:rules|instructions|restrictions)"
    r"|раскрой\s+(?:системн|скрыт)"
    r"|reveal\s+(?:the\s+)?(?:system\s+)?prompt"
    r")",
)


def is_prompt_injection_attempt(message: str) -> bool:
    normalized = message.strip()
    if not normalized:
        return False
    return bool(_PROMPT_INJECTION_PATTERN.search(normalized))


def build_prompt_injection_clarification_answer(*, missing_fields: list[str], question: str | None = None) -> str:
    fields_part = ", ".join(missing_fields) if missing_fields else "параметры поездки"
    clarification = question.strip() if question else f"Нужны уточнения: {fields_part}."
    return (
        "Запрос с попыткой обхода корпоративных правил отклонён. "
        f"{clarification} "
        "Создание заявки без подтверждения невозможно. "
        "Источник: [POLICY-GENERAL:SEC-APPROVAL]."
    )
