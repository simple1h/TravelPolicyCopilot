from __future__ import annotations

from src.evals.llm_judge import _clamp_score, _extract_json_payload


def test_extract_json_payload_from_wrapped_text():
    payload = _extract_json_payload("result: {\"relevance\": 4, \"clarity\": 5}")
    assert payload["relevance"] == 4
    assert payload["clarity"] == 5


def test_clamp_score_bounds():
    assert _clamp_score(0) == 1.0
    assert _clamp_score(9) == 5.0
    assert _clamp_score(3.25) == 3.25
