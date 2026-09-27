from __future__ import annotations

import re

from src.domain.models import PolicyChunkRef

_INJECTION_PATTERNS = (
    r"ignore\s+(all|any|previous)\s+instructions",
    r"system\s+prompt",
    r"call\s+tool",
    r"override\s+policy",
    r"execute\s+code",
    r"sudo\s+",
    r"rm\s+-rf",
)


def is_malicious_chunk(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in _INJECTION_PATTERNS)


def sanitize_chunk_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    safe_lines: list[str] = []
    for line in lines:
        if is_malicious_chunk(line):
            continue
        safe_lines.append(line)
    return " ".join(safe_lines).strip()


def assemble_policy_context(
    chunks: list[PolicyChunkRef],
    *,
    max_chunks: int = 5,
    max_chars: int = 2400,
) -> tuple[str, list[PolicyChunkRef]]:
    assembled_parts: list[str] = []
    used_chunks: list[PolicyChunkRef] = []
    chars = 0

    for chunk in chunks[:max_chunks]:
        safe_text = sanitize_chunk_text(chunk.chunk_text or "")
        if not safe_text:
            continue
        source = f"[{chunk.document_id}:{chunk.section_id}]"
        part = f"{source} {safe_text}"
        if chars + len(part) > max_chars:
            break
        assembled_parts.append(part)
        used_chunks.append(chunk)
        chars += len(part)

    return "\n".join(assembled_parts), used_chunks
