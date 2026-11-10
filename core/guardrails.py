
"""Pure-Python guardrail helpers (no LangChain imports, easy to unit test)."""

import re

NO_ANSWER = "I couldn't find this in the available documents"

# Matches citations such as [1], [2, 3], or [1, 2, 3]
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def resolve_access_rank(level: str, levels: dict) -> int:
    """Fail closed: unknown level -> lowest rank (public)."""
    return levels.get((level or "").strip().lower(), 0)


def normalize_and_validate_citations(answer: str, n_passages: int):
    """
    - Expands "[1, 2]" into "[1][2]".
    - Removes citations that point to non-existent passages.
    - Returns (clean_answer, sorted list of valid cited passage numbers).
    """
    used = set()

    def _sub(match):
        nums = [int(x.strip()) for x in match.group(1).split(",")]
        valid = [n for n in nums if 1 <= n <= n_passages]

        used.update(valid)

        return "".join(f"[{n}]" for n in valid)

    cleaned = _CITATION.sub(_sub, answer).strip()
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    return cleaned, sorted(used)


def is_refusal(answer: str) -> bool:
    return NO_ANSWER.lower() in answer.lower()
