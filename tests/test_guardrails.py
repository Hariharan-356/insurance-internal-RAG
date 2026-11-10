import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.guardrails import NO_ANSWER, normalize_and_validate_citations, resolve_access_rank, is_refusal

LEVELS = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


def test_valid_citations_kept():
    out, used = normalize_and_validate_citations("Claims are acknowledged in 24 hours [1].", 3)
    assert used == [1] and "[1]" in out


def test_invalid_citation_removed():
    out, used = normalize_and_validate_citations("Fact [7].", 3)
    assert used == [] and "[7]" not in out


def test_grouped_citation_expanded():
    out, used = normalize_and_validate_citations("Fact [1, 2].", 3)
    assert used == [1, 2] and "[1][2]" in out


def test_unknown_level_fails_closed():
    assert resolve_access_rank("superadmin", LEVELS) == 0
    assert resolve_access_rank("Restricted", LEVELS) == 3


def test_refusal_detection():
    assert is_refusal(NO_ANSWER + ".")
