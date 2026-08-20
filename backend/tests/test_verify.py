"""Chapter 7: programmatic re-execution catches unfaithful traces."""
from app.reasoning.verify import Rung, check_arithmetic, check_structure


def test_correct_arithmetic_passes():
    assert check_arithmetic("we computed 0.18 * 4250 = 765").passed


def test_wrong_arithmetic_is_caught():
    v = check_arithmetic("we computed 0.18 * 4250 = 800")
    assert not v.passed and v.rung is Rung.PROGRAMMATIC
    assert "765" in v.detail


def test_thousands_separators_are_handled():
    assert check_arithmetic("4,250 - 250 = 4,000").passed


def test_division_is_checked():
    assert not check_arithmetic("100 / 4 = 30").passed


def test_non_arithmetic_prose_passes():
    assert check_arithmetic("the vendor was Meridian Cells Ltd").passed


def test_empty_answer_fails_structurally():
    v = check_structure("   ", require_citations=False)
    assert not v.passed and v.rung is Rung.STRUCTURAL


def test_missing_citation_fails_when_required():
    v = check_structure("The window is 30 days.", require_citations=True)
    assert not v.passed and "citation" in v.detail


def test_citation_present_passes():
    v = check_structure("The window is 30 days [policy-refunds-2026#window].",
                        require_citations=True)
    assert v.passed
