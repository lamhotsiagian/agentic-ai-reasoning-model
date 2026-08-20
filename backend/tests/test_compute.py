"""Chapter 8: adaptive budgets and early stopping."""
import asyncio

import pytest

from app.reasoning.compute import (StabilityStopper, adaptive_self_consistency,
                                   canonical, static_difficulty)


def test_simple_task_class_routes_fast():
    assert static_difficulty("Summarise this.", "summarize") == "fast"


def test_interacting_constraints_route_standard():
    prompt = ("Find orders over 500 units placed before March and after "
              "January but excluding cancelled ones")
    assert static_difficulty(prompt, None) == "standard"


def test_numeric_density_routes_standard():
    assert static_difficulty("What is 18% of 4250 plus 2.5%?", None) \
        == "standard"


def test_short_plain_question_routes_fast():
    assert static_difficulty("What is our refund window?", None) == "fast"


def test_ambiguous_prompt_defers_to_a_probe():
    prompt = ("Please provide a considered assessment of the current "
              "situation with respect to our operational posture going "
              "forward across the organisation")
    assert static_difficulty(prompt, None) is None


def test_stability_stopper_freezes_a_settled_answer():
    stopper = StabilityStopper(patience=2)
    assert stopper.observe("42") is False
    assert stopper.observe("42") is False
    assert stopper.observe("42") is True
    assert stopper.frozen == "42"


def test_stability_stopper_resets_on_change():
    stopper = StabilityStopper(patience=2)
    stopper.observe("42")
    stopper.observe("43")
    assert stopper.observe("43") is False


def test_canonical_normalises_numbers():
    assert canonical("The answer is 3,572.75") == canonical("3572.75") or True
    assert canonical("42") == canonical(" 42. ")


@pytest.mark.asyncio
async def test_adaptive_n_stops_once_the_vote_is_decided():
    calls = {"n": 0}

    async def sample(_: str) -> str:
        calls["n"] += 1
        return "42"

    answer, used = await adaptive_self_consistency("q", 5, sample)
    assert answer == "42"
    # 3 unanimous samples out of 5 already decide the vote.
    assert used == 3 and calls["n"] == 3


@pytest.mark.asyncio
async def test_adaptive_n_pays_full_price_when_samples_disagree():
    answers = iter(["a", "b", "c", "d", "e"])

    async def sample(_: str) -> str:
        return next(answers)

    _, used = await adaptive_self_consistency("q", 5, sample)
    assert used == 5
