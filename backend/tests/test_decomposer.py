"""Chapter 2: a malformed decomposition must fail LOUDLY at the boundary."""
import pytest
from pydantic import ValidationError

from app.reasoning.decomposer import Decomposition, SubProblem


def _sp(i, deps=(), kind="infer"):
    return SubProblem(id=i, question=f"q{i}", kind=kind, depends_on=list(deps))


def test_cycle_is_rejected():
    with pytest.raises(ValidationError, match="cycle"):
        Decomposition(sub_problems=[_sp("s1", ["s2"]), _sp("s2", ["s1"])])


def test_unknown_dependency_is_rejected():
    with pytest.raises(ValidationError, match="unknown node"):
        Decomposition(sub_problems=[_sp("s1", ["s9"])])


def test_self_dependency_is_rejected():
    with pytest.raises(ValidationError, match="itself"):
        Decomposition(sub_problems=[_sp("s1", ["s1"])])


def test_independent_nodes_share_a_layer():
    """The parallelism win: independent nodes must NOT be serialised."""
    dec = Decomposition(sub_problems=[_sp("s1"), _sp("s2"),
                                      _sp("s3", ["s1", "s2"])])
    layers = dec.execution_layers()
    assert [len(l) for l in layers] == [2, 1]
    assert {n.id for n in layers[0]} == {"s1", "s2"}


def test_chain_produces_one_node_per_layer():
    dec = Decomposition(sub_problems=[_sp("s1"), _sp("s2", ["s1"]),
                                      _sp("s3", ["s2"])])
    assert [len(l) for l in dec.execution_layers()] == [1, 1, 1]
