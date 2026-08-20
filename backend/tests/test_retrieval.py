"""Chapter 6: RRF and evidence accumulation."""
from app.reasoning.retrieval.loop import EvidenceSet, Passage, rrf_fuse


def p(doc, chunk, source, score=0.0):
    return Passage(doc_id=doc, chunk_id=chunk, text=f"{doc}/{chunk}",
                   source=source, score=score)


def test_rrf_rewards_agreement_across_channels():
    """A passage ranked well by two channels must outrank a channel leader."""
    dense = [p("a", "1", "vector"), p("b", "1", "vector")]
    sparse = [p("b", "1", "sparse"), p("c", "1", "sparse")]
    fused = rrf_fuse([dense, sparse])
    assert fused[0].key == ("b", "1")


def test_rrf_needs_no_score_normalisation():
    """Cosine and BM25 magnitudes are incomparable; ranks are not."""
    dense = [p("a", "1", "vector", score=0.71)]
    sparse = [p("a", "1", "sparse", score=94.2)]
    fused = rrf_fuse([dense, sparse])
    assert len(fused) == 1 and fused[0].score < 1.0


def test_evidence_set_deduplicates_on_key():
    ev = EvidenceSet()
    assert ev.add([p("a", "1", "vector")]) == 1
    assert ev.add([p("a", "1", "sparse")]) == 0
    assert len(ev.passages) == 1


def test_citations_are_stable_and_unique():
    ev = EvidenceSet()
    ev.add([p("a", "1", "vector"), p("a", "2", "vector")])
    assert ev.citations() == ["a#1", "a#2"]
