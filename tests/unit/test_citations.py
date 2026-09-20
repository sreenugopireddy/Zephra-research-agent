from app.llm.structured_models import FinalAnswer, SynthesizedClaim
from app.schemas.search import ExtractionStatus, FetchedDocument
from app.validation.citations import (
    extract_citation_ids,
    sanitize_answer,
    validate_citations,
)


def docs(*ids, status=ExtractionStatus.success, url_template="https://example.org/{}"):
    return [
        FetchedDocument(
            source_id=sid,
            url=url_template.format(sid.lower()),
            content="body",
            extraction_status=status,
        )
        for sid in ids
    ]


def answer(markdown, claims=(), used=()):
    return FinalAnswer(
        answer_markdown=markdown,
        claims=[SynthesizedClaim(**c) for c in claims],
        source_ids_used=list(used),
    )


def test_extracts_unique_citation_ids_in_order():
    assert extract_citation_ids("a [S2] b [S1] c [S2]") == ["S2", "S1"]


def test_valid_answer_passes():
    validation = validate_citations(
        answer("Fact [S1] and [S2].", [{"text": "x", "source_ids": ["S1"]}]), docs("S1", "S2")
    )
    assert validation.valid and validation.reason == "ok"
    assert validation.unknown_ids == []


def test_unknown_citation_in_prose_is_detected():
    validation = validate_citations(answer("Claim [S9]."), docs("S1"))
    assert not validation.valid
    assert validation.unknown_ids == ["S9"]


def test_unknown_citation_in_claim_source_ids_is_detected():
    validation = validate_citations(
        answer("Text with no marker.", [{"text": "x", "source_ids": ["S7"]}]), docs("S1")
    )
    assert not validation.valid and "S7" in validation.unknown_ids


def test_invented_source_ids_used_is_detected():
    validation = validate_citations(answer("Text [S1].", used=["S1", "S42"]), docs("S1"))
    assert not validation.valid and "S42" in validation.unknown_ids


def test_cited_source_with_invalid_url_is_rejected():
    bad = docs("S1", url_template="file:///etc/{}")
    validation = validate_citations(answer("Claim [S1]."), bad)
    assert not validation.valid and validation.invalid_url_ids == ["S1"]


def test_answer_without_any_citation_fails_when_evidence_exists():
    validation = validate_citations(answer("No citations at all."), docs("S1"))
    assert not validation.valid and "no citations" in validation.reason


def test_partially_extracted_sources_are_flagged_but_not_invalid():
    validation = validate_citations(
        answer("Claim [S1]."), docs("S1", status=ExtractionStatus.partial)
    )
    assert validation.valid
    assert validation.not_fully_inspected_ids == ["S1"]


def test_uncited_claims_are_reported():
    validation = validate_citations(
        answer("Text [S1].", [{"text": "floating", "source_ids": []}]), docs("S1")
    )
    assert validation.uncited_claims == ["floating"]


def test_sanitize_marks_invalid_claims_unsupported_and_keeps_good_ones():
    bad_answer = answer(
        "Good [S1]. Bad [S9].",
        [
            {"text": "good", "source_ids": ["S1"], "confidence": "high"},
            {"text": "bad", "source_ids": ["S9"], "confidence": "high"},
        ],
    )
    validation = validate_citations(bad_answer, docs("S1"))
    text, claims = sanitize_answer(bad_answer, validation)

    assert "[S9]" not in text and "[unsupported]" in text
    assert claims[0].confidence == "high" and claims[0].source_ids == ["S1"]
    assert claims[1].confidence == "unsupported" and claims[1].source_ids == []


def test_sanitize_downgrades_high_confidence_on_partial_sources():
    partial_answer = answer(
        "Claim [S1].", [{"text": "c", "source_ids": ["S1"], "confidence": "high"}]
    )
    validation = validate_citations(partial_answer, docs("S1", status=ExtractionStatus.partial))
    _, claims = sanitize_answer(partial_answer, validation)
    assert claims[0].confidence == "medium"
