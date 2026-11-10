"""Verifies the no-retrieval path never calls the LLM (needs langchain-core installed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from core.guardrails import NO_ANSWER
from core.rag import InsuranceAssistant
from core.retriever import Hit, HybridRetriever
from core.vectorstore import get_vector_store


class EmptyRetriever:
    def retrieve(self, q, rank):
        return []


class ExplodingLLM:
    def invoke(self, *_a, **_k):
        raise AssertionError("LLM must not be called when nothing is retrieved")


class OneHitRetriever:
    def retrieve(self, q, rank):
        return [Hit(doc=Document(page_content="Acknowledge claims in 24 hours.",
                                 metadata={"source": "x.md", "access_level": "internal",
                                           "access_rank": 1, "chunk_id": "a"}), rrf_score=1.0)]


class FakeLLM:
    def __init__(self, text): self.text = text
    def invoke(self, *_a, **_k): return self.text


def test_no_hits_no_llm():
    r = InsuranceAssistant(EmptyRetriever(), ExplodingLLM()).answer("anything", "internal")
    assert r.answer == NO_ANSWER and not r.llm_called


def test_uncited_answer_rejected():
    r = InsuranceAssistant(OneHitRetriever(), FakeLLM("Within 24 hours.")).answer("q", "internal")
    assert r.answer == NO_ANSWER and not r.grounded


def test_cited_answer_accepted():
    r = InsuranceAssistant(OneHitRetriever(), FakeLLM("Within 24 hours [1].")).answer("q", "internal")
    assert r.grounded and r.sources[0].cited


class MockAccessRetriever:
    """Simulates HybridRetriever access rank filtering and relevance thresholding."""
    def __init__(self):
        self.corpus = [
            Hit(
                doc=Document(
                    page_content="Claims up to 50,000 can be approved by a Claims Executive.",
                    metadata={"source": "internal/claims_handling_procedure.md", "access_level": "internal", "access_rank": 1, "chunk_id": "c1"}
                ),
                rrf_score=0.9
            ),
            Hit(
                doc=Document(
                    page_content="Any claim with three or more red flags must be referred to SIU.",
                    metadata={"source": "restricted/fraud_investigation_playbook.md", "access_level": "restricted", "access_rank": 3, "chunk_id": "c2"}
                ),
                rrf_score=0.9
            ),
        ]

    def retrieve(self, query: str, max_rank: int):
        q = query.lower()
        if "capital" in q or "weather" in q:
            # Off-topic: below relevance threshold -> no hits
            return []
        results = []
        for hit in self.corpus:
            if hit.doc.metadata["access_rank"] <= max_rank:
                if "approv" in q or "limit" in q:
                    if "approv" in hit.doc.page_content.lower() or "50,000" in hit.doc.page_content:
                        results.append(hit)
                elif "fraud" in q or "referral" in q:
                    if "fraud" in hit.doc.page_content.lower() or "red flags" in hit.doc.page_content.lower():
                        results.append(hit)
        return results


def test_acceptance_1_internal_user_claim_limits():
    retriever = MockAccessRetriever()
    llm = FakeLLM("Claims up to 50,000 can be approved by a Claims Executive [1].")
    assistant = InsuranceAssistant(retriever, llm)
    res = assistant.answer("What are the claim approval limits?", "internal")
    assert res.grounded
    assert "[1]" in res.answer
    assert res.llm_called
    assert res.sources[0].access_level == "internal"


def test_acceptance_2_internal_user_fraud_referral_refused():
    retriever = MockAccessRetriever()
    llm = ExplodingLLM()  # LLM must NOT be called when nothing is retrieved
    assistant = InsuranceAssistant(retriever, llm)
    res = assistant.answer("What are the fraud referral thresholds?", "internal")
    assert res.answer == NO_ANSWER
    assert not res.llm_called
    assert not res.sources
    assert "red flags" not in res.answer


def test_acceptance_3_restricted_user_fraud_referral_accepted():
    retriever = MockAccessRetriever()
    llm = FakeLLM("Any claim with three or more red flags must be referred to SIU [1].")
    assistant = InsuranceAssistant(retriever, llm)
    res = assistant.answer("What are the fraud referral thresholds?", "restricted")
    assert res.grounded
    assert "[1]" in res.answer
    assert res.llm_called
    assert res.sources[0].access_level == "restricted"


def test_acceptance_4_off_topic_question_refused_no_llm():
    retriever = MockAccessRetriever()
    llm = ExplodingLLM()  # LLM must NOT be called
    assistant = InsuranceAssistant(retriever, llm)
    res = assistant.answer("What is the capital of France?", "restricted")
    assert res.answer == NO_ANSWER
    assert not res.llm_called
    assert not res.sources


def test_endorsement_question_is_grounded_even_when_llm_omits_citations():
    class EndorsementRetriever:
        def retrieve(self, q, max_rank):
            return [
                Hit(
                    doc=Document(
                        page_content="An endorsement is a written amendment or rider to an insurance policy.",
                        metadata={"source": "public/auto_policy_faq.md", "access_level": "public", "access_rank": 0, "chunk_id": "e1"},
                    ),
                    rrf_score=0.9,
                )
            ]

    assistant = InsuranceAssistant(EndorsementRetriever(), FakeLLM("An endorsement is a written amendment or rider to an insurance policy."))
    res = assistant.answer("What is an endorsement?", "public")
    assert res.grounded
    assert "endorsement" in res.answer.lower()
    assert "[1]" in res.answer


def test_public_corpus_contains_endorsement_definition():
    store = get_vector_store()
    retriever = HybridRetriever(store)
    hits = retriever.retrieve("what is an endorsement", 0)
    assert hits, "Expected an endorsement entry to be present in the public corpus"
    text = "\n".join(h.doc.page_content.lower() for h in hits)
    assert "endorsement" in text
    assert "amendment" in text or "rider" in text
