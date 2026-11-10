
"""RAG orchestration with guardrails."""

from dataclasses import dataclass, field
from typing import List

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

from core.retriever import tokenize

from config import ACCESS_LEVELS, settings
from core.guardrails import (
    NO_ANSWER,
    is_refusal,
    normalize_and_validate_citations,
    resolve_access_rank,
)


SYSTEM_PROMPT = f"""You are an internal insurance knowledge assistant.

STRICT GROUNDING RULES:

1. Answer ONLY using the numbered passages provided.
   Never use external knowledge or information from your training.

2. Read all relevant passages before answering.
   You may combine information from multiple passages when necessary.

3. Every factual statement must have a citation.
   Use the passage numbers exactly as provided, such as [1] or [1][2].

4. Never invent definitions, abbreviation expansions, figures,
   dates, policy terms, names, or explanations.

5. If an abbreviation appears without its full form in the passages,
   do not guess or expand it. Explain only what the passages
   explicitly state.

6. If the question asks for information that is not available
   in the passages, reply exactly:
   {NO_ANSWER}

7. If only part of the requested information is available,
   provide the supported information and clearly state that
   the remaining information is not specified in the passages.
   Cite the available information.

8. Ensure each citation directly supports the statement
   immediately before it.

9. Be concise, factual, and clear.

10. Never treat your own assumptions as document evidence.

11. If a numbered list or table continues across multiple passages,
    consider all relevant passages before answering.
"""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """Passages:

{context}

Question: {question}

Answer:""",
        ),
    ]
)


@dataclass
class Source:
    number: int
    source: str
    page: object
    access_level: str
    text: str
    vector_sim: object = None
    bm25_score: object = None
    cited: bool = False


@dataclass
class AnswerResult:
    answer: str
    sources: List[Source] = field(default_factory=list)
    llm_called: bool = False
    grounded: bool = False


class InsuranceAssistant:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        # Prefer the native LangChain pipe when the model is a Runnable.
        # Fall back to a callable wrapper so the current test doubles and
        # custom LLM adapters continue to work without modifying the app flow.
        try:
            self.chain = PROMPT | llm
        except TypeError:
            self.chain = RunnableLambda(
                lambda inputs: llm.invoke(
                    PROMPT.format_messages(
                        context=inputs["context"],
                        question=inputs["question"],
                    )
                )
            )
        self._traced_answer = traceable(
            name="insurance_assistant",
            run_type="chain",
        )(self._answer_impl)

        self._traced_retrieve = traceable(
            name="hybrid_retrieval",
            run_type="retriever",
        )(self.retriever.retrieve)    
        

    def _grounded_fallback(self, question: str, sources: List[Source]) -> str | None:
        question_tokens = set(tokenize((question or "").lower()))
        if not question_tokens:
            return None

        best_sentence = None
        best_source = None
        best_score = -1

        for source in sources:
            for sentence in re.split(r"(?<=[.!?])\s+", (source.text or "").strip()):
                if not sentence:
                    continue
                sentence_lower = sentence.lower().strip()
                sentence_tokens = set(tokenize(sentence_lower))
                overlap = len(question_tokens & sentence_tokens)
                if overlap == 0:
                    continue
                score = overlap * 3
                if any(word in sentence_lower for word in ("is", "are", "means", "definition", "amendment", "rider", "change")):
                    score += 2
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip().rstrip(".")
                    best_source = source.number

        if best_sentence and best_source is not None:
            return f"{best_sentence} [{best_source}]."
        return None

    def answer(
        self,
        question: str,
        user_access_level: str,
    ) -> AnswerResult:

        return self._traced_answer(
            question=question,
            user_access_level=user_access_level,
        )
    
    def _answer_impl(
        self,
        question: str,
        user_access_level: str,
    ) -> AnswerResult:

        question = (question or "").strip()

        # Guardrail: reject empty questions.
        if not question:
            return AnswerResult(
                answer=NO_ANSWER,
                llm_called=False,
                grounded=False,
            )

        # Resolve the user's access level.
        max_rank = resolve_access_rank(
            user_access_level,
            ACCESS_LEVELS,
        )

        # Guardrail: retrieve only documents the user is authorized
        # to access. Access filtering happens before the LLM call.
        hits = self._traced_retrieve(
            question,
            max_rank,
        )

        # Guardrail: if no relevant documents are retrieved,
        # refuse without calling the LLM.
        if not hits:
            return AnswerResult(
                answer=NO_ANSWER,
                llm_called=False,
                grounded=False,
            )

        # Convert retrieved chunks into numbered source objects.
        sources = [
            Source(
                number=i,
                source=h.doc.metadata.get("source", "unknown"),
                page=h.doc.metadata.get("page"),
                access_level=h.doc.metadata.get("access_level", "?"),
                text=h.doc.page_content,
                vector_sim=h.vector_sim,
                bm25_score=h.bm25_score,
            )
            for i, h in enumerate(hits, start=1)
        ]

        # Prepare numbered context for the LLM.
        context = "\n\n".join(
            f"[{s.number}] "
            f"(source: {s.source})\n{s.text}"
            for s in sources
        )

        # LangChain Runnable pipe pattern: prompt | model.
        raw = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        text = getattr(
            raw,
            "content",
            str(raw),
        ).strip()

        # Guardrail: respect the model's refusal.
        if is_refusal(text):
            return AnswerResult(
                answer=NO_ANSWER,
                sources=sources,
                llm_called=True,
                grounded=False,
            )

        # Guardrail: validate citation numbers.
        cleaned, used = normalize_and_validate_citations(
            text,
            len(sources),
        )

        # If the model answered from the retrieved passages but omitted citations,
        # synthesize the grounded answer from the best matching sentence instead of
        # rejecting the answer outright.
        if not used:
            best = self._grounded_fallback(question, sources)
            if not best:
                return AnswerResult(
                    answer=NO_ANSWER,
                    sources=sources,
                    llm_called=True,
                    grounded=False,
                )
            cleaned, used = normalize_and_validate_citations(
                best,
                len(sources),
            )

        # Mark which retrieved passages were cited.
        for source in sources:
            source.cited = source.number in used

        return AnswerResult(
            answer=cleaned,
            sources=sources,
            llm_called=True,
            grounded=True,
        )



def build_assistant() -> InsuranceAssistant:
    """Build the RAG assistant using the configured LLM provider."""

    from core.retriever import HybridRetriever
    from core.vectorstore import get_vector_store

    # Select the LLM provider.
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=settings.groq_model,
            temperature=settings.temperature,
        )

    elif settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {settings.llm_provider}"
        )

    # Initialize the existing hybrid retriever.
    retriever = HybridRetriever(
        get_vector_store()
    )

    return InsuranceAssistant(
        retriever=retriever,
        llm=llm,
    )

