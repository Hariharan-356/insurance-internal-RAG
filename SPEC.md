# SPEC: Internal Insurance Knowledge Assistant

## 1. Purpose
An internal-only question-answering assistant over insurance documents (policy FAQs, claims procedures, underwriting guidelines, fraud playbooks). Answers must be grounded in retrieved passages, cited, and restricted to what the user is cleared to read.

### Operational baseline
The project must continue to run in the same verified local setup as it does now: Python virtual environment, local Ollama, local Chroma persistence, and Streamlit UI. The baseline workflow is intentionally simple and must remain stable for everyday use without changing the app's core architecture or breaking the current run commands.

### The `|` operator
The `|` syntax is a LangChain Runnable pattern used to compose prompt, model, and output-processing stages. In this project, the validated baseline continues to use the current manual orchestration flow, and it should keep running as it does now even if the pipe syntax is discussed or optionally adopted later. The important requirement is that the working project remains runnable and stable.

## 2. Technology stack
| Layer | Choice |
|---|---|
| Orchestration | LangChain (`langchain-core`, `langchain-community`, `langchain-text-splitters`) |
| LLM | Ollama, model `qwen2.5:3b`, via `langchain-ollama` `ChatOllama`, temperature 0 |
| Embeddings | `Qwen/Qwen3-Embedding-8B` via `sentence-transformers` (normalized; `query` prompt for queries) |
| Vector DB | Chroma (persistent, cosine space) via `langchain-chroma` |
| Keyword search | BM25Okapi (`rank-bm25`) |
| Splitter | `RecursiveCharacterTextSplitter` (chunk 800, overlap 120, separators `\n\n`, `\n`, `. `, ` `, ``) |
| UI | Streamlit |

## 3. Architecture
```
Ingest:  files (folder = access level) -> loader -> RecursiveCharacterTextSplitter
         -> chunk metadata {source, page, access_level, access_rank, chunk_id}
         -> Qwen3 embeddings -> Chroma

Query:   question + user access level
         -> access_rank resolved (unknown => public)
         -> Vector search (Chroma filter access_rank <= user_rank)  --+
         -> BM25 search (index built only from allowed chunks)       --+-> relevance gates
         -> Reciprocal Rank Fusion (k=60) -> top FINAL_K chunks
         -> no chunks?  => return "I couldn't find this in the available documents" (LLM NOT called)
         -> numbered passages [1..n] -> prompt -> Ollama qwen2.5:3b
         -> citation validation -> answer + passage panel
```

## 4. Access levels
`public(0) < internal(1) < confidential(2) < restricted(3)`. A user at rank R can read chunks with rank <= R.
Level is set by the folder a document is placed in. Unknown folders are skipped on ingest (fail closed). Unknown user levels resolve to `public`.

## 5. Guardrails (requirements)
| ID | Requirement | Implementation |
|---|---|---|
| G1 | Model answers only from numbered passages and cites each as `[n]` | System prompt rules; temperature 0; post-check in `core/guardrails.py` removes citations to non-existent passages and rejects any answer with zero valid citations (returns the refusal message) |
| G2 | If nothing relevant is retrieved, return "I couldn't find this in the available documents" without calling the LLM | Per-retriever thresholds (`VECTOR_MIN_SIM`, `BM25_MIN_SCORE`); `core/rag.py` returns early when hit list is empty |
| G3 | Access level filtered before anything reaches the LLM | Chroma `where access_rank <= N`; BM25 corpus limited to allowed chunks; final defensive re-check on every hit; LLM only ever receives the filtered passages |

## 6. Hybrid retrieval
* Vector: top `VECTOR_K=8`, similarity = 1 - cosine distance, keep if >= `VECTOR_MIN_SIM` (0.35).
* BM25: top `BM25_K=8`, keep if score >= `BM25_MIN_SCORE` (1.0). Tokenizer lowercases and removes stop words.
* Fusion: RRF, `score = sum(1 / (60 + rank))` across both lists, dedupe by `chunk_id`, keep top `FINAL_K=4`.

## 7. Prompt contract
System: answer only from passages; cite `[n]` for each statement; never cite unknown numbers; otherwise reply exactly with the refusal sentence; no guessing.
Human: `Passages:\n[1] (source: ...) text\n[2] ...\n\nQuestion: ...\n\nAnswer:`

## 8. UI (Streamlit)
* Sidebar: access level selector (demo), model info, reload cache, clear chat.
* Main: chat history; answer with `[n]` citations; expander listing passages with source, page, level, vector/BM25 scores, and a cited marker.
* Refusals show whether the LLM was called.

## 9. Project layout
```
insurance_kb_assistant/
  app.py  ingest.py  config.py  requirements.txt  README.md  SPEC.md  .env.example
  core/ embeddings.py vectorstore.py retriever.py guardrails.py rag.py
  data/documents/{public,internal,confidential,restricted}/
  tests/ test_guardrails.py test_rag_flow.py
```

## 10. Acceptance tests
1. Question answerable by an internal doc, asked as `internal` -> grounded answer with valid `[n]`.
2. Same question about a restricted doc asked as `internal` -> refusal, `llm_called = False`, no restricted text appears anywhere.
3. Off-topic question ("capital of France") -> refusal, `llm_called = False`.
4. Model output without citations -> refusal message returned.
5. Model cites `[9]` with only 4 passages -> citation stripped / rejected.
6. Re-running `ingest.py` rebuilds the index without duplicates.

## 11. Non-functional notes
* Fully local: no data leaves the machine (Ollama, local embeddings, local Chroma).
* `Qwen3-Embedding-8B` needs about 16 GB VRAM in fp16; `Qwen3-Embedding-0.6B` is the drop-in dev fallback (re-ingest after switching).
* `qwen2.5:3b` is small: keep FINAL_K low (3 to 5) and chunk size moderate.
* Production hardening: SSO-derived access level, audit log of queries/sources, per-user rate limits.
* Out of scope: multi-turn memory, document upload UI, re-ranking, OCR.
