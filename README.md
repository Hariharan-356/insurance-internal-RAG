# Insurance Knowledge Assistant

Internal RAG assistant. LangChain + Ollama (`qwen2.5:3b`) + `Qwen/Qwen3-Embedding-8B` + Chroma + Streamlit.
Full details: see `SPEC.md`.

## Current working baseline
This project is intentionally kept in its current working configuration: local Python environment, local Ollama model, local Chroma DB, and a Streamlit UI. The expected runtime flow is the same as the commands below and should continue to work without changing the app architecture or the core orchestration pattern.

### The `|` operator
The `|` syntax is a LangChain Runnable pattern, but this project currently runs on the same validated manual orchestration flow. The application is intentionally kept stable and should continue to run as now after any pipe-style explanation or optional refactor. In other words, the app is working in its current baseline, and the `|` operator is documented here as a LangChain concept rather than a required project change.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama pull qwen2.5:3b          # and make sure `ollama serve` is running
```

## Add documents
Put files (.md, .txt, .pdf) in the folder matching their access level:
```
data/documents/public/ | internal/ | confidential/ | restricted/
```

## Index and run
```bash
python ingest.py
streamlit run app.py
```

## Tests
```bash
pytest -q
```

## Notes
* `Qwen3-Embedding-8B` needs ~16 GB GPU memory (fp16). For a laptop set
  `EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B` and re-run `python ingest.py`.
* If you change the embedding model or chunk settings, always re-ingest.
* The access-level dropdown in the UI is a DEMO. In production, take the level from SSO/IdP claims.
* Tune `VECTOR_MIN_SIM` and `BM25_MIN_SCORE` on real questions; they decide when the assistant refuses.
