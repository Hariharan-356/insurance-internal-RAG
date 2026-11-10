"""Streamlit UI.  Run:  streamlit run app.py"""
import streamlit as st

from config import ACCESS_LEVELS, settings
from core.guardrails import NO_ANSWER
from core.rag import build_assistant

from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Insurance Knowledge Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for a polished, enterprise look
st.markdown(
    """
    <style>
    .main {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .badge-cited {
        background-color: #d1fae5;
        color: #065f46;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-uncited {
        background-color: #f3f4f6;
        color: #4b5563;
        padding: 2px 8px;
        border-radius: 9999px;
        font-size: 0.8rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models and index...")
def get_assistant():
    return build_assistant()


assistant = get_assistant()

with st.sidebar:
    st.header("🛡️ Session & Access")
    # DEMO ONLY: in production derive this from your SSO / IdP claims, never from a dropdown.
    level = st.selectbox(
        "Your access level",
        list(ACCESS_LEVELS),
        index=1,
        help="In production, this is supplied by the enterprise SSO / OAuth identity token.",
    )
    st.caption("🔒 Documents above this level are strictly filtered out before reaching the LLM.")
    st.divider()

    st.subheader("Model Configuration")

    if settings.llm_provider == "groq":
        st.markdown(f"**LLM:** `Groq - {settings.groq_model}`")
    else:
        st.markdown(f"**LLM:** `Ollama - {settings.ollama_model}`")

    st.markdown(f"**Embeddings:** `{settings.embedding_model}`")
    st.markdown(f"**Chroma Path:** `{settings.chroma_dir}`")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Reload cache", use_container_width=True):
            assistant.retriever.reset_cache()
            st.success("Cache cleared")
    with col2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

st.title("🛡️ Insurance Knowledge Assistant")
st.caption("Internal Insurance Knowledge Assistant with access-restricted RAG, hybrid search, and strict citation guardrails.")

st.session_state.setdefault("history", [])


def render_sources(sources):
    if not sources:
        return

    # Keep only the most relevant passage for display. If the model cited a source,
    # prefer that. Otherwise fall back to the first retrieved result.
    visible = [s for s in sources if s.cited]
    if not visible:
        visible = [sources[0]]
    else:
        visible = visible[:1]

    with st.expander(f"📚 Most Relevant Passage ({len(visible)})", expanded=False):
        for s in visible:
            cited_badge = (
                '<span class="badge-cited">✅ cited</span>'
                if s.cited
                else '<span class="badge-uncited">not cited</span>'
            )
            page_info = f", p.{int(s.page) + 1}" if isinstance(s.page, int) else ""
            st.markdown(
                f"**[{s.number}]** `{s.source}`{page_info} · Level: `{s.access_level}` · {cited_badge}",
                unsafe_allow_html=True,
            )
            scores = []
            if s.vector_sim is not None:
                scores.append(f"Vector Sim: `{s.vector_sim:.3f}`")
            if s.bm25_score is not None:
                scores.append(f"BM25: `{s.bm25_score:.3f}`")
            if scores:
                st.caption(" | ".join(scores))
            st.markdown(f"> {s.text}")
            st.divider()


for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            if turn.get("content") == NO_ANSWER:
                llm_called = turn.get("llm_called", False)
                st.caption("ℹ️ LLM was called, but no grounded answer was found." if llm_called else "ℹ️ LLM was not called (no matching documents retrieved).")
            render_sources(turn.get("sources"))

if question := st.chat_input("Ask about policies, claims handling, underwriting, or fraud..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            result = assistant.answer(question, level)
        st.markdown(result.answer)
        if result.answer == NO_ANSWER:
            st.caption("ℹ️ LLM was called, but no grounded answer was found." if result.llm_called else "ℹ️ LLM was not called (no matching documents retrieved).")
        render_sources(result.sources)
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": result.answer,
            "sources": result.sources,
            "llm_called": result.llm_called,
        }
    )
