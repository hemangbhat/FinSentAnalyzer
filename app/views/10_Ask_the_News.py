"""
Ask the News: retrieval-augmented Q&A over the 2,388 real financial headlines.
"""

import html
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import streamlit as st  # pyre-ignore

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared import (  # pyre-ignore
    active_model,
    empty_state,
    exhibit,
    get_sentiment_color,
    model_label,
    page_header,
    probability_band,
    source_note,
    status_banner,
)

selected_model, predictor = active_model()

if predictor is None:
    st.stop()

page_header(
    "Ask the News",
    "Ask a question about the real financial headlines. The answer is built only from the headlines "
    "retrieved for it, every claim cites its source, and each source is scored by the active model.",
    meta="Corpus <b>zeroshot/twitter-financial-news-sentiment</b> · 2,388 headlines · no dates",
)


def _setting(name: str) -> str | None:
    """Environment variable first, then Streamlit secrets (copied into the environment for the OpenAI client)."""
    value = os.getenv(name)
    if not value:
        try:
            value = st.secrets.get(name)
        except Exception:  # noqa: BLE001  (no secrets file)
            value = None
        if value:
            os.environ[name] = str(value)
    return value


api_key = _setting("OPENAI_API_KEY")
chat_model = _setting("FINSIGHT_LLM_MODEL")
_setting("FINSIGHT_EMBEDDING_MODEL")

try:
    import rag_news  # pyre-ignore

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

if not RAG_AVAILABLE:
    status_banner("LangChain is not installed: <code>pip install langchain-core langchain-openai</code>.", "warning")
    st.stop()

if not api_key or not chat_model:
    missing = [n for n, v in (("OPENAI_API_KEY", api_key), ("FINSIGHT_LLM_MODEL", chat_model)) if not v]
    empty_state(
        "Needs an OpenAI key and a chat model",
        f"Set {' and '.join(f'<code>{m}</code>' for m in missing)} as environment variables or in "
        "<code>.streamlit/secrets.toml</code> (on Streamlit Cloud: app settings → Secrets). "
        "Embedding the corpus once costs roughly 50k tokens; each question is one embedding and one chat call.",
        icon="warning",
    )
    st.stop()

embedding_model = rag_news.get_embedding_model()


@st.cache_resource(show_spinner=False)
def _store(embedding_model: str):
    corpus = rag_news.load_corpus()
    return rag_news.build_index(rag_news.make_embeddings(embedding_model), corpus, embedding_model)


@st.cache_resource(show_spinner=False)
def _answerer(model: str):
    return rag_news.make_answer_runnable(model)


EXAMPLES = [
    "Which companies cut their guidance or outlook?",
    "What is the news saying about oil prices?",
    "Which banks are mentioned with positive news?",
    "What happened to Tesla?",
]


def _use_example() -> None:
    if st.session_state.get("ask_example"):
        st.session_state.ask_q = st.session_state.ask_example


st.pills("Try", EXAMPLES, key="ask_example", on_change=_use_example, label_visibility="collapsed")
with st.form("ask"):
    question = st.text_input("Question", key="ask_q", placeholder="e.g. Which companies raised dividends?")
    k = st.slider("Headlines to retrieve", 3, 15, 8, help="More context helps broad questions and costs more tokens.")
    asked = st.form_submit_button("Ask", type="primary")

if not asked:
    st.stop()
if not question.strip():
    status_banner("Type a question first.", "info")
    st.stop()

try:
    with st.spinner("Searching the headlines…"):
        sources = rag_news.retrieve(_store(embedding_model), question, k=k)
    preds = predictor.predict([s["headline"] for s in sources]) if sources else []
    for s, p in zip(sources, preds):
        s["model_label"] = p["label"]
    with st.spinner("Writing a grounded answer…"):
        result = rag_news.answer_question(question, sources, _answerer(chat_model))
except Exception as exc:  # noqa: BLE001  (network, auth, rate limit: show it, don't crash the page)
    status_banner(f"The OpenAI call failed: {html.escape(str(exc)[:300])}", "error")
    st.stop()

# ── Answer ─────────────────────────────────────────────────────────────────────
exhibit("Answer", f"{chat_model} · temperature 0 · answers only from the {len(sources)} headlines below")
answer_html = re.sub(
    r"\[(\d+)\]",
    lambda m: f"<sup><b>[{m.group(1)}]</b></sup>",
    html.escape(result["answer"]),
)
st.markdown(f"<div style='font-size:1.05rem;line-height:1.6;'>{answer_html}</div>", unsafe_allow_html=True)
if result["citations"]:
    st.caption("Cites " + ", ".join(f"[{c}]" for c in result["citations"]))
elif result["answer"] != rag_news.NOT_FOUND:
    status_banner("The model gave no citations, so treat this answer as unsupported.", "warning")
if result["dropped_citations"]:
    status_banner(
        f"Removed citation(s) {', '.join(map(str, result['dropped_citations']))}: "
        "they pointed at headlines that were not retrieved.",
        "warning",
    )

# ── Sentiment of the evidence ──────────────────────────────────────────────────
mix = rag_news.sentiment_mix([s["model_label"] for s in sources])
exhibit(
    "Sentiment of the retrieved headlines",
    f"Scored by {model_label(selected_model)}. A quick check on whether the answer's tone matches its evidence.",
)
st.markdown(probability_band(mix, max(mix, key=mix.get)), unsafe_allow_html=True)

# ── Sources ────────────────────────────────────────────────────────────────────
exhibit("Sources", "Retrieved by cosine similarity between the question and each headline embedding.")
cited = set(result["citations"])
rows = []
for s in sources:
    mark = "●" if s["n"] in cited else ""
    model_lab, gold = s["model_label"], s["label_name"]
    rows.append(
        f"<tr><td class='n fs-num'>{s['n']} {mark}</td>"
        f"<td class='t'>{html.escape(s['headline'])}</td>"
        f"<td style='color:{get_sentiment_color(model_lab)};'>{model_lab}</td>"
        f"<td style='color:{get_sentiment_color(gold)};'>{gold}</td>"
        f"<td class='n fs-num'>{s['similarity']:.2f}</td></tr>"
    )
st.markdown(
    "<table class='fs-terms'><thead><tr><th>#</th><th class='t'>Headline</th><th>Model</th>"
    "<th>Human label</th><th style='text-align:right;'>Similarity</th></tr></thead>"
    f"<tbody>{''.join(rows)}</tbody></table>",
    unsafe_allow_html=True,
)
source_note(
    f"● cited in the answer. Embeddings: {embedding_model}, index cached in data/cache. "
    "Headlines are tweets without timestamps, so answers describe the dataset, not today's market."
)
