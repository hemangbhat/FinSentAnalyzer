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
    """Environment variable first, then Streamlit secrets (copied into the environment for the API clients)."""
    value = os.getenv(name)
    if not value:
        try:
            value = st.secrets.get(name)
        except Exception:  # noqa: BLE001  (no secrets file)
            value = None
        if value:
            os.environ[name] = str(value)
    return value


for _name in (
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "FINSIGHT_LLM_PROVIDER",
    "FINSIGHT_LLM_MODEL",
    "FINSIGHT_LLM_TEMPERATURE",
    "FINSIGHT_EMBEDDINGS",
    "FINSIGHT_EMBEDDING_MODEL",
):
    _setting(_name)

try:
    import llm_provider  # pyre-ignore
    import rag_news  # pyre-ignore

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

if not RAG_AVAILABLE:
    status_banner(
        "LangChain is not installed: <code>pip install langchain-core langchain-anthropic langchain-openai</code>.",
        "warning",
    )
    st.stop()

try:
    provider = llm_provider.get_provider()
    chat_model = llm_provider.get_model_name(provider=provider)
    embedding_kind = rag_news.get_embedding_kind()
except ValueError as exc:
    status_banner(html.escape(str(exc)), "warning")
    st.stop()

needed = [llm_provider.KEY_ENV[provider]]
if embedding_kind == "openai" and provider != "openai":
    needed.append("OPENAI_API_KEY")
missing = [n for n in needed if not os.getenv(n)]
if missing:
    empty_state(
        "Needs an API key",
        f"Set {' and '.join(f'<code>{m}</code>' for m in missing)} as an environment variable or in "
        "<code>.streamlit/secrets.toml</code> (on Streamlit Cloud: app settings → Secrets). "
        f"Provider: <b>{provider}</b>, model: <b>{html.escape(chat_model)}</b>. Search runs locally by default, "
        "so each question costs one chat call.",
        icon="warning",
    )
    st.stop()

embedding_model = rag_news.get_embedding_model(kind=embedding_kind)


@st.cache_resource(show_spinner=False)
def _store(embedding_model: str, kind: str):
    corpus = rag_news.load_corpus()
    embeddings = rag_news.make_embeddings(embedding_model, kind=kind, corpus=corpus)
    # Local embeddings rebuild in seconds; only API embeddings are worth saving to disk.
    cache_dir = rag_news.CACHE_DIR if kind == "openai" else None
    return rag_news.build_index(embeddings, corpus, embedding_model, cache_dir=cache_dir)


@st.cache_resource(show_spinner=False)
def _answerer(provider: str, model: str):
    return rag_news.make_answer_runnable(model, provider)


EXAMPLES = [
    "Which companies cut their guidance or outlook?",
    "What is the news saying about oil prices?",
    "Which stocks were upgraded by analysts?",
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
        sources = rag_news.retrieve(_store(embedding_model, embedding_kind), question, k=k)
    preds = predictor.predict([s["headline"] for s in sources]) if sources else []
    for s, p in zip(sources, preds):
        s["model_label"] = p["label"]
    with st.spinner("Writing a grounded answer…"):
        result = rag_news.answer_question(question, sources, _answerer(provider, chat_model))
except Exception as exc:  # noqa: BLE001  (network, auth, rate limit: show it, don't crash the page)
    status_banner(f"The {provider} API call failed: {html.escape(str(exc)[:300])}", "error")
    st.stop()

# ── Answer ─────────────────────────────────────────────────────────────────────
exhibit("Answer", f"{chat_model} ({provider}) · answers only from the {len(sources)} headlines below")
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
    f"● cited in the answer. Embeddings: {embedding_model}. "
    "Headlines are tweets without timestamps, so answers describe the dataset, not today's market."
)
