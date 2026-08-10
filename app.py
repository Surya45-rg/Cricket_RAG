import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from pdf_loader import load_documents, list_tournaments
from embeddings import get_embedding_model
from vector_store import PERSIST_DIR, get_vector_store, safe_count, index_documents
from retriever import retrieve_documents
from prompt import build_prompt
from llm import get_llm

load_dotenv()

# Bridge Streamlit Cloud secrets into an env var so llm.py works either way.
# st.secrets raises if no secrets.toml exists at all (e.g. local .env-only setups).
if not os.environ.get("GROQ_API_KEY"):
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except FileNotFoundError:
        pass

st.set_page_config(page_title="Cricket RAG", page_icon=":material/sports_cricket:")

# Sample questions per tournament, shown in the sidebar so it's obvious what
# kind of question actually works well for each source. Phrased to avoid the
# known weak spots (exact dates, "the final") on the IPL/BBL match chunks -
# those favor narrative/lookup questions instead.
SAMPLE_QUESTIONS = {
    "All": [
        "Who won the 2023 Cricket World Cup?",
        "Tell me about the history of the IPL.",
        "Which teams play in the Big Bash League?",
    ],
    "IPL": [
        "Tell me about a match between Mumbai Indians and Chennai Super Kings.",
        "Which bowler took a hat-trick in the IPL?",
        "Who has scored a century in an IPL match?",
    ],
    "BBL": [
        "Tell me about a match between Sydney Sixers and Perth Scorchers.",
        "Who scored a fifty in the Big Bash League?",
        "Which team won by the largest margin in the BBL?",
    ],
    "T20_WC_2024_MEN": [
        "Which teams reached the semi-finals of the T20 World Cup 2024?",
        "Tell me about India's squad for the 2024 T20 World Cup.",
        "What was the format of the ICC Men's T20 World Cup 2024?",
    ],
    "T20_WC_2024_WOMEN": [
        "Which teams competed in the Women's T20 World Cup 2024?",
        "Tell me about the venues for the 2024 Women's T20 World Cup.",
    ],
    "CWC_2023": [
        "Who won the 2023 Cricket World Cup?",
        "Which venues hosted the 2023 Cricket World Cup?",
        "Tell me about the format of the 2023 World Cup.",
    ],
    "CHAMPIONS_TROPHY_2025": [
        "Which teams played in the Champions Trophy 2025?",
        "Tell me about the history of the ICC Champions Trophy.",
    ],
    "WTC_FINAL_2021": [
        "Who played in the 2021 World Test Championship final?",
        "Tell me about the venue for the 2021 WTC final.",
    ],
    "WTC_FINAL_2025": [
        "Who played in the 2025 World Test Championship final?",
        "Tell me about the 2025 WTC final.",
    ],
    "CWC_2019_GUIDE": [
        "Tell me about the format of the 2019 Cricket World Cup.",
        "Which teams competed in the 2019 World Cup?",
    ],
}


def classify_tournament(tournament):
    """Map a filename-derived tournament tag to a SAMPLE_QUESTIONS bucket."""

    t = tournament.upper()

    if "IPL" in t:
        return "IPL"
    if "BBL" in t:
        return "BBL"
    if "T20" in t and "WOMEN" in t:
        return "T20_WC_2024_WOMEN"
    if "T20" in t and "MEN" in t:
        return "T20_WC_2024_MEN"
    if "WORLD TEST CHAMPIONSHIP" in t and "2021" in t:
        return "WTC_FINAL_2021"
    if "WORLD TEST CHAMPIONSHIP" in t and "2025" in t:
        return "WTC_FINAL_2025"
    if "CHAMPIONS" in t:
        return "CHAMPIONS_TROPHY_2025"
    if "CRICKET WORLD CUP 2023" in t:
        return "CWC_2023"
    if "2019" in t or "MEDIAGUIDE" in t:
        return "CWC_2019_GUIDE"

    return "All"


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return get_embedding_model()


@st.cache_resource(show_spinner=False)
def load_llm():
    return get_llm()


@st.cache_resource(show_spinner=False)
def load_vector_db():
    """Open the persisted store, building (or resuming/repairing) it as needed."""

    embeddings = load_embeddings()
    vector_db = get_vector_store(embeddings)

    count = safe_count(vector_db)

    if count is None:
        # Store is corrupted (e.g. an earlier index build was interrupted
        # mid-write) - a corrupted index can't be resumed, only rebuilt.
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        vector_db = get_vector_store(embeddings)
        count = 0

    documents = load_documents()
    total = len(documents)

    if count < total:
        progress = st.progress(
            count / total, text=f"Indexing {count}/{total} documents..."
        )

        def report(done, total):
            progress.progress(
                done / total,
                text=f"Indexing {done}/{total} documents...",
            )

        index_documents(vector_db, documents, on_progress=report, start_at=count)
        progress.empty()

    return vector_db


st.title(":material/sports_cricket: Cricket RAG")
st.caption(
    "Ask questions about IPL, BBL, and ICC tournament PDFs (World Cups, "
    "Champions Trophy, WTC finals)."
)

with st.sidebar:
    st.subheader("Filters")
    tournament_options = ["All"] + list_tournaments()
    tournament = st.selectbox("Tournament", tournament_options)
    top_k = st.slider("Results retrieved", min_value=1, max_value=10, value=5)

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Try asking")

    bucket = "All" if tournament == "All" else classify_tournament(tournament)

    for i, question in enumerate(SAMPLE_QUESTIONS.get(bucket, SAMPLE_QUESTIONS["All"])):
        if st.button(question, key=f"sample_{bucket}_{i}", width="stretch"):
            st.session_state.pending_question = question
            st.rerun()

try:
    llm = load_llm()
except ValueError as e:
    st.error(str(e))
    st.stop()

with st.spinner("Preparing the cricket database (first run only)...", show_time=True):
    vector_db = load_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt_text = st.chat_input("Ask about a match, player, or season...")

if not prompt_text and "pending_question" in st.session_state:
    prompt_text = st.session_state.pop("pending_question")

if prompt_text:
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    with st.chat_message("user"):
        st.write(prompt_text)

    with st.chat_message("assistant"):
        with st.spinner("Searching the database..."):
            filter_tournament = None if tournament == "All" else tournament

            results = retrieve_documents(
                vector_db,
                prompt_text,
                k=top_k,
                tournament=filter_tournament,
            )

            context = "\n\n".join(doc.page_content for doc, score in results)
            full_prompt = build_prompt(context, prompt_text)

            response = llm.invoke(full_prompt)

        st.write(response.content)

        if results:
            with st.expander("Sources"):
                for doc, score in results:
                    label = doc.metadata.get("tournament")
                    page = doc.metadata.get("page")

                    if page:
                        label = f"{label} · page {page}"

                    st.caption(f"{label} · score {score:.3f}")
                    st.code(doc.page_content, language="text")

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content}
    )
