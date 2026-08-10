# Cricket RAG

A Retrieval-Augmented Generation chat app over cricket PDFs — IPL and BBL match-by-match
records plus six ICC tournament media guides (World Cups, Champions Trophy, WTC finals).
Built with Streamlit, Chroma, HuggingFace embeddings, and Groq.

**Live app:** [cricketrag-56tonikbhenwajbqs9exzj.streamlit.app](https://cricketrag-56tonikbhenwajbqs9exzj.streamlit.app/)

## What it covers

| Source | Type | Notes |
|---|---|---|
| IPL 2008–2026 match results | Tabular PDF | Chunked per match |
| BBL 2011–2025 match results | Tabular PDF | Chunked per match |
| ICC Men's Cricket World Cup 2023 | Media guide | Chunked per page |
| ICC Men's T20 World Cup 2024 | Media guide | Chunked per page |
| ICC Women's T20 World Cup 2024 | Media guide | Chunked per page |
| ICC World Test Championship Final 2021 | Media guide | Chunked per page |
| ICC World Test Championship Final 2025 | Media guide | Chunked per page |
| ICC Cricket World Cup 2019 media guide | Media guide | Chunked per page |

All PDFs are processed as unstructured text — no fields are extracted (season, winner,
player-of-the-match, etc. are not separate metadata). IPL/BBL are split into one chunk per
match instead of one chunk per page so a specific match stays findable; everything else is
chunked per page.

### Known limitation

Because nothing is extracted into structured fields, questions that need an exact
date/season pinpointed (e.g. "who won the IPL final in 2024") are unreliable — semantic
search over near-identical match records can't reliably disambiguate one exact year from
another. Broad narrative questions (tournament history, team/player profiles, "who won the
World Cup") work well since that content is embedded in actual prose.

## Architecture

```
data/*.pdf
    │
    ▼
pdf_loader.py          → extracts + chunks every PDF (per-match for IPL/BBL, per-page for guides)
    │
    ▼
embeddings.py           → sentence-transformers/all-MiniLM-L6-v2
    │
    ▼
vector_store.py         → Chroma, persisted to chroma_db/
    │
    ▼
retriever.py            → similarity search, optional tournament filter
    │
    ▼
prompt.py + llm.py      → Groq (llama-3.3-70b-versatile) answers from retrieved context
    │
    ▼
app.py                  → Streamlit chat UI
```

`build_index.py` builds the index standalone (useful for a one-off/background build before
first launch); `app.py` builds it automatically on first run if `chroma_db/` is empty, with
a progress bar. Indexing is resumable — an interrupted run picks up from the last document
persisted instead of starting over. A corrupted store (e.g. from an unclean shutdown
mid-write) is detected and rebuilt automatically.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Groq API key:

```
GROQ_API_KEY=your-groq-api-key-here
```

Then:

```bash
streamlit run app.py
```

First launch builds `chroma_db/` from the PDFs in `data/` (a minute or two); every launch
after that is instant.

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub (already done if you're reading this from there).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at this repo,
   branch `main`, main file `app.py`.
3. Add your Groq key under **Advanced settings → Secrets**:
   ```
   GROQ_API_KEY = "your-actual-groq-api-key"
   ```
4. Deploy. First boot rebuilds the index from the PDFs in `data/`.

## Project layout

```
app.py                 Streamlit UI
main.py                CLI entry point (same pipeline, terminal Q&A loop)
build_index.py         Standalone index builder
pdf_loader.py           PDF discovery, chunking, per-match splitting for IPL/BBL
embeddings.py           Embedding model
vector_store.py         Chroma persistence, resumable/corruption-safe indexing
retriever.py            Similarity search with optional tournament filter
prompt.py               Prompt template
llm.py                  Groq LLM client
data/                   Source PDFs (indexed into the RAG)
docs/                   Reference PDFs not wired into the RAG
```
