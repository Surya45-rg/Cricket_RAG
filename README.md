# Cricket RAG

A Retrieval-Augmented Generation chat app over cricket PDFs — IPL and BBL match-by-match
records, season-by-season winner records for seven T20 franchise leagues, and six ICC
tournament media guides (World Cups, Champions Trophy, WTC finals). Built with Streamlit,
Chroma, HuggingFace embeddings, and Groq.

**Live app:** [cricketrag-56tonikbhenwajbqs9exzj.streamlit.app](https://cricketrag-56tonikbhenwajbqs9exzj.streamlit.app/)

## What it covers

| Source | Type | Notes |
|---|---|---|
| IPL 2008–2026 match results | Tabular PDF | Chunked per match |
| BBL 2011–2025 match results | Tabular PDF | Chunked per match |
| CPL, LPL, PSL, MLC, SA20, The Hundred, ILT20 — winner records | Reference PDF | One file, chunked per league per season |
| ICC Men's Cricket World Cup 2023 | Media guide | Chunked per page |
| ICC Men's T20 World Cup 2024 | Media guide | Chunked per page |
| ICC Women's T20 World Cup 2024 | Media guide | Chunked per page |
| ICC World Test Championship Final 2021 | Media guide | Chunked per page |
| ICC World Test Championship Final 2025 | Media guide | Chunked per page |
| ICC Cricket World Cup 2019 media guide | Media guide | Chunked per page |

Every PDF is processed as unstructured text — no table parsing or field extraction into
metadata beyond `tournament` and `page`. Three different chunking strategies are used
depending on what the source actually looks like:

- **IPL/BBL** are dense match-by-match tables (~25–30 near-identical rows per page) - split
  into one chunk per match instead of one per page, so a specific match stays findable.
- **The winner-records file** bundles seven independent T20 leagues (CPL, LPL, PSL, MLC,
  SA20, The Hundred, ILT20) into one PDF, each under its own header. It's split into one
  chunk per league per season (plus a trailing "Notes" chunk per league for
  rebrands/defunct-franchise context), and - unlike every other file here - each chunk is
  tagged with its *league's* name as the tournament, not the filename. A league's season can
  wrap across a page break in the source PDF with no repeated header, so this is parsed as
  one continuous text stream rather than page by page.
- **Everything else** (the six ICC media guides) is chunked per page.

For all four of IPL, BBL, and the seven winner-records leagues, the exact team roster is
also extracted directly from the source (the "Team A vs Team B" line for IPL/BBL, the
Winner/Runner-up lines for the winner-records leagues) and always included in that
tournament's context - independent of whatever similarity search happens to retrieve. This
exists because top-k retrieval alone was unreliable for "which teams played in X"-style
questions: with hundreds of near-identical match chunks (or a dozen single-season chunks),
the few chunks retrieval happens to surface only ever show whichever teams are in *those*
chunks, not the complete list.

### Known limitation

Because nothing is extracted into structured fields, questions that need an exact
date/season pinpointed in the IPL/BBL match tables (e.g. "who won the IPL final in 2024")
are unreliable — semantic search over hundreds of near-identical match records can't
reliably disambiguate one exact year from another. This doesn't apply to the winner-records
leagues, since each season is already its own small, distinct chunk - "who won the SA20 in
2024?" retrieves cleanly. Broad narrative questions (tournament history, team/player
profiles, "who won the World Cup") also work well since that content is embedded in actual
prose.

## Architecture

```
data/*.pdf
    │
    ▼
pdf_loader.py          → extracts + chunks every PDF (per-match for IPL/BBL, per-season
                          per-league for the winner-records file, per-page for guides)
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
pdf_loader.py           PDF discovery, chunking (per-match for IPL/BBL, per-season per-league
                        for the winner-records file), and roster extraction (list_teams())
embeddings.py           Embedding model
vector_store.py         Chroma persistence, resumable/corruption-safe indexing
retriever.py            Similarity search with optional tournament filter
prompt.py               Prompt template
llm.py                  Groq LLM client
data/                   Source PDFs (indexed into the RAG)
docs/                   Staging area for PDFs not yet wired into the RAG (currently empty)
```
