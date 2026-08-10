import glob
import os
import re

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"

# Only split a page further if it's unusually long - otherwise one chunk per page.
MAX_PAGE_CHARS = 2000
CHUNK_OVERLAP = 200

# These two are dense match-by-match tables: ~25-30 near-identical records
# packed onto every page. One chunk per page blends them into a single
# embedding that can't distinguish any specific match. So for these files
# only, split each page into one chunk per match instead - still no field
# extraction (no season/winner/player_of_match metadata), just smaller,
# semantically distinct blocks of raw text.
MATCH_TABLE_FILES = {
    "IPL_2008_2026_All_Matches.pdf",
    "BBL_2011_2025_All_Matches.pdf",
}

DATE_RE = re.compile(r"^\d{2} [A-Za-z]{3} \d{4}$")
SEASON_HEADER_RE = re.compile(r"^(IPL|BBL) \S+\s+\(\d+ matches\)$")
TABLE_HEADER_FIELDS = {
    "Date",
    "Match",
    "Winner",
    "Result",
    "Player of the Match",
    "Top Score (Batter)",
    "Best Bowling",
}


def _clean_table_lines(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    return [
        line for line in lines
        if line not in TABLE_HEADER_FIELDS
        and not SEASON_HEADER_RE.match(line)
        and not line.startswith("Notes:")
        and not line.startswith("run-outs are not credited")
        and not line.startswith("Every match:")
        and "Complete Match-by-Match Records" not in line
    ]


def _split_into_match_chunks(text):
    """One chunk per match record, anchored on the date line that starts each.

    Doesn't extract fields - just groups a match's raw lines together so
    the chunk stays small and semantically distinct from every other match.
    """

    lines = _clean_table_lines(text)
    idxs = [i for i, line in enumerate(lines) if DATE_RE.match(line)]

    if not idxs:
        # No recognizable match record on this page (e.g. a title page).
        cleaned = "\n".join(lines)
        return [cleaned] if cleaned else []

    chunks = []

    for n, i in enumerate(idxs):
        end = idxs[n + 1] if n + 1 < len(idxs) else len(lines)
        chunks.append("\n".join(lines[i:end]))

    return chunks


def _discover_pdfs():
    documents = []

    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf"))):
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0].replace("_", " ")
        tournament = re.sub(r"\s+", " ", stem).strip()
        documents.append((path, tournament))

    return documents


def list_tournaments():
    return [tournament for _, tournament in _discover_pdfs()]


def load_documents():
    """Generic PDF loader: one chunk per page, tagged by tournament + page.

    Every PDF in data/ is treated as unstructured text - no table parsing,
    no per-match fields. Only split a page further if it's unusually long.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_PAGE_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
    )

    documents = []

    for path, tournament in _discover_pdfs():
        print(f"Loading {tournament} (PDF)...")

        is_match_table = os.path.basename(path) in MATCH_TABLE_FILES

        reader = PdfReader(path)
        pages_with_text = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                continue  # image-only / blank page, nothing to embed

            pages_with_text += 1

            if is_match_table:
                chunks = _split_into_match_chunks(text)
            else:
                chunks = (
                    [text]
                    if len(text) <= MAX_PAGE_CHARS
                    else splitter.split_text(text)
                )

            for chunk in chunks:
                metadata = {"tournament": tournament, "page": page_number}
                page_content = f"[{tournament} - page {page_number}]\n{chunk}"

                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )

        print(f"  {pages_with_text}/{len(reader.pages)} pages had extractable text\n")

    print(f"Total chunks: {len(documents)}\n")

    return documents
