import glob
import os
import re
from functools import lru_cache

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"

MAX_PAGE_CHARS = 2000
CHUNK_OVERLAP = 200

MATCH_TABLE_FILES = {
    "IPL_2008_2026_All_Matches.pdf",
    "BBL_2011_2025_All_Matches.pdf",
}

LEAGUE_WINNERS_FILES = {
    "Cricket_Leagues_Winners_Full.pdf",
}

DATE_RE = re.compile(r"^\d{2} [A-Za-z]{3} \d{4}$")
SEASON_HEADER_RE = re.compile(r"^(IPL|BBL) \S+\s+\(\d+ matches\)$")
LEAGUE_HEADER_YEARS_RE = re.compile(r"^\d{4}(?:-\d{2})? to \d{4}(?:-\d{2})?$")
SEASON_LINE_RE = re.compile(r"^\d{4}(-\d{2})?$")


def _league_header_name(line):
    """"<League Name> — 2013 to 2025" -> "<League Name>", or None if line
    isn't a league-winners section header. Tries each dash variant rather
    than one backtracking regex, since PDF text extraction can yield a
    plain hyphen, en dash, or em dash depending on the source font."""

    for dash in (" — ", " – ", " - "):
        if dash not in line:
            continue

        name, _, years = line.rpartition(dash)

        if LEAGUE_HEADER_YEARS_RE.match(years.strip()):
            return name.strip()

    return None


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


def _pdf_paths():
    return sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))


def _discover_pdfs():
    documents = []

    for path in _pdf_paths():
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0].replace("_", " ")
        tournament = re.sub(r"\s+", " ", stem).strip()
        documents.append((path, tournament))

    return documents


def list_tournaments():
    tournaments = []

    for path, tournament in _discover_pdfs():
        if os.path.basename(path) in LEAGUE_WINNERS_FILES:
            tournaments.extend(_league_names(path))
        else:
            tournaments.append(tournament)

    return tournaments


def match_table_tournaments():
    """Tournaments backed by a match-table PDF (the ones with a parseable
    'Team A vs Team B' line per record, so list_teams() works on them)."""

    return [
        tournament
        for path, tournament in _discover_pdfs()
        if os.path.basename(path) in MATCH_TABLE_FILES
    ]


@lru_cache(maxsize=None)
def _parse_league_winners(path):
    """Split a league-winners PDF into per-league sections, each made of
    one block per season plus a trailing "Notes" block.

    Unlike every other PDF here, this one file bundles several
    independent leagues under their own headers, and a league's section
    can span a page break with no repeated header - so this reads the
    whole document as one line stream instead of processing page by page.

    Returns a list of {"league": name, "blocks": [{"label", "page",
    "lines"}, ...]} dicts, in document order.
    """

    reader = PdfReader(path)

    lines = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append((line, page_number))

    leagues = []
    current_league = None
    current_block = None

    for line, page_number in lines:
        league_name = _league_header_name(line)

        if league_name:
            current_league = {"league": league_name, "blocks": []}
            leagues.append(current_league)
            current_block = None
            continue

        if current_league is None:
            continue  # stray text before the first league header

        if SEASON_LINE_RE.match(line):
            current_block = {"label": line, "page": page_number, "lines": []}
            current_league["blocks"].append(current_block)
            continue

        if line.startswith("Note:"):
            current_block = {"label": "Notes", "page": page_number, "lines": []}
            current_league["blocks"].append(current_block)

        if current_block is None:
            continue  # stray line before the first season/notes block

        current_block["lines"].append(line)

    return leagues


def _league_names(path):
    return [league["league"] for league in _parse_league_winners(path)]


def league_winner_tournaments():
    """Tournaments backed by a league-winners PDF section (the ones with
    parseable Winner:/Runner-up: lines per season, so list_teams() works
    on them)."""

    names = []

    for path, _ in _discover_pdfs():
        if os.path.basename(path) in LEAGUE_WINNERS_FILES:
            names.extend(_league_names(path))

    return names


def roster_tournaments():
    """Every tournament with a deterministic, complete team roster
    available independent of retrieval - see list_teams()."""

    return match_table_tournaments() + league_winner_tournaments()


YEAR_MENTION_RE = re.compile(r"\b(?:19|20)\d{2}(?:-\d{2})?\b")


def years_mentioned(text):
    """4-digit years (or a "2025-26"-style season) mentioned in free text,
    for pulling an exact league-winners season by year out of a question."""

    return YEAR_MENTION_RE.findall(text)


def season_context(tournament, year):
    """The exact season block for a league-winners tournament and year,
    bypassing similarity search entirely.

    Every season's chunk is a few near-identical lines (Winner/Runner-up/
    Player of the Series) differing mainly in names and one 4-digit year,
    which a general-purpose sentence embedding doesn't reliably
    distinguish - "who won the CPL in 2023" can easily retrieve the 2022
    or 2024 chunk instead. A season is a single, cheap, unambiguous
    lookup once we already know the (tournament, year) pair, so there's
    no reason to leave it to retrieval rank.

    Returns None if that tournament has no block for that year.
    """

    for path in _pdf_paths():
        if os.path.basename(path) not in LEAGUE_WINNERS_FILES:
            continue

        for league in _parse_league_winners(path):
            if league["league"] != tournament:
                continue

            for block in league["blocks"]:
                if block["label"] == year:
                    return "\n".join(block["lines"])

    return None


def _teams_from_match_tables(tournament):
    teams = set()

    for path, doc_tournament in _discover_pdfs():
        if os.path.basename(path) not in MATCH_TABLE_FILES:
            continue
        if tournament and doc_tournament != tournament:
            continue

        reader = PdfReader(path)

        for page in reader.pages:
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            for line in _clean_table_lines(text):
                if " vs " not in line:
                    continue

                team_a, team_b = line.split(" vs ", 1)
                teams.add(team_a.strip())
                teams.add(team_b.strip())

    return teams


def _looks_like_team_name(name):
    return bool(name) and not name.startswith(("(", "—", "–", "-"))


def _teams_from_league_winners(tournament):
    teams = set()

    for path in _pdf_paths():
        if os.path.basename(path) not in LEAGUE_WINNERS_FILES:
            continue

        for league in _parse_league_winners(path):
            if tournament and league["league"] != tournament:
                continue

            for block in league["blocks"]:
                for line in block["lines"]:
                    for prefix in ("Winner: ", "Runner-up: "):
                        if line.startswith(prefix):
                            name = line[len(prefix):].strip()
                            if _looks_like_team_name(name):
                                teams.add(name)

    return teams


@lru_cache(maxsize=None)
def list_teams(tournament=None):
    """Distinct team names appearing in match-table records and/or
    league-winners records.

    Similarity search only ever surfaces a handful of individual chunks,
    so "which teams played in X" questions were only ever seeing
    whichever few teams happened to be in the top-k results. This scans
    every source row directly - independent of retrieval - so the list is
    always complete.

    tournament=None scans every roster-eligible tournament.
    """

    teams = _teams_from_match_tables(tournament) | _teams_from_league_winners(tournament)

    truncated = {
        team for team in teams
        if any(other.startswith(team + " ") for other in teams if other != team)
    }

    return sorted(teams - truncated)


def _load_league_winners(path):
    """Build one Document per season (plus one per league's trailing
    Notes block), each tagged with that league's own name instead of the
    filename - this file bundles several independent leagues, unlike
    every other PDF here."""

    documents = []

    for league in _parse_league_winners(path):
        name = league["league"]

        for block in league["blocks"]:
            content = "\n".join(block["lines"])

            if not content:
                continue

            documents.append(
                Document(
                    page_content=f"[{name} - {block['label']}]\n{content}",
                    metadata={"tournament": name, "page": block["page"]},
                )
            )

    return documents


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
        if os.path.basename(path) in LEAGUE_WINNERS_FILES:
            league_docs = _load_league_winners(path)
            leagues = _league_names(path)

            print(f"Loading league winners (PDF)... {len(leagues)} leagues: {', '.join(leagues)}")
            print(f"  {len(league_docs)} season/notes chunks\n")

            documents.extend(league_docs)
            continue

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
