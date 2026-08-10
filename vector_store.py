from langchain_chroma import Chroma

PERSIST_DIR = "chroma_db"


def get_vector_store(embeddings):
    """Open the persisted Chroma store (creates an empty one on first run)."""

    return Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def safe_count(vector_db):
    """Like count(), but returns None instead of raising on a corrupted store."""

    try:
        return vector_db._collection.count()
    except Exception:
        return None


def is_populated(vector_db):
    count = safe_count(vector_db)
    return bool(count)


def index_documents(vector_db, documents, batch_size=1000, on_progress=None, start_at=0):
    """Embed and add documents in batches, reporting progress along the way.

    start_at lets a resumed run skip documents already persisted from a
    previous, interrupted attempt (e.g. the machine went to sleep mid-run).
    """

    total = len(documents)

    for start in range(start_at, total, batch_size):

        batch = documents[start:start + batch_size]

        vector_db.add_documents(batch)

        if on_progress:
            on_progress(min(start + batch_size, total), total)
