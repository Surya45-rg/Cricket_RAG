"""One-off script to build the persisted Chroma index from the PDF sources.

Run this once (or let it run in the background) before deploying the
Streamlit app so the app doesn't have to embed everything on first load.

Safe to re-run:
- If chroma_db/ is already fully populated, it's a no-op.
- If a previous run was interrupted partway (e.g. the machine slept),
  it resumes from roughly where it left off instead of starting over.
- If chroma_db/ is corrupted (unreadable), it wipes it and rebuilds
  from scratch, since a corrupted index can't be resumed.
"""

import shutil

from dotenv import load_dotenv

from pdf_loader import load_documents
from embeddings import get_embedding_model
from vector_store import PERSIST_DIR, get_vector_store, safe_count, index_documents

load_dotenv()


def main():
    embeddings = get_embedding_model()
    vector_db = get_vector_store(embeddings)

    count = safe_count(vector_db)

    if count is None:
        print(f"{PERSIST_DIR}/ is corrupted, wiping it and rebuilding from scratch.")
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        vector_db = get_vector_store(embeddings)
        count = 0

    documents = load_documents()
    total = len(documents)

    if count >= total:
        print(f"chroma_db/ already has all {total} documents, nothing to do.")
        return

    if count:
        print(f"Resuming: {count}/{total} documents already indexed.")

    def report(done, total):
        print(f"Indexed {done}/{total} documents", flush=True)

    index_documents(vector_db, documents, on_progress=report, start_at=count)
    print("Indexing complete.")


if __name__ == "__main__":
    main()
