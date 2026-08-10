import shutil

from dotenv import load_dotenv

from pdf_loader import load_documents
from embeddings import get_embedding_model
from vector_store import PERSIST_DIR, get_vector_store, safe_count, index_documents
from retriever import retrieve_documents
from prompt import build_prompt
from llm import get_llm

load_dotenv()

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
    print("Vector DB already built, reusing chroma_db/\n")
else:
    if count:
        print(f"Resuming: {count}/{total} documents already indexed.")

    def report(done, total):
        print(f"Indexed {done}/{total} documents")

    index_documents(vector_db, documents, on_progress=report, start_at=count)

llm = get_llm()

print("Vector DB Ready\n")

while True:

    question = input("Ask Question : ")

    if question.lower() == "exit":
        break

    results = retrieve_documents(
        vector_db,
        question
    )

    context = "\n\n".join(
        doc.page_content
        for doc, score in results
    )

    prompt = build_prompt(
        context,
        question
    )

    response = llm.invoke(prompt)

    print("\nAnswer\n")

    print(response.content)
