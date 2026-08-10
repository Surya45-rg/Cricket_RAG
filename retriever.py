def retrieve_documents(vector_db, question, k=5, tournament=None):

    filter_dict = {"tournament": {"$eq": tournament}} if tournament else None

    results = vector_db.similarity_search_with_score(
        question,
        k=k,
        filter=filter_dict,
    )

    return results
