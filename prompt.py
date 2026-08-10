def build_prompt(context, question):

    return f"""
You are an expert cricket analyst.

Answer ONLY using the context below.

If the answer is not available,
reply:

I couldn't find that information in the cricket database.

Context:

{context}

Question:

{question}

Answer:
"""