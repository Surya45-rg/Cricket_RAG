import os

from langchain_groq import ChatGroq


def get_llm():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to a .env file "
            "(GROQ_API_KEY=...) or to .streamlit/secrets.toml."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=api_key,
    )

    return llm
