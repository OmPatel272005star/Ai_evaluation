from pathlib import Path
from langchain_ollama import ChatOllama


# Application LLM
llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0
)


def load_document():
    document_path = Path(__file__).parent / "document.txt"

    with open(document_path, "r", encoding="utf-8") as file:
        return file.read()


def document_qa(question: str) -> str:

    document = load_document()

    prompt = f"""
        You are a Document Q&A assistant.

        Answer the user's question using ONLY the information provided
        in the document.

        If the answer is not available in the document, say:
        "I cannot find the answer in the document."

        Do not invent information.

        DOCUMENT:
        {document}

        QUESTION:
        {question}

        ANSWER:
        """

    response = llm.invoke(prompt)

    return str(response.content)