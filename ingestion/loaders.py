from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
import os

def files_process(files):

    all_docs = []

    os.makedirs("temp", exist_ok=True)

    for file in files:

        temp_path = f"temp/{file.name}"

        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        extension = os.path.splitext(temp_path)[1].lower()

        match extension:

            case ".pdf":
                loader = PyPDFLoader(temp_path)

            case ".docx":
                loader = Docx2txtLoader(temp_path)

            case ".txt" | ".md":
                loader = TextLoader(temp_path)

            case _:
                continue

        docs = loader.load()

        all_docs.extend(docs)

    return all_docs
