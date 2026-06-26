from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

def chunking(all_docs):

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
    )

    chunks = splitter.split_documents(all_docs)

    return chunks