from .embedding_model import embeddings
from langchain_community.vectorstores import FAISS

def vectors(chunks):

    if len(chunks) == 0:
        raise ValueError(
            "Documents maybe not contain text (possible : scanned images)."
        )
    
    vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
    )

    vectorstore.save_local("vector_db")
