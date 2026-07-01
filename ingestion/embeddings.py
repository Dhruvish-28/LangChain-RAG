from .embedding_model import embeddings
from langchain_community.vectorstores import FAISS

def vectors(chunks):

    vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
    )

    vectorstore.save_local("vector_db")
