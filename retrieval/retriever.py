from langchain_community.vectorstores import FAISS
from ingestion.embedding_model import embeddings

def retrieve(question):

    vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
    )

    results = retriever.invoke(
    question
    )

    return results