from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from ingestion.embedding_model import embeddings
from ingestion.splitter import chunking

def combine_retriever(chunks):
    
    vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 20,
        "fetch_k": 30,
        "lambda_mult": 0.5
        }
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 20

    ensemble = EnsembleRetriever( retrievers=[bm25,retriever], weights=[0.4, 0.6] )

    return ensemble
    