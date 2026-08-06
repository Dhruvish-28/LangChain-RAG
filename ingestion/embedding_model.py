from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embedding_model()
