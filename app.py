import streamlit as st
import os

from ingestion.loaders import files_process
from ingestion.splitter import chunking
from ingestion.embeddings import vectors
from ingestion.embedding_model import embeddings
from retrieval.retriever import retrieve

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide"
    )
    
st.header("**RAG USING LANGCHAIN**")
st.divider()

st.header("Document Upload")
            
files = st.file_uploader(
    "Upload your Files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)



question = st.chat_input("Enter your Question:")

if files:

    docs = files_process(files)

    chunks = chunking(docs)

    vectors(chunks)

    if question:

        result = retrieve(question)
    
        for doc in result:
    
            st.write(doc.page_content)


