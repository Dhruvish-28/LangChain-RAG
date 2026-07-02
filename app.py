import streamlit as st
import os

from ingestion.loaders import files_process
from ingestion.splitter import chunking
from ingestion.embeddings import vectors
from ingestion.embedding_model import embeddings
from retrieval.prompt_response import prompt_template

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
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

history = []

question = st.chat_input("Enter your Question:")

if files:

    docs = files_process(files)

    chunks = chunking(docs)

    vectors(chunks)

    if question:
        
        history = st.session_state.messages[-6:]

        response = prompt_template(question,history,chunks)

        st.session_state.messages.append( HumanMessage(content=question) )
        st.session_state.messages.append( AIMessage(content=response.content) )

        st.write(response.content)


