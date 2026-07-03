import streamlit as st

from ingestion.loaders import files_process
from ingestion.splitter import chunking
from ingestion.embeddings import vectors
from retrieval.prompt_response import prompt_template

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)

import time
import os

st.set_page_config(
    page_title="RAG Using LangChain",
    page_icon="📄",
    layout="wide"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "chunks" not in st.session_state:
    st.session_state.chunks = None

SUPPORTED_FILES = [".pdf", ".docx", ".txt", ".md"]

invalid_files = []
valid_files = []

def process_documents(files):

    with st.spinner("Processing documents..."):

        docs = files_process(files)

        chunks = chunking(docs)

        vectors(chunks)

    st.session_state.chunks = chunks
    st.session_state.processed = True

    st.success("Knowledge Base Created Successfully!")

def metadata(docs):
    
    sources = set()

    for doc in docs:
         sources.add(os.path.basename(doc.metadata.get("source", "Unknown")))

    return list(sources)
    
def display_chat():

    for message in st.session_state.messages:

        if message["role"] == "user":
    
            with st.chat_message("user"):
                st.write(message["content"])
    
        else:
    
            with st.chat_message("assistant"):
    
                st.write(message["content"])
    
                # with st.expander("Metadata"):
    
                #     st.write("**Sources:**")
                #     for source in message["sources"]:
                #         st.write(f"• {source}")

                #     total_time = message["total_time"]
                #     st.write(f"**Time:** {total_time} sec")

                #     st.write("**Tokens:**")
                #     tokens = message["tokens"]

                #     st.write(f"Input Tokens : {tokens['input_tokens']}")
                #     st.write(f"Output Tokens : {tokens['output_tokens']}")
                #     st.write(f"Total Tokens : {tokens['total_tokens']}")

                with st.expander("Metadata"):

                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        st.write("**📄 Source**")
                        st.write(", ".join(message["sources"]))
                
                    with col2:
                        st.write("**⏱ Time**")
                        st.write(f"{message['total_time']} sec")
                
                    with col3:
                        tokens = message["tokens"]
                
                        st.write("**🪙 Tokens**")
                        st.write(f"In : {tokens['input_tokens']}")
                        st.write(f"Out : {tokens['output_tokens']}")
                        st.write(f"Total : {tokens['total_tokens']}")


def generate_response(question):

    history = st.session_state.messages[-6:]

    start_time = time.time()
        
    response , docs = prompt_template(
        question,
        history,
        st.session_state.chunks
    )
    st.write("reponse generated")
    end_time = time.time()
    
    st.session_state.messages.append(
    {
        "role": "user",
        "content": question
    }
    )

    sources  = metadata(docs)

    st.session_state.messages.append(
    {
        "role": "assistant",
        "content": response.content,
        "sources": sources,
        "tokens": response.usage_metadata,
        "total_time": round(end_time - start_time, 2)
    }
)


st.title("RAG Using LangChain")

st.divider()

st.subheader("Document Upload")

files = st.file_uploader(
    "Upload your Files",
    # type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)


if files:

    for file in files:

        extension = os.path.splitext(file.name)[1].lower()

        if extension in SUPPORTED_FILES:
            valid_files.append(file)
        else:
            invalid_files.append(file.name)

    if invalid_files:

        st.error(
            f"Unsupported file format(s): {', '.join(invalid_files)}"
        )

    if valid_files:
        if st.button("Process Files"):
            
            process_documents(valid_files)
    else:

        st.warning("Please upload at least one supported document.")


st.divider()


if st.session_state.processed:

    
    question = st.chat_input("Enter your Question:")

    if question:
       
        try:
            
            generate_response(question)
            
        except Exception as e:
            
            if "quota" in str(e).lower():

                st.error("Gemini API quota exhausted. This application currently uses the free tier.")

            else:

                st.exception(e)

    display_chat()
else:

    st.info("Upload documents and click 'Process Files' to begin.")
    st.info("Supported formats: PDF, DOCX, TXT and MD |  Unsupported files will be discarded...")