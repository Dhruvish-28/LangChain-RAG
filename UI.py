import streamlit as st

from ingestion.loaders import files_process
from ingestion.splitter import chunking
from ingestion.embeddings import vectors
from retrieval.prompt_response import prompt_template

from langchain_core.messages import (
    HumanMessage,
    AIMessage
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
import shutil
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

def stream_text(text):

    for word in text.split():

        yield word + " "

        time.sleep(0.03)

def display_metadata(message):

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
    
def display_chat():

    for message in st.session_state.messages:

        if message["role"] == "user":
    
            with st.chat_message("user"):
                st.write(message["content"])
    
        else:
    
            with st.chat_message("assistant"):
    
                st.write(message["content"])

                with st.expander("Metadata"):

                    display_metadata(message)


def generate_response(question, history):

    start_time = time.time()

    with st.chat_message("assistant"):
        
        with st.spinner("Thinking..."):

            response , docs = prompt_template(question,history,st.session_state.chunks)

        streamed_text = st.write_stream(stream_text(response.content))
        
    end_time = time.time()

    sources  = metadata(docs)

    st.session_state.messages.append(
    {
        "role": "assistant",
        "content": streamed_text,
        "sources": sources,
        "tokens": response.usage_metadata,
        "total_time": round(end_time - start_time, 2)
    }
)
    with st.expander("Metadata"):
        col1, col2, col3 = st.columns(3)
                    
        with col1:
            st.write("**📄 Source**")
            st.write(", ".join(sources))
    
        with col2:
            st.write("**⏱ Time**")
            st.write(f"{round(end_time - start_time, 2)} sec")
    
        with col3:
            tokens= response.usage_metadata
            st.write("**🪙 Tokens**")
            st.write(f"In : {tokens['input_tokens']}")
            st.write(f"Out : {tokens['output_tokens']}")
            st.write(f"Total : {tokens['total_tokens']}")

def process_files(files):

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

        col1,col2 = st.columns(2)

        with(col1):
            if st.button("Process Files"):
                
                process_documents(valid_files)
                
        with(col2):
            if st.button("Reset all"):
    
                if os.path.exists("vector_db"):
                    shutil.rmtree("vector_db")
                
                if os.path.exists("temp"):
                    shutil.rmtree("temp")
    
                st.session_state.clear()
    
                st.rerun()
            
    else:

        st.warning("Please upload at least one supported document.")
    

st.title("RAG using LangChain")

st.divider()

if __name__ == "__main__":

    files = st.file_uploader(
    "Upload your Files",
    # type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
    )

    if files:

        process_files(files)
    
    if st.session_state.processed:
    
        display_chat()
    
        history = st.session_state.messages[-6:]
        
        question = st.chat_input("Enter your Question:")
    
        if question:
            
            st.session_state.messages.append(
                {
                "role": "user",
                "content": question
                }
            )
            with st.chat_message("user"):
                st.write(question)
           
            try:
                
                generate_response(question , history)
                
            
            except ChatGoogleGenerativeAIError as e:
                
                st.error(
            "Gemini API quota exhausted.\n\n"
            "Please wait before retrying."
                )
    
            except Exception as e:
    
                st.exception(e)
    
        # st.rerun()
    
        
    else:
    
        st.info("Upload documents and click 'Process Files' to begin.")
        st.info("Supported formats: PDF, DOCX, TXT and MD |  Unsupported files will be discarded...")
