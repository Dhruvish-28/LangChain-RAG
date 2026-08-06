from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os 
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

@st.cache_resource
def load_gemini_client():
    return  ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key = os.getenv("GEMINI_API_KEY"),
    temperature=0
)

llm = load_gemini_client()
