from langchain_community.vectorstores import FAISS
from ingestion.embedding_model import embeddings
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .llm_model import llm

def retrieve(question, history):

    vectorstore = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 20,
        "lambda_mult": 0.5
        }
    )

    rephrase_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a query rewriting assistant.

Given the conversation history and the latest user question,
rewrite the latest question so it can be understood without
the previous conversation.

Do NOT answer the question.

Only return the rewritten standalone question.

If the latest question is already standalone,
return it unchanged.
"""
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "{input}"
        )
    ]
)

    history_aware_retriever = create_history_aware_retriever(llm , retriever , rephrase_prompt)

    result = history_aware_retriever.invoke(
        {
            "input" : question,
            "chat_history" : history
        }
    )

    return result