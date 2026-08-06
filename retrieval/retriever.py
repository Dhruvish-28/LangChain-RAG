from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .ensemble_retriever import combine_retriever
from models.llm_model import llm
from models.transformer import reranker 

def ranking(docs, question):
    """Rerank documents with the cross-encoder and return (top docs, scores).

    The scores are returned alongside the docs so the UI can display
    provenance / evidence with relevance confidence.
    """
    pairs = [(question, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:6]

    docs = [doc for _, doc in ranked]
    scores = [float(score) for score, _ in ranked]

    return docs, scores
    
def retrieve(question, history , chunks):

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

    retriever = combine_retriever(chunks)    

    history_aware_retriever = create_history_aware_retriever(llm , retriever , rephrase_prompt)

    result = history_aware_retriever.invoke(
        {
            "input" : question,
            "chat_history" : history
        }
    )

    result , scores = ranking(result, question)

    return result , scores
