from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .ensemble_retriever import combine_retriever
from models.llm_model import llm
from models.transformer import reranker 

def ranking(docs , question) :
    
    pairs = [ (question, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted( zip(scores, docs), key=lambda x: x[0], reverse=True )    

    docs = [ doc for _, doc in ranked[:6]]

    return docs
    
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

    result = ranking(result  , question)

    return result