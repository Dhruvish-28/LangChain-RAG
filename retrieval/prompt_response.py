from langchain_core.prompts import ChatPromptTemplate
from .retriever import retrieve
from .llm_model import llm

def prompt_template(question , history):

    docs = retrieve(question)

    context = "\n\n".join( doc.page_content for doc in docs )

    template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply:

"I couldn't find that information in the uploaded documents."

Keep answers concise.
"""
        ),

        (
        "placeholder",
        "{history}"
        ),

        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)

    prompt = template.invoke(
    {
        "history" : history,
        "context": context,
        "question": question
    }
)

    response = llm.invoke(prompt)

    return response