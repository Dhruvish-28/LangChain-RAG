from langchain_core.prompts import ChatPromptTemplate
from .retriever import retrieve
from models.llm_model import llm

def prompt_template(question , history,chunks):

    docs = retrieve(question , history , chunks)

    context = "\n\n".join( doc.page_content for doc in docs )

    template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Instructions:
- Use ALL relevant information from the retrieved context.
- Combine information from multiple retrieved documents when appropriate.
- If the user asks for a detailed explanation, provide a comprehensive, structured answer.
- Do not rely only on the first retrieved chunk.
- If information is missing from the context, explicitly say : "I couldn't find that information in the uploaded documents." .

Answer ONLY from the provided context.

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

    return response , docs