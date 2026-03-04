"""Cadena RAG mejorada: combina retrieval + generación con OpenAI y memoria."""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from app.config import OPENAI_API_KEY, OPENAI_MODEL


# Prompt template para el RAG
RAG_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente inteligente que responde preguntas basándose en documentos.

Contexto recuperado de los documentos:
{context}

Pregunta del usuario: {question}

Instrucciones:
- Responde basándote SOLO en el contexto proporcionado arriba.
- Si la información no está en el contexto, di "No encontré esa información en los documentos disponibles."
- Sé claro, conciso y directo.
- Responde en español.

Respuesta:
""")


def format_docs(docs: list) -> str:
    """Concatena el contenido de los documentos recuperados en un solo string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_llm() -> ChatOpenAI:
    """Retorna la instancia del LLM (ChatOpenAI) configurada."""
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.3,
    )


def build_rag_chain(vectorstore):
    """Construye la cadena RAG completa:

    1. Retriever: busca los 4 chunks más relevantes por similitud semántica
    2. Prompt: inyecta el contexto recuperado + la pregunta del usuario
    3. LLM: genera la respuesta con OpenAI GPT
    4. Parser: extrae el texto limpio de la respuesta
    """

    # Retriever: busca documentos similares a la query
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # LLM: modelo de lenguaje via OpenAI
    llm = get_llm()

    # Cadena RAG usando LCEL (LangChain Expression Language)
    chain = (
        {
            "context": retriever | format_docs,   # Recuperar y formatear docs
            "question": RunnablePassthrough(),     # Pasar la pregunta tal cual
        }
        | RAG_PROMPT   # Armar el prompt con contexto + pregunta
        | llm          # Generar respuesta
        | StrOutputParser()  # Extraer texto plano
    )

    return chain
