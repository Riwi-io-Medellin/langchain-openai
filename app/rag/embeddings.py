"""Configuración del modelo de embeddings con OpenAI API."""

from langchain_openai import OpenAIEmbeddings
from app.config import OPENAI_API_KEY, OPENAI_EMBED_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    """Retorna el modelo de embeddings conectado a OpenAI API.

    Usa 'text-embedding-3-small' por defecto: económico y efectivo para RAG.
    Los embeddings convierten texto en vectores numéricos para búsqueda semántica.
    """

    return OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        api_key=OPENAI_API_KEY,
    )
