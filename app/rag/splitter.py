"""División de documentos en chunks para indexar en el vector store."""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(docs: list) -> list:
    """Divide documentos en fragmentos más pequeños (chunks).

    - chunk_size: tamaño máximo de cada fragmento en caracteres.
    - chunk_overlap: solapamiento entre fragmentos para no perder contexto.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separadores ordenados por prioridad: párrafos > líneas > espacios
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"🔪 {len(docs)} páginas → {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
