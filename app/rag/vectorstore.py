"""Creación y carga del vector store con FAISS."""

import os
import shutil
from langchain_community.vectorstores import FAISS
from app.config import FAISS_PATH
from app.rag.embeddings import get_embeddings


def create_vectorstore(chunks: list) -> FAISS:
    """Crea un vector store nuevo a partir de chunks e indexa los embeddings.

    FAISS persiste en disco (save_local) para no recalcular embeddings cada vez.
    Es más rápido que ChromaDB para búsqueda local y no requiere servidor.
    """

    # Limpiar vector store anterior si existe
    if os.path.exists(FAISS_PATH):
        try:
            shutil.rmtree(FAISS_PATH)
            print("🗑️  Vector store anterior eliminado")
        except PermissionError:
            print("⚠️  No se pudo eliminar vector store anterior (permisos). Continuando...")

    embeddings = get_embeddings()

    print(f"📦 Creando índice FAISS con {len(chunks)} chunks...")

    # FAISS puede procesar todos los chunks de una vez (más eficiente que ChromaDB)
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    # Persistir en disco
    os.makedirs(FAISS_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_PATH)

    print(f"💾 Índice FAISS creado y guardado en '{FAISS_PATH}' ({len(chunks)} documentos)")
    return vectorstore


def load_vectorstore() -> FAISS | None:
    """Carga un índice FAISS existente desde disco.

    Retorna None si no existe o está vacío.
    """

    index_file = os.path.join(FAISS_PATH, "index.faiss")
    if not os.path.exists(index_file):
        print("⚠️  No existe un índice FAISS. Ejecuta primero la indexación.")
        return None

    embeddings = get_embeddings()

    try:
        vectorstore = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        # Verificar que tenga datos
        doc_count = vectorstore.index.ntotal
        if doc_count == 0:
            print("⚠️  Índice FAISS vacío. Ejecuta primero la indexación.")
            return None

        print(f"📂 Índice FAISS cargado desde '{FAISS_PATH}' ({doc_count} vectores)")
        return vectorstore

    except Exception as e:
        print(f"⚠️  Error al cargar índice FAISS: {e}")
        return None
