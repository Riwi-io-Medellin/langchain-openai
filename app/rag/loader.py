"""Carga de documentos PDF desde una carpeta."""

import os
from langchain_community.document_loaders import PyPDFLoader
from app.config import DOCS_PATH


def load_pdfs() -> list:
    """Carga todos los PDFs de la carpeta configurada y retorna una lista de documentos.

    Cada página del PDF se convierte en un Document de LangChain con su contenido
    y metadatos (fuente, número de página).
    """

    docs = []
    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"⚠️  No se encontraron PDFs en '{DOCS_PATH}'. Agrega al menos uno.")
        return docs

    for filename in sorted(pdf_files):
        filepath = os.path.join(DOCS_PATH, filename)
        print(f"📄 Cargando: {filename}")

        # PyPDFLoader extrae texto página por página
        loader = PyPDFLoader(filepath)
        docs.extend(loader.load())

    print(f"✅ {len(docs)} páginas cargadas de {len(pdf_files)} archivo(s)")
    return docs
