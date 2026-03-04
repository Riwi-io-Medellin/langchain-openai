"""Configuración centralizada del proyecto, cargada desde variables de entorno."""

import os
from dotenv import load_dotenv

# Cargar .env solo en desarrollo local (en Docker se inyectan directamente)
load_dotenv()

# --- OpenAI API ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# --- Rutas ---
DOCS_PATH: str = os.getenv("DOCS_PATH", "./docs")
FAISS_PATH: str = os.getenv("FAISS_PATH", "./data/faiss")

# --- Parámetros de chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
