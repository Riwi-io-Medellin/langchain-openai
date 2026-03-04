"""Punto de entrada: indexa PDFs y lanza un chat interactivo con el agente multi-herramienta."""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from app.config import OPENAI_API_KEY
from app.rag.loader import load_pdfs
from app.rag.splitter import split_documents
from app.rag.vectorstore import create_vectorstore, load_vectorstore
from app.rag.agents import create_agent

console = Console()


def verify_api_key() -> bool:
    """Verifica que la API Key de OpenAI esté configurada."""

    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-..."):
        console.print(
            Panel(
                "❌ OPENAI_API_KEY no configurada.\n\n"
                "1. Copia .env.example a .env:\n"
                "   cp .env.example .env\n\n"
                "2. Edita .env y coloca tu API Key:\n"
                "   OPENAI_API_KEY=sk-tu-key-real-aquí",
                title="⚠️  Configuración requerida",
                border_style="red",
            )
        )
        return False

    console.print("✅ OpenAI API Key configurada\n", style="green")
    return True


def index_documents():
    """Pipeline de indexación: cargar PDFs → dividir en chunks → crear índice FAISS."""

    console.print(Panel("📚 Indexando documentos", style="blue"))

    # Paso 1: Cargar PDFs
    docs = load_pdfs()
    if not docs:
        return None

    # Paso 2: Dividir en chunks
    chunks = split_documents(docs)

    # Paso 3: Crear índice FAISS con OpenAI Embeddings
    vectorstore = create_vectorstore(chunks)
    return vectorstore


def chat_loop(agent):
    """Bucle interactivo de preguntas y respuestas con el agente multi-herramienta."""

    console.print(
        Panel(
            "🤖 Agente Multi-Herramienta listo.\n\n"
            "Puedo ayudarte con:\n"
            "  📄 Preguntas sobre documentos (RAG con FAISS)\n"
            "  🗄️  Consultas a base de datos (SQL/PostgreSQL)\n"
            "  ⎈  Gestión de Kubernetes (infraestructura)\n\n"
            "Escribe 'salir' para terminar.",
            style="green",
        )
    )

    while True:
        try:
            question = console.input("\n[bold cyan]❓ Pregunta:[/] ")
        except (KeyboardInterrupt, EOFError):
            break

        if question.strip().lower() in ("salir", "exit", "quit", "q"):
            break

        if not question.strip():
            continue

        console.print("💭 Pensando...", style="dim")

        try:
            # Invocar el agente — decide automáticamente qué herramienta usar
            answer = agent.invoke({"input": question})
            output = answer.get("output", str(answer))
            console.print(
                Panel(
                    Markdown(output),
                    title="🧠 Respuesta",
                    border_style="green",
                )
            )
        except KeyboardInterrupt:
            console.print("\n⚠️  Consulta interrumpida", style="yellow")
            continue
        except Exception as e:
            console.print(f"❌ Error: {e}", style="red")

    console.print("\n👋 ¡Hasta luego!", style="bold blue")


def main():
    """Flujo principal del programa."""

    console.print(
        Panel.fit(
            "🦜🔗 LangChain RAG Multi-Agente con OpenAI + FAISS",
            style="bold magenta",
        )
    )

    # 1. Verificar API Key de OpenAI
    if not verify_api_key():
        sys.exit(1)

    # 2. Indexar documentos o cargar índice existente
    vectorstore = load_vectorstore()
    if vectorstore is None:
        vectorstore = index_documents()
        if vectorstore is None:
            console.print(
                "❌ No hay documentos para procesar. Agrega PDFs en la carpeta 'docs/'.",
                style="red",
            )
            sys.exit(1)

    # 3. Crear agente multi-herramienta con memoria
    agent = create_agent(vectorstore)

    # 4. Iniciar chat interactivo
    chat_loop(agent)


if __name__ == "__main__":
    main()
