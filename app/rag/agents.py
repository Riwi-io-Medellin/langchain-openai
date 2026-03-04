"""Agente orquestador multi-agente con LangChain.

El agente principal recibe la pregunta del usuario y decide automáticamente
a qué herramienta (Tool) delegarla:
- Preguntas sobre documentos → Buscar_Documentos (FAISS RAG)
- Preguntas sobre base de datos → Consultar_Base_de_Datos (PQL mock)
- Preguntas sobre infraestructura → Gestionar_Kubernetes (K8s mock)
"""

from langchain.agents import initialize_agent, AgentType
from app.rag.chain import get_llm
from app.rag.memory import create_memory
from app.rag.tools import create_all_tools


AGENT_SYSTEM_PREFIX = """Eres un asistente multi-agente inteligente encargado de responder preguntas basándote ESTRICTAMENTE en las herramientas proporcionadas.

Tienes acceso a las siguientes herramientas:
- **Buscar_Documentos**: Para buscar información en documentos PDF indexados (RAG).
- **Consultar_Base_de_Datos**: Para consultar datos en la base de datos PostgreSQL.
- **Gestionar_Kubernetes**: Para gestionar y consultar la infraestructura Kubernetes.

REGLAS CRÍTICAS:
1. NUNCA respondas usando tu conocimiento general si la pregunta parece referirse a un proyecto, empresa, tecnología específica o concepto que podría estar documentado.
2. SIEMPRE usa la herramienta `Buscar_Documentos` primero si el usuario pregunta "qué es", "cómo funciona" o menciona nombres propios (ej. Cafetech, proyectos, propuestas).
3. Si la herramienta no devuelve información útil, dilo explícitamente ("No encontré información sobre esto en los documentos").
4. Si la pregunta es sobre datos, registros, SQL o reportes, usa `Consultar_Base_de_Datos`.
5. Si la pregunta es sobre pods, deployments, infraestructura o DevOps, usa `Gestionar_Kubernetes`.
6. Responde siempre en español.
"""


def create_agent(vectorstore):
    """Crea el agente orquestador con herramientas y memoria conversacional.

    Usa AgentType.CONVERSATIONAL_REACT_DESCRIPTION para:
    - Mantener historial de conversación (memoria)
    - Decidir cuándo usar herramientas vs. responder directo
    - Razonamiento ReAct (Reasoning + Acting)
    """

    # 1. LLM compartido por el agente
    llm = get_llm()

    # 2. Memoria conversacional
    memory = create_memory()

    # 3. Herramientas especializadas
    tools = create_all_tools(vectorstore)

    # 4. Inicializar agente con ReAct conversacional
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,  # Muestra el razonamiento del agente en consola
        handle_parsing_errors=True,  # Maneja errores de parsing gracefully
        agent_kwargs={
            "prefix": AGENT_SYSTEM_PREFIX,
        },
    )

    return agent
