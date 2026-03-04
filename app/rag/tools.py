"""Definición de herramientas (Tools) para el sistema multi-agente.

Cada Tool es una función que el agente orquestador puede invocar
según la intención del usuario:
- RAG/Docs: busca en documentos indexados en FAISS
- PQL/SQL: simula consultas a base de datos PostgreSQL
- K8s: simula operaciones de Kubernetes
"""

from langchain.agents import Tool
from app.rag.chain import build_rag_chain


# ---------------------------------------------------------------------------
# Tool 1: Búsqueda en documentos (RAG con FAISS)
# ---------------------------------------------------------------------------

def create_rag_tool(vectorstore) -> Tool:
    """Crea la herramienta RAG que busca respuestas en los documentos indexados."""

    rag_chain = build_rag_chain(vectorstore)

    def search_documents(query: str) -> str:
        """Busca información relevante en los documentos PDF indexados."""
        try:
            return rag_chain.invoke(query)
        except Exception as e:
            return f"Error al buscar en documentos: {e}"

    return Tool(
        name="Buscar_Documentos",
        func=search_documents,
        description=(
            "Útil para buscar información en los documentos PDF indexados. "
            "ESTA ES LA HERRAMIENTA PRINCIPAL. Úsala siempre que el usuario pregunte "
            "sobre proyectos (como Cafetech), propuestas, conceptos, definiciones, "
            "manuales, o cualquier cosa que parezca estar en la base de conocimiento "
            "antes de intentar responder por tu cuenta."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 2: Consultas SQL / PostgreSQL (Mock)
# ---------------------------------------------------------------------------

def _mock_pql_query(query: str) -> str:
    """Simula una consulta a base de datos PostgreSQL.

    En una implementación real, esto se conectaría a PostgreSQL y ejecutaría
    consultas SQL generadas por el LLM.
    """

    return f"""🗄️ **Agente PQL/SQL** (Simulación)

**Pregunta recibida:** {query}

**Acción que realizaría:**
1. Analizar la intención de la consulta del usuario.
2. Generar la consulta SQL apropiada para PostgreSQL.
3. Ejecutar la consulta contra la base de datos.
4. Formatear y devolver los resultados.

**Ejemplo de consulta SQL generada:**
```sql
-- Basado en la pregunta: "{query}"
SELECT * FROM tabla_relevante
WHERE condicion ILIKE '%{query.split()[0] if query.split() else "dato"}%'
ORDER BY fecha_creacion DESC
LIMIT 10;
```

**Nota:** Esta es una simulación educativa. En producción, el agente se conectaría
a una base de datos PostgreSQL real y ejecutaría consultas de forma segura con
validación y restricciones de solo lectura (SELECT)."""


def create_pql_tool() -> Tool:
    """Crea la herramienta de consultas SQL (simulada)."""

    return Tool(
        name="Consultar_Base_de_Datos",
        func=_mock_pql_query,
        description=(
            "Útil para consultar información en la base de datos PostgreSQL. "
            "Usa esta herramienta cuando el usuario pregunte sobre datos "
            "almacenados, registros, tablas, estadísticas de la base de datos, "
            "consultas SQL, reportes o información que se obtendría de una BD."
        ),
    )


# ---------------------------------------------------------------------------
# Tool 3: Operaciones Kubernetes (Mock)
# ---------------------------------------------------------------------------

def _mock_k8s_operation(query: str) -> str:
    """Simula operaciones de Kubernetes.

    En una implementación real, esto ejecutaría comandos kubectl o se
    conectaría al API de Kubernetes.
    """

    return f"""⎈ **Agente Kubernetes** (Simulación)

**Solicitud recibida:** {query}

**Acción que realizaría:**
1. Interpretar la solicitud del usuario sobre infraestructura.
2. Determinar los comandos kubectl o recursos K8s necesarios.
3. Ejecutar las operaciones de forma segura.
4. Reportar el estado y resultados.

**Comandos kubectl que ejecutaría:**
```bash
# Basado en la solicitud: "{query}"
kubectl get pods -n production
kubectl describe deployment app-deployment -n production
kubectl logs -f deployment/app-deployment -n production --tail=50
```

**Estado del clúster (simulado):**
- Pods activos: 3/3 ✅
- Deployments: 2 activos
- Services: 3 (ClusterIP, NodePort, LoadBalancer)
- Namespaces: default, production, staging

**Nota:** Esta es una simulación educativa. En producción, el agente se conectaría
al clúster Kubernetes real via kubeconfig y ejecutaría operaciones con RBAC
(Role-Based Access Control) restringido."""


def create_k8s_tool() -> Tool:
    """Crea la herramienta de operaciones Kubernetes (simulada)."""

    return Tool(
        name="Gestionar_Kubernetes",
        func=_mock_k8s_operation,
        description=(
            "Útil para gestionar y consultar infraestructura Kubernetes. "
            "Usa esta herramienta cuando el usuario pregunte sobre pods, "
            "deployments, servicios, clústeres, escalado, estado de la "
            "infraestructura, kubectl, contenedores o DevOps."
        ),
    )


# ---------------------------------------------------------------------------
# Crear todas las herramientas
# ---------------------------------------------------------------------------

def create_all_tools(vectorstore) -> list[Tool]:
    """Crea y retorna la lista completa de herramientas para el agente."""

    return [
        create_rag_tool(vectorstore),
        create_pql_tool(),
        create_k8s_tool(),
    ]
