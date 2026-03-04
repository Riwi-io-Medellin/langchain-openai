# Guía para Nuevos Desarrolladores (Onboarding)

¡Bienvenido al código fuente del sistema **RAG Multi-Agente con OpenAI y FAISS**! Este documento está diseñado para ayudarte a entender rápidamente qué hace el proyecto, qué tecnologías usa y por qué se eligieron.

---

## 1. Entendiendo el Concepto Base

### ¿Qué es un RAG?
**RAG** significa *Retrieval-Augmented Generation* (Generación Aumentada por Recuperación). En lugar de que el modelo de Inteligencia Artificial (LLM) responda solo de memoria, un sistema RAG le permite **leer documentos internos** primero. 
1. El usuario pregunta algo.
2. El sistema busca párrafos relevantes en un PDF.
3. El sistema le pasa esos párrafos al LLM y le dice: *"Responde al usuario usando solo esta información"*.

### ¿Qué es un Sistema Multi-Agente?
A diferencia de un chat estático, este proyecto usa un **Agente (Agent)**. Un agente es un LLM al que se le da un conjunto de **Herramientas (Tools)** y se le permite decidir cuál usar. Si le preguntas por información de un PDF, él decide usar la herramienta `Buscar_Documentos`. Si le preguntas sobre infraestructura, usa `Gestionar_Kubernetes`.

---

## 2. Explicación de las Dependencias (requirements.txt)

Cada librería tiene un propósito específico en el flujo de trabajo de la IA:

### LangChain (`langchain`, `langchain-community`)
Es el framework principal. Orquesta todas las piezas del rompecabezas. En lugar de escribir docenas de líneas de código para pedirle algo a la API de OpenAI, procesar su respuesta, juntarla con documentos y manejar la memoria, LangChain provee clases de Python pre-construidas para hacerlo con pocas líneas.

### LangChain OpenAI (`langchain-openai`)
Es el puente de integración oficial entre LangChain y la API de OpenAI.
- Lo usamos para importar `ChatOpenAI` (el que genera el texto de las respuestas, usamos GPT-4o-mini).
- Lo usamos para importar `OpenAIEmbeddings` (el que convierte el texto en números o "vectores").

### FAISS (`faiss-cpu`)
**FAISS** (*Facebook AI Similarity Search*) es una base de datos vectorial súper rápida creada por Meta/Facebook.
- **¿Qué es un vector?** Cuando cargas un PDF, OpenAI convierte los textos en largas listas de números (embeddings).
- **¿Para qué sirve FAISS?** Cuando el usuario pregunta "Qué es Cafetech", OpenAI convierte la pregunta a números. FAISS calcula matemáticamente qué párrafos del PDF (en números) son más cercanos a la pregunta. La gran ventaja de FAISS vs ChromaDB es que es ligerísima, corre en memoria (RAM) y no requiere levantar servicios pesados. Usamos la versión `cpu` para que funcione en cualquier computadora sin requerir tarjeta gráfica.

### PyPDF (`pypdf`)
Librería súper ligera para extraer el texto de los archivos `.pdf` que hay en la carpeta `docs/`. LangChain la usa por debajo (`PyPDFLoader`) para leer las páginas.

### Python Dotenv (`python-dotenv`)
Se encarga de leer el archivo `.env` (donde está la `OPENAI_API_KEY`) y cargar sus valores como variables de entorno del sistema operativo temporalmente mientras corre Python, manteniendo seguras tus credenciales tecnológicas.

### Rich (`rich`)
No es de IA, pero es vital. Se encarga de hacer que la terminal se vea bonita. Pinta los bordes de la consola de colores, las respuestas con emoticonos, el texto en negrita y permite hacer renderizado de Markdown (como las listas y el código) en la terminal de Docker para que sea súper legible y parezca una interfaz visual.

---

## 3. ¿El Archivo Más Importante a Revisar Primero?

Si un nuevo Dev entra al proyecto, la curva de lectura del código debería ser esta:

1. `app/main.py`: Entiende cómo arranca todo (el punto de entrada central, la verificación de keys y el bucle visual del input de usuario).
2. `app/rag/agents.py`: Este es el "Cerebro". Aquí se inicializa a OpenAI, se le conecta la Memoria y se le pasan las Herramientas que puede usar. Lee detenidamente la variable `AGENT_SYSTEM_PREFIX` para entender cómo se controla el comportamiento del agente.
3. `app/rag/tools.py`: Aquí es donde ocurre la magia técnica (acciones funcionales). Fíjate cómo la herramienta `Buscar_Documentos` (la real) está conectada a la base de datos de FAISS.
4. `app/rag/vectorstore.py` y `embeddings.py`: Aquí está cómo y dónde se guarda el índice FAISS.

---

## 5. Arquitectura del Código: Paso a Paso

Para entender en detalle cómo LangChain orquesta todo, aquí te pasamos las secciones núcleo del proyecto:

### A. Preparando la Base Intelectual (RAG)
Todo empieza convirtiendo los PDFs en conocimiento matemático para la IA.

```python
# app/rag/vectorstore.py
from langchain_community.vectorstores import FAISS
from app.rag.embeddings import get_embeddings

def create_vectorstore(chunks: list) -> FAISS:
    # 1. Obtenemos el traductor de texto->números de OpenAI
    embeddings = get_embeddings()
    
    # 2. FAISS toma cada chunk de texto y lo incrusta en el espacio vectorial
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 3. Guardamos esto en el disco duro para no volver a gastar créditos
    vectorstore.save_local("./data/faiss")
    return vectorstore
```

### B. Definiendo las Herramientas (Tools)
El agente no hace nada mágico por sí solo, LangChain le pasa "cajas de herramientas". Una herramienta no es más que una función de Python envuelta.

```python
# app/rag/tools.py
from langchain.agents import Tool

def create_rag_tool(vectorstore) -> Tool:
    rag_chain = build_rag_chain(vectorstore)
    
    return Tool(
        name="Buscar_Documentos",
        # Al ejecutar la tool, se llama a esta función:
        func=lambda q: rag_chain.invoke(q),
        # ESTO ES CRÍTICO: el LLM lee esta descripción para decidir si usa la herramienta o no
        description="Úsala siempre que el usuario pregunte sobre proyectos (como Cafetech), manuales..."
    )
```

### C. El Cerebro (El Agente Orquestador)
Esta es la función nuclear del proyecto. Aquí unimos a OpenAI, la Memoria y las Herramientas.

```python
# app/rag/agents.py
from langchain.agents import initialize_agent, AgentType

def create_agent(vectorstore):
    llm = get_llm() # Nuestro modelo ChatOpenAI
    tools = create_all_tools(vectorstore)
    memory = create_memory() # ConversationBufferMemory

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        # Este flag le dice a LangChain: "Usa el formato ReAct (Reason+Act) pero conversacional"
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        # Si la IA escupe JSON con formato malo, LangChain lo arregla en lugar de crashear
        handle_parsing_errors=True,
        agent_kwargs={"prefix": AGENT_SYSTEM_PREFIX} # Reglas de comportamiento estrictas
    )
    return agent
```

### D. El Bucle Infinito del Usuario
Cuando corres la app, simplemente entramos en un `while True` que le pasa el input del usuario al `agent.invoke()`.

```python
# app/main.py
while True:
    question = console.input("[bold cyan]❓ Pregunta:[/] ")
    if question == 'salir': break

    # Aquí el Agente hace todo el trabajo pesado:
    # Piensa -> Selecciona Tool -> Busca en FAISS -> Combina la respuesta
    answer = agent.invoke({"input": question})
    
    print(answer.get("output"))
```

---

## 6. Recomendaciones Rápidas para Operar

1. **Si quieres hacer más inteligente al Bot**: No toques la infraestructura ni LangChain directamente. Primero vete al prompt del agente (`app/rag/agents.py`) o re-escribe la descripción de las herramientas en (`app/rag/tools.py`). El LLM lee las descripciones de las funciones en `tools.py` para decidir si usarlas o no.
2. **Si el agente deja de responder con información del documento**: Suele ser un problema en el `chunking`. Ve a `app/config.py` y revisa los valores `CHUNK_SIZE` y `CHUNK_OVERLAP`. Son las "tijeras" que cortan las hojas en pedacitos antes de convertirlas a vectores.
3. **Persistencia**: FAISS carga los datos rapidísimo y cuando se termina, los deja en disco local (`data/faiss/index.faiss`). Si borras los archivos en `data/` o si pones PDFs nuevos en `docs/`, debes reiniciar la app para que vuelva a crear el index de lo contrario usará la caché vectorial.
