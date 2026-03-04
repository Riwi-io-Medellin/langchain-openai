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

---

## 🔥 7. La Verdadera "Magia" del Sistema (Arquitectura Profunda)

Para dominar este proyecto y llevarlo a producción masiva, no necesitas ser un experto en LangChain; necesitas entender las **implicaciones técnicas** de estas 4 piezas críticas. Si algo se rompe, el problema estará en una de estas:

### 1️⃣ Chunking Semántico (Más Allá de FAISS)
FAISS no busca palabras, busca **similitud semántica matemática**. Si tu `CHUNK_SIZE` es muy pequeño o el `CHUNK_OVERLAP` (solapamiento) es de cero, los "trozos" de texto pierden contexto.
> 💡 *Implicación:* Si el texto original dice "...final. Cafetech es..." y cortas justo en el punto, pierdes continuidad. El *overlap* existe para que las ideas se "peguen". Si FAISS devuelve tonterías, el problema no es la base de datos, es tu estrategia de corte.

*Nota de Producción:* FAISS en memoria es genial para testing (hasta unos 500k chunks / ~3GB RAM). Para escalar a millones de documentos, tendrías que migrar la interfaz de FAISS a una base de datos vectorial externa como Pinecone o Qdrant.

### 2️⃣ Diseño Conductual de Herramientas (Tools)
El código `description="Úsala siempre que..."` en `app/rag/tools.py` es la pieza de ingeniería más subestimada. El LLM **lee** esa descripción. Si la descripción es ambigua, el Agente decidirá no usar el RAG e inventar la respuesta de memoria.
> 💡 *Implicación:* Si el bot dice "No sé" o ignora los PDFs, el 90% de las veces no es un bug de código, es un *bug verbal* en la descripción de la Tool.

### 3️⃣ El Verdadero Cerebro (`AGENT_SYSTEM_PREFIX`)
Las instrucciones que le pasamos al Agente controlan sus **alucinaciones**. 
> 💡 *Implicación:* Si tu prompt no dice explícitamente *"Si no encuentras información en las herramientas di que no sabes"*, el LLM va a inventar datos para agradar al usuario. Un prefijo robusto y estricto reduce dramáticamente el comportamiento errático en producción.

### 4️⃣ El Tipo de Agente (`CONVERSATIONAL_REACT_DESCRIPTION`)
El patrón **ReAct (Reason + Act)** significa que el agente sigue un bucle eterno de: Pensamiento $\rightarrow$ Acción $\rightarrow$ Observación $\rightarrow$ Respuesta.
Este tipo de agente toma decisiones *dinámicas* y es muy flexible, pero al llevar a producción real debes saber que es **menos determinista** (puede que a veces resuelva distinto).
> 💡 *Implicación:* Si necesitas precisión militar, deberías abandonar este agente ReAct y pasar a uno de *Structured Chat* o de llamadas obligatorias a funciones.

---

## ⚠️ 8. Riesgos Silenciosos al Pasar a Producción

Si este es el inicio de tu proyecto empresarial, ten cuidado con estas trampas:

1. **Incompatibilidad de Vectores**: Si cambias el modelo de Embeddings (ej. pasas a uno más barato), **debes borrar la carpeta `data/faiss/`**. Un nuevo modelo escupe vectores de dimensiones diferentes, y FAISS crasheará.
2. **Explosión de Ventana de Contexto**: Si FAISS recupera 20 chunks para ser súper preciso, quizá te pases del límite de "Tokens" permitidos por GPT-4o-mini en una sola llamada y reviente la API.
3. **Memoria Infinita**: Usamos `ConversationBufferMemory` que guarda TODO el historial. En producción con un usuario intenso, el chat crecerá tanto que consumirá todo el presupuesto de OpenAI en un solo mensaje. *Solución:* Usar `ConversationSummaryMemory` (que resume el chat antiguo) o `ConversationBufferWindowMemory` (que solo recuerda los últimos K mensajes).
4. **Desacoplamiento Obligatorio**: Hoy el script indexa en FAISS y luego levanta el chat. En vida real, la ingesta (lectura de PDFs) debe ser un proceso Backend/CronJob aislado, y el Agente debe vivir detrás de una API REST.
