"""Gestión de memoria conversacional para el sistema de chat."""

from langchain.memory import ConversationBufferMemory


def create_memory() -> ConversationBufferMemory:
    """Crea una instancia de memoria conversacional.

    ConversationBufferMemory almacena el historial completo de la conversación,
    permitiendo que el agente recuerde preguntas y respuestas anteriores
    dentro de la misma sesión.

    - memory_key: clave usada para inyectar el historial en el prompt del agente.
    - return_messages: retorna mensajes como objetos (ChatMessage) en vez de texto plano.
    """

    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
