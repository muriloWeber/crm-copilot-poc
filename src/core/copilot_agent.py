# src/core/copilot_agent.py

import os
import re
import logging
from typing import TypedDict, List, Annotated, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub # Para carregar o prompt do LangChain Hub
from langchain.prompts import ChatPromptTemplate
import httpx # Para o cliente HTTP customizado com verify=False
from langchain_core.documents import Document
import operator # <--- AGORA NO LUGAR CERTO!

# Importar get_chroma_instance para ter acesso consistente ao vetor store
from src.data_ingestion.incremental_ingestor import get_chroma_instance

# Configura o logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Definição do Estado do Agente ---
class AgentState(TypedDict):
    """
    Representa o estado do agente de IA durante a orquestração.
    Os atributos são usados para passar informações entre os diferentes nós (funções)
    no gráfico de LangGraph.
    """
    question: str  # A pergunta original do usuário
    context: List[Document]  # Chunks de documentos relevantes recuperados
    source_docs: List[Dict[str, Any]] # Metadados detalhados das fontes
    answer: str  # A resposta gerada pelo LLM
    messages: Annotated[List[BaseMessage], operator.add] # Histórico de mensagens
    filters: Dict[str, Any] # Filtros de metadados para a recuperação RAG

# --- Modelos de Linguagem ---
def get_llm():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada para LLM.")

    # Configura o cliente HTTP para ignorar a verificação SSL
    custom_http_client = httpx.Client(verify=False)

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        http_client=custom_http_client,
        streaming=True # Habilita o streaming para respostas
    )

# Inicializa o LLM uma única vez
llm = get_llm()

# --- Nós do Grafo (Funções de Estado) ---

def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó responsável por recuperar chunks relevantes do ChromaDB
    usando a pergunta do usuário e aplicando filtros de metadados.
    """
    logger.info("Executando retrieve_context_node...")
    question = state["question"]
    filters = state.get("filters", {}) # Pega os filtros do estado

    # Obter a instância do ChromaDB
    vector_store = get_chroma_instance()

    # Construir o filtro 'where' para o retriever do ChromaDB
    # O ChromaDB espera um dicionário simples para o filtro 'where'
    where_filter = {}
    if filters:
        if filters.get("client_name"):
            # Para correspondência exata, o valor deve ser uma string, não uma lista
            where_filter["client_name"] = filters["client_name"].upper() 
        if filters.get("project_code"):
            where_filter["project_code"] = filters["project_code"].upper()
        if filters.get("doc_type"):
            where_filter["doc_type"] = filters["doc_type"].upper()
        
        logger.info(f"Filtros aplicados na recuperação: {where_filter}")

    # Configurar o retriever com o filtro de metadados
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 5,  # Número de chunks a recuperar
            "filter": where_filter # Aplica o filtro de metadados aqui
        }
    )

    docs = retriever.invoke(question) # Invoca o retriever para buscar documentos

    if not docs:
        logger.warning(f"Nenhum documento encontrado para a pergunta '{question}' com os filtros '{where_filter}'.")
        return {"context": [], "source_docs": [], "answer": "Não encontrei informações relevantes para sua pergunta com os filtros especificados."}

    source_docs = []
    for doc in docs:
        metadata = doc.metadata
        source_docs.append({
            "source": metadata.get("source", "N/A"),
            "client_name": metadata.get("client_name", "N/A"),
            "doc_type": metadata.get("doc_type", "N/A"),
            "project_code": metadata.get("project_code", "N/A"),
            "page_number": metadata.get("page_number", "N/A")
        })
    
    logger.info(f"Documentos recuperados: {len(docs)}")
    return {"context": docs, "source_docs": source_docs}


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó responsável por gerar uma resposta usando o LLM,
    baseado na pergunta do usuário e nos chunks de contexto recuperados.
    """
    logger.info("Executando generate_response_node...")
    question = state["question"]
    context = state["context"]
    messages = state["messages"]

    # Carregar um prompt de RAG (pode ser ajustado ou substituído)
    # Exemplo: um prompt simples para RAG
    rag_prompt = ChatPromptTemplate.from_template(
        """Você é um assistente útil e preciso que responde perguntas sobre documentos de projetos.
        Responda à pergunta do usuário APENAS com base no CONTEXTO fornecido.
        Se a resposta não puder ser encontrada no contexto fornecido, diga que não tem informações suficientes.
        Seja conciso, profissional e direto.

        CONTEXTO:
        {context}

        PERGUNTA DO USUÁRIO: {question}
        """
    )
    
    # Adicionar histórico de chat ao prompt, se houver
    # Apenas as últimas mensagens para não estourar o limite de tokens
    chat_history_str = ""
    for msg in messages[-4:-1]: # Pega as últimas 3 mensagens (excluindo a atual do usuário)
        if isinstance(msg, HumanMessage):
            chat_history_str += f"HUMAN: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            chat_history_str += f"ASSISTANT: {msg.content}\n"

    full_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_prompt.messages[0].prompt.template), # O system message do RAG
        ("human", "{chat_history}\nPERGUNTA DO USUÁRIO: {question}\nCONTEXTO: {context}") # Inclui histórico
    ])


    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Adicionando um fallback para o caso de o contexto estar vazio
    if not context:
        answer = "Não encontrei informações relevantes para sua pergunta na base de conhecimento."
    else:
        answer = rag_chain.invoke({"question": question, "context": context}) # Passa o contexto formatado

    logger.info(f"Resposta gerada: {answer[:100]}...")
    return {"answer": answer}

def format_citation_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó responsável por formatar a resposta final, incluindo as citações das fontes.
    """
    logger.info("Executando format_citation_node...")
    answer = state["answer"]
    source_docs = state["source_docs"]

    citations = []
    seen_sources = set() # Para evitar citações duplicadas
    for doc in source_docs:
        # Usamos uma combinação de source e page_number para identificar unicamente uma fonte citada
        source_id = f"{doc['source']}-{doc['page_number']}"
        if source_id not in seen_sources:
            citations.append(
                f"Documento: {doc['source']} "
                f"Cliente: {doc['client_name']} "
                f"Tipo: {doc['doc_type']} "
                f"Página: {doc['page_number']}"
            )
            seen_sources.add(source_id)

    if citations:
        # Formata as fontes consultadas com um cabeçalho para cada citação
        citations_str = "\n\n".join([f"Fontes Consultadas: {c}" for c in citations])
        final_answer = f"{answer}\n\n{citations_str}"
    else:
        final_answer = answer # Sem citações se não houver documentos fonte

    logger.info("Resposta final com citações formatada.")
    return {"answer": final_answer}

def format_docs(docs: List[Document]) -> str:
    """Formata os documentos para serem injetados no prompt do LLM."""
    return "\n\n".join(doc.page_content for doc in docs)
