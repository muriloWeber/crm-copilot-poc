#src/core/copilot_agent.py

import os
import re
import logging
from typing import TypedDict, List, Annotated, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.prompts import ChatPromptTemplate
import httpx
from langchain_core.documents import Document
import operator # AGORA NO LUGAR CERTO!
from langgraph.graph import StateGraph, END # Importar StateGraph e END aqui para o build_graph

# Importar get_chroma_instance e get_known_entities para ter acesso consistente ao vetor store e às entidades conhecidas
from src.data_ingestion.incremental_ingestor import get_chroma_instance, get_known_entities 

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
    # Opcional: Para ambientes de produção, considere configurar a verificação SSL corretamente
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

# Variável global para armazenar as entidades conhecidas (carregadas uma vez)
_cached_known_entities: Dict[str, List[str]] = {}

def load_known_entities_once():
    """
    Carrega as entidades conhecidas do ChromaDB apenas uma vez e as armazena em cache.
    """
    global _cached_known_entities
    if not _cached_known_entities:
        _cached_known_entities = get_known_entities()
        logger.info(f"Entidades conhecidas carregadas com sucesso: {_cached_known_entities}")

# --- Nós do Grafo (Funções de Estado) ---

def extract_filters_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó responsável por extrair metadados (client_name, doc_type, project_code)
    da pergunta do usuário através de correspondência com entidades conhecidas no ChromaDB.
    """
    logger.info("Executando extract_filters_node (baseado em entidades conhecidas)...")
    
    # Garante que as entidades sejam carregadas no cache na primeira execução
    load_known_entities_once() 
    known_entities = _cached_known_entities
    
    question = state["question"].upper() # Normaliza a pergunta para maiúsculas para comparação
    extracted_filters = {}

    # Extrair Client Name
    for client_name in known_entities.get("client_names", []):
        # Usamos uma busca por palavra inteira ou "NOME DO CLIENTE" para evitar falsos positivos
        if re.search(r'\b' + re.escape(client_name) + r'\b', question):
            extracted_filters["client_name"] = client_name
            logger.debug(f"Filtro de client_name detectado: {client_name}")
            break # Assume um cliente por pergunta

    # Extrair Project Code
    for project_code in known_entities.get("project_codes", []):
        if re.search(r'\b' + re.escape(project_code) + r'\b', question):
            extracted_filters["project_code"] = project_code
            logger.debug(f"Filtro de project_code detectado: {project_code}")
            break # Assume um projeto por pergunta
            
    # Extrair Doc Type (MIT041, etc.)
    for doc_type in known_entities.get("doc_types", []):
        if re.search(r'\b' + re.escape(doc_type) + r'\b', question):
            extracted_filters["doc_type"] = doc_type
            logger.debug(f"Filtro de doc_type detectado: {doc_type}")
            break # Assume um tipo de doc por pergunta

    logger.info(f"Filtros extraídos da pergunta: {extracted_filters if extracted_filters else 'Nenhum'}")

    return {"filters": extracted_filters}


def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó responsável por recuperar chunks relevantes do ChromaDB
    usando a pergunta do usuário e aplicando filtros de metadados.
    """
    logger.info("Executando retrieve_context_node...")
    question = state["question"]
    filters = state.get("filters", {}) # Pega os filtros do estado (agora preenchidos pelo extract_filters_node)

    # Obter a instância do ChromaDB
    vector_store = get_chroma_instance()

    # Construir o filtro 'where' para o retriever do ChromaDB
    where_conditions = []
    if filters.get("client_name"):
        where_conditions.append({"client_name": filters["client_name"]})
    if filters.get("project_code"):
        where_conditions.append({"project_code": filters["project_code"]})
    if filters.get("doc_type"):
        where_conditions.append({"doc_type": filters["doc_type"]})
    
    where_filter = {}
    if where_conditions:
        if len(where_conditions) == 1:
            where_filter = where_conditions[0] # Se houver apenas uma condição, use-a diretamente
        else:
            # Se houver múltiplas condições, combine-as usando o operador "$and" do ChromaDB
            where_filter = {"$and": where_conditions} 
    
    # Prepara os search_kwargs
    search_kwargs = {"k": 5} # Número de chunks a recuperar
    
    # Adiciona o filtro 'where' apenas se ele não estiver vazio
    if where_filter: 
        search_kwargs["filter"] = where_filter
        logger.info(f"Filtros aplicados na recuperação: {where_filter}")
    else:
        logger.info("Nenhum filtro de metadados extraído da pergunta. Realizando busca geral.")

    # Configurar o retriever com ou sem o filtro de metadados
    retriever = vector_store.as_retriever(
        search_kwargs=search_kwargs
    )

    docs = retriever.invoke(question) # Invoca o retriever para buscar documentos

    # Logging para depuração detalhada dos documentos recuperados
    logger.debug(f"[DEBUG_RETRIEVAL] Retrieved {len(docs)} documents from ChromaDB.")
    for i, doc in enumerate(docs):
        # A informação de score só aparece se for uma query, não uma busca direta com 'where'
        # Em alguns casos, Chroma retorna '_score', em outros não quando filtros são aplicados.
        score_info = f"(Score: {doc.metadata.get('_score', 'N/A')})" if '_score' in doc.metadata else ""
        logger.debug(f"[DEBUG_RETRIEVAL] Document {i+1} {score_info} Metadata: {doc.metadata}")
        logger.debug(f"[DEBUG_RETRIEVAL] Document {i+1} Content snippet: {doc.page_content[:200]}...")

    if not docs:
        logger.warning(f"Nenhum documento encontrado para a pergunta '{question}' com os filtros '{where_filter}'.")
        # Retorna uma mensagem amigável se nenhum documento for encontrado
        return {"context": [], "source_docs": [], "answer": "Não encontrei informações relevantes para sua pergunta com os filtros especificados." if where_filter else "Não encontrei informações relevantes para sua pergunta na base de conhecimento."}

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

    logger.debug(f"[DEBUG_LLM_CONTEXT] Context received by LLM ({len(context)} chunks):")
    for i, doc in enumerate(context):
        logger.debug(f"[DEBUG_LLM_CONTEXT] Chunk {i+1} Metadata: {doc.metadata}")
        logger.debug(f"[DEBUG_LLM_CONTEXT] Chunk {i+1} Content snippet: {doc.page_content[:200]}...")

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
    chat_history_str = ""
    # Pega as últimas 3 mensagens (excluindo a atual do usuário)
    # A lógica aqui pode precisar de refinamento para um histórico mais inteligente
    # Mas para o escopo atual, manter como base
    for msg in messages[-4:-1]: 
        if isinstance(msg, HumanMessage):
            chat_history_str += f"HUMAN: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            chat_history_str += f"ASSISTANT: {msg.content}\n"

    # Nota: A maneira como o LangChain lida com histórico pode variar.
    # Para este prompt simples, `rag_prompt` já inclui {context} e {question}.
    # Se quiser incorporar `chat_history_str` diretamente, o `rag_prompt` precisaria de uma variável {chat_history}.
    # Por enquanto, mantemos o `rag_prompt` simples como definido.

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
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
        citations_str = "\n\n".join([f"Fontes Consultadas: {c}" for c in citations])
        final_answer = f"{answer}\n\n{citations_str}"
    else:
        final_answer = answer # Sem citações se não houver documentos fonte

    logger.info("Resposta final com citações formatada.")
    return {"answer": final_answer}

def format_docs(docs: List[Document]) -> str:
    """Formata os documentos para serem injetados no prompt do LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- Construção do Grafo ---
def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Nó para extrair filtros da pergunta
    workflow.add_node("extract_filters", extract_filters_node)
    
    # 2. Nó para recuperar o contexto (agora com filtros)
    workflow.add_node("retrieve_context", retrieve_context_node)
    
    # 3. Nó para gerar a resposta
    workflow.add_node("generate_response", generate_response_node)
    
    # 4. Nó para formatar as citações
    workflow.add_node("format_citation", format_citation_node)

    # Definir a sequência de execução
    workflow.set_entry_point("extract_filters") # Começa extraindo os filtros
    workflow.add_edge("extract_filters", "retrieve_context") # Depois de extrair, recupera
    workflow.add_edge("retrieve_context", "generate_response") # Depois de recuperar, gera a resposta
    workflow.add_edge("generate_response", "format_citation") # Depois de gerar, formata a citação
    workflow.add_edge("format_citation", END) # Fim do processo

    return workflow.compile()