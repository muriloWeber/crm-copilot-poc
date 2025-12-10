"""
Módulo principal para a construção e orquestração do agente TCRM Copilot usando LangGraph.

Define o estado do agente, os nós de processamento (extração de filtros, recuperação de contexto,
geração de resposta e formatação de citação) e o fluxo do grafo.
Utiliza OpenAI para LLM e Embeddings, e ChromaDB para a base de conhecimento vetorial.
"""

import os
import re
import logging
from typing import TypedDict, List, Annotated, Dict, Any, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import httpx # Necessário para configurar o cliente HTTP com verificação SSL
import operator # Usado para a anotação Annotated[List[BaseMessage], operator.add]
from langgraph.graph import StateGraph, END # Componentes essenciais do LangGraph
from langchain_core.documents import Document # Para tipagem dos documentos recuperados

# Importar get_chroma_instance e get_known_entities para ter acesso consistente ao vetor store e às entidades conhecidas
from src.data_ingestion.incremental_ingestor import get_chroma_instance, get_known_entities 

# --- Configuração de Logging ---
# Configura o logger para este módulo, garantindo que as mensagens de log sejam formatadas de forma consistente.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Definição do Estado do Agente ---
class AgentState(TypedDict):
    """
    Representa o estado transitório do agente de IA durante a orquestração do LangGraph.
    Os atributos são usados para passar informações e o progresso entre os diferentes nós (funções)
    no gráfico de execução do agente.

    Atributos:
        question (str): A pergunta original submetida pelo usuário.
        context (List[Document]): Uma lista de objetos Document (chunks de texto)
                                  recuperados da base de conhecimento relevante para a pergunta.
        source_docs (List[Dict[str, Any]]): Metadados detalhados das fontes dos documentos
                                           recuperados, usados para citação.
        answer (str): A resposta gerada pelo LLM com base na pergunta e no contexto.
        messages (Annotated[List[BaseMessage], operator.add]): Histórico completo de mensagens
                                                               da conversação (usuário e assistente).
                                                               'operator.add' indica que novas mensagens
                                                               devem ser adicionadas à lista existente.
        filters (Dict[str, Any]): Dicionário de filtros de metadados extraídos da pergunta do usuário,
                                  utilizados para refinar a busca na base de conhecimento.
    """
    question: str
    context: List[Document]
    source_docs: List[Dict[str, Any]]
    answer: str
    messages: Annotated[List[BaseMessage], operator.add]
    filters: Dict[str, Any]

# --- Modelos de Linguagem ---
def get_llm() -> ChatOpenAI:
    """
    Inicializa e retorna uma instância do modelo de linguagem OpenAI (ChatOpenAI).

    Configura o modelo, a chave da API, a base_url (se aplicável) e um cliente HTTP
    customizado para lidar com a verificação SSL.

    Raises:
        ValueError: Se a variável de ambiente OPENAI_API_KEY não estiver configurada.

    Returns:
        ChatOpenAI: Uma instância configurada do modelo de linguagem.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        logger.error("Variável de ambiente OPENAI_API_KEY não configurada. O LLM não será inicializado.")
        raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada para LLM.")

    # Configura o cliente HTTP para ignorar a verificação SSL.
    # ATENÇÃO: 'verify=False' desabilita a verificação de certificados SSL, o que pode
    # ser um RISCO DE SEGURANÇA em ambientes de produção. É usado aqui para PoC/testes
    # com proxies internos que podem não ter certificados confiáveis.
    # Para produção, configure a verificação SSL corretamente ou use certificados CA.
    custom_http_client = httpx.Client(verify=False)

    return ChatOpenAI(
        model="gpt-4o-mini", # Modelo selecionado para a PoC
        temperature=0.2,     # Controla a aleatoriedade da saída (0.0 para mais determinístico)
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        http_client=custom_http_client,
        streaming=True       # Habilita o streaming para respostas em tempo real
    )

# Inicializa o LLM uma única vez ao carregar o módulo para evitar re-inicializações.
llm = get_llm()

# --- Cache de Entidades Conhecidas ---
# Variável global para armazenar as entidades conhecidas (nomes de clientes, códigos de projeto, etc.).
# É carregada apenas uma vez para otimizar o desempenho na extração de filtros.
_cached_known_entities: Dict[str, List[str]] = {}

def load_known_entities_once():
    """
    Carrega as entidades conhecidas (clientes, projetos, tipos de documento)
    do ChromaDB apenas uma vez e as armazena em cache global.
    Isso otimiza o desempenho, evitando múltiplas leituras do DB para cada requisição.
    """
    global _cached_known_entities
    if not _cached_known_entities:
        _cached_known_entities = get_known_entities()
        logger.info(f"Entidades conhecidas carregadas com sucesso: {list(_cached_known_entities.keys())}")
    else:
        logger.debug("Entidades conhecidas já carregadas no cache.")

# --- Nós do Grafo (Funções de Estado) ---

def extract_filters_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó de processamento responsável por extrair filtros de metadados da pergunta do usuário.
    Procura por nomes de clientes, códigos de projeto e tipos de documento conhecidos
    na pergunta para refinar a busca RAG.

    Assume que apenas um filtro de cada tipo (cliente, projeto, doc_type) será extraído por pergunta.

    Args:
        state (AgentState): O estado atual do agente, contendo a pergunta do usuário.

    Returns:
        Dict[str, Any]: Um dicionário com o campo 'filters' atualizado no estado,
                        contendo os metadados extraídos.
    """
    logger.info("Executando extract_filters_node (extração de filtros de metadados da pergunta)...")
    
    load_known_entities_once() # Garante que as entidades estejam no cache
    known_entities = _cached_known_entities
    
    question = state["question"].upper() # Normaliza a pergunta para maiúsculas para comparação consistente
    extracted_filters = {}

    # Extrai 'client_name' da pergunta se corresponder a uma entidade conhecida.
    for client_name in known_entities.get("client_names", []):
        # Usa regex para buscar a palavra inteira, evitando falsos positivos.
        if re.search(r'\b' + re.escape(client_name) + r'\b', question):
            extracted_filters["client_name"] = client_name
            logger.debug(f"Filtro 'client_name' detectado: {client_name}")
            break # Assume um cliente por pergunta

    # Extrai 'project_code' da pergunta se corresponder a uma entidade conhecida.
    for project_code in known_entities.get("project_codes", []):
        if re.search(r'\b' + re.escape(project_code) + r'\b', question):
            extracted_filters["project_code"] = project_code
            logger.debug(f"Filtro 'project_code' detectado: {project_code}")
            break # Assume um código de projeto por pergunta
            
    # Extrai 'doc_type' da pergunta se corresponder a uma entidade conhecida.
    for doc_type in known_entities.get("doc_types", []):
        if re.search(r'\b' + re.escape(doc_type) + r'\b', question):
            extracted_filters["doc_type"] = doc_type
            logger.debug(f"Filtro 'doc_type' detectado: {doc_type}")
            break # Assume um tipo de documento por pergunta

    logger.info(f"Filtros extraídos da pergunta: {extracted_filters if extracted_filters else 'Nenhum filtro identificado.'}")

    return {"filters": extracted_filters}


def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó de processamento responsável por recuperar chunks de documentos relevantes
    do ChromaDB. Utiliza a pergunta do usuário e os filtros de metadados extraídos
    para realizar uma busca mais precisa.

    Args:
        state (AgentState): O estado atual do agente, contendo a pergunta e os filtros.

    Returns:
        Dict[str, Any]: Um dicionário com os campos 'context' (documentos recuperados)
                        e 'source_docs' (metadados para citação) atualizados no estado.
    """
    logger.info("Executando retrieve_context_node (recuperação de contexto da base de vetores)...")
    question = state["question"]
    filters = state.get("filters", {}) # Obtém os filtros de metadados do estado atual

    vector_store = get_chroma_instance() # Obtém a instância do ChromaDB

    # Constrói o dicionário de filtro 'where' para a busca no ChromaDB.
    # Combina múltiplas condições com o operador "$and" para filtragem precisa.
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
            where_filter = where_conditions[0] # Se apenas uma condição, usa-a diretamente
        else:
            # Se múltiplas condições, as combina com $and (requisito do ChromaDB para AND lógico)
            where_filter = {"$and": where_conditions} 
    
    # Prepara os parâmetros de busca para o retriever.
    # 'k' define o número de chunks mais relevantes a serem recuperados.
    search_kwargs = {"k": 5} # Valor padrão, pode ser ajustado para otimização
    
    # Aplica o filtro 'where' somente se ele foi construído.
    if where_filter: 
        search_kwargs["filter"] = where_filter
        logger.info(f"Filtros aplicados na recuperação: {where_filter}")
    else:
        logger.info("Nenhum filtro de metadados extraído. Realizando busca geral por similaridade.")

    # Configura o retriever do ChromaDB com ou sem o filtro de metadados.
    retriever = vector_store.as_retriever(
        search_kwargs=search_kwargs
    )

    docs = retriever.invoke(question) # Executa a busca pelos documentos mais relevantes

    # Logging detalhado para depuração dos documentos recuperados, incluindo metadados e snippet do conteúdo.
    logger.debug(f"[DEBUG_RETRIEVAL] Recuperados {len(docs)} documentos do ChromaDB.")
    for i, doc in enumerate(docs):
        # A informação de score pode não estar presente se a busca for puramente por filtro 'where'.
        score_info = f"(Score: {doc.metadata.get('_score', 'N/A')})" if '_score' in doc.metadata else ""
        logger.debug(f"[DEBUG_RETRIEVAL] Documento {i+1} {score_info} Metadados: {doc.metadata}")
        logger.debug(f"[DEBUG_RETRIEVAL] Documento {i+1} Trecho: {doc.page_content[:200]}...")

    # Se nenhum documento for encontrado, retorna uma resposta padrão para o usuário.
    if not docs:
        logger.warning(f"Nenhum documento encontrado para a pergunta '{question}' com os filtros '{where_filter}'.")
        return {"context": [], "source_docs": [], "answer": "Não encontrei informações relevantes para sua pergunta com os filtros especificados na base de conhecimento." if where_filter else "Não encontrei informações relevantes para sua pergunta na base de conhecimento."}

    # Extrai e formata os metadados dos documentos recuperados para citação.
    source_docs_for_citation = []
    for doc in docs:
        metadata = doc.metadata
        source_docs_for_citation.append({
            "source": metadata.get("source", "N/A"), # Nome do arquivo
            "client_name": metadata.get("client_name", "N/A"),
            "doc_type": metadata.get("doc_type", "N/A"),
            "project_code": metadata.get("project_code", "N/A"),
            "page_number": metadata.get("page_number", "N/A") # Número da página
        })
    
    logger.info(f"Documentos contextuais recuperados: {len(docs)}")
    return {"context": docs, "source_docs": source_docs_for_citation}


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó de processamento responsável por gerar uma resposta concisa e profissional
    usando o LLM (Large Language Model) da OpenAI. A resposta é baseada na pergunta
    do usuário e nos chunks de contexto recuperados, garantindo que o LLM não alucine.

    Args:
        state (AgentState): O estado atual do agente, contendo a pergunta, o contexto
                            e o histórico de mensagens.

    Returns:
        Dict[str, Any]: Um dicionário com o campo 'answer' atualizado no estado,
                        contendo a resposta gerada pelo LLM.
    """
    logger.info("Executando generate_response_node (geração de resposta pelo LLM)...")
    question = state["question"]
    context = state["context"]
    # messages = state["messages"] # 'messages' está disponível, mas não usado no prompt atual.

    logger.debug(f"[DEBUG_LLM_CONTEXT] Contexto recebido pelo LLM ({len(context)} chunks):")
    for i, doc in enumerate(context):
        logger.debug(f"[DEBUG_LLM_CONTEXT] Chunk {i+1} Metadados: {doc.metadata}")
        logger.debug(f"[DEBUG_LLM_CONTEXT] Chunk {i+1} Trecho: {doc.page_content[:200]}...")

    # Define o template do prompt para o RAG. Instruções são claras para evitar alucinações.
    # O prompt solicita que o LLM use APENAS o contexto fornecido.
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
    
    # ATENÇÃO: O histórico de chat (state["messages"]) não está sendo explicitamente
    # passado para o 'rag_prompt' neste momento, pois o template atual não possui um placeholder para ele.
    # Se for desejado que o LLM considere o histórico de conversas para respostas encadeadas,
    # o 'rag_prompt' e o RunnablePassthrough precisarão ser modificados para incluir e formatar o 'messages'.
    
    # Cria uma cadeia de execução para o RAG:
    # 1. Atribui o contexto formatado.
    # 2. Injeta o contexto e a pergunta no prompt.
    # 3. Passa o prompt para o LLM.
    # 4. Analisa a saída do LLM como string.
    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    if not context:
        # Retorna uma resposta padrão se nenhum contexto foi fornecido pelo nó anterior.
        answer = "Não encontrei informações relevantes na base de conhecimento para responder sua pergunta."
    else:
        # Invoca a cadeia RAG com a pergunta e o contexto recuperado.
        answer = rag_chain.invoke({"question": question, "context": context})

    logger.info(f"Resposta gerada pelo LLM (trecho): {answer[:100]}...")
    return {"answer": answer}

def format_citation_node(state: AgentState) -> Dict[str, Any]:
    """
    Nó de pós-processamento responsável por formatar a resposta final,
    incluindo as citações das fontes de onde a informação foi extraída.
    Evita citações duplicadas.

    Args:
        state (AgentState): O estado atual do agente, contendo a resposta gerada
                            e os metadados das fontes.

    Returns:
        Dict[str, Any]: Um dicionário com o campo 'answer' atualizado no estado,
                        contendo a resposta final com as citações.
    """
    logger.info("Executando format_citation_node (formatação de citações)...")
    answer = state["answer"]
    source_docs = state["source_docs"]

    citations = []
    seen_sources = set() # Usado para garantir que cada fonte única seja citada apenas uma vez

    # Itera sobre os metadados das fontes para construir as citações.
    for doc in source_docs:
        # Cria um ID único para a fonte (arquivo + página) para verificar duplicação.
        source_id = f"{doc['source']}-{doc['page_number']}"
        if source_id not in seen_sources:
            citations.append(
                f"Documento: {doc['source']} | "
                f"Cliente: {doc['client_name']} | "
                f"Tipo: {doc['doc_type']} | "
                f"Página: {doc['page_number']}"
            )
            seen_sources.add(source_id)

    # Adiciona as citações à resposta final se houver.
    if citations:
        citations_str = "\n\n" + "\n".join([f"Fontes Consultadas: {c}" for c in citations])
        final_answer = f"{answer}{citations_str}"
    else:
        final_answer = answer # Retorna a resposta sem modificação se não houver fontes citáveis

    logger.info("Resposta final com citações formatada.")
    return {"answer": final_answer}

def format_docs(docs: List[Document]) -> str:
    """
    Função auxiliar para formatar uma lista de objetos Document em uma única string.
    Cada Document.page_content é unido por uma quebra de linha dupla,
    adequado para ser injetado no prompt do LLM como contexto.

    Args:
        docs (List[Document]): Uma lista de objetos Document.

    Returns:
        str: Uma string contendo o conteúdo concatenado de todos os documentos.
    """
    return "\n\n".join(doc.page_content for doc in docs)

# --- Construção do Grafo ---
def build_graph() -> StateGraph:
    """
    Constrói e compila o grafo de orquestração do agente TCRM Copilot usando LangGraph.
    Define a sequência de nós de processamento e suas transições.

    Returns:
        StateGraph: O grafo compilado pronto para ser invocado.
    """
    logger.info("Construindo o grafo de orquestração do LangGraph...")
    workflow = StateGraph(AgentState)

    # Adiciona cada nó de processamento ao fluxo de trabalho.
    workflow.add_node("extract_filters", extract_filters_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("format_citation", format_citation_node)

    # Define a sequência de execução do grafo.
    # O processo começa com a extração de filtros.
    workflow.set_entry_point("extract_filters")
    # Transições entre os nós.
    workflow.add_edge("extract_filters", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", "format_citation")
    # O nó 'format_citation' marca o fim de uma execução.
    workflow.add_edge("format_citation", END)

    logger.info("Grafo de orquestração compilado com sucesso.")
    return workflow.compile()