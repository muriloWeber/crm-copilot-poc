import os
import re
import sys
import pathlib
from typing import TypedDict, List, Optional, Any, Dict, Set
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
import httpx
from chromadb.config import Settings
from dotenv import load_dotenv
import logging

# Configura o logger
logging.basicConfig(level=logging.INFO)

# Carrega as variáveis de ambiente do .env
load_dotenv()

# --- Constantes de Configuração ---
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vector_db', 'chroma_db'))
CHROMA_COLLECTION_NAME = 'tcrm_copilot_kb'
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
CONTEXT_EXPANSION_WINDOW = 2 # Quantos chunks antes e depois do chunk semente buscar

# --- 1. Definição do AgentState ---
class AgentState(TypedDict):
    """
    Representa o estado do agente Copilot.
    question: A pergunta original do usuário.
    context: Uma lista de documentos (chunks) recuperados da base de conhecimento.
    source_docs: Uma lista de dicionários contendo metadados detalhados dos chunks.
    answer: A resposta gerada pelo LLM.
    messages: List[BaseMessage] # Histórico de mensagens da conversa
    filters: Optional[dict] # Adicionado aqui para permitir filtros no estado
    """
    question: str
    context: List[Document]
    source_docs: List[dict]
    answer: str
    messages: List[BaseMessage]
    filters: Optional[dict]

# --- 2. Configurações e Inicialização do LLM e Embeddings/Vector Store ---

def get_llm():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada.")

    # Configura o cliente HTTP para ignorar a verificação SSL, essencial para o proxy da TOTVS
    custom_http_client = httpx.Client(verify=False)

    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        temperature=0,
        http_client=custom_http_client
    )

llm = get_llm()

def get_embeddings_model():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada para embeddings.")

    custom_http_client = httpx.Client(verify=False)

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        http_client=custom_http_client
    )

embeddings_model = get_embeddings_model()

_vector_store: Optional[Chroma] = None

def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        if not os.path.exists(CHROMA_DB_DIRECTORY):
            raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DB_DIRECTORY}. "
                                    f"Please run src/data_ingestion/knowledge_base_builder.py first.")
        _vector_store = Chroma(
            persist_directory=str(CHROMA_DB_DIRECTORY), # Usar str() é mais seguro para Chroma
            embedding_function=embeddings_model,
            collection_name=CHROMA_COLLECTION_NAME
        )
        logging.info(f"ChromaDB carregado de {CHROMA_DB_DIRECTORY} com coleção '{CHROMA_COLLECTION_NAME}'.")
    return _vector_store

try:
    _vector_store = get_vector_store()
except FileNotFoundError as e:
    logging.warning(f"{e}. O Copilot pode não funcionar corretamente até que a base de conhecimento seja construída.")
    _vector_store = None

# --- FUNÇÃO PARA CARREGAR NOMES DE CLIENTES DINAMICAMENTE DO CHROMADB ---
_known_client_names: Set[str] = set()

def _load_known_client_names_from_chroma():
    global _known_client_names
    if _vector_store:
        logging.info("Carregando nomes de clientes conhecidos do ChromaDB...")
        try:
            chroma_client_native = _vector_store._client # Acessa o cliente nativo do chromadb
            collection = chroma_client_native.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
            
            count = collection.count()
            if count == 0:
                logging.info("Coleção ChromaDB vazia. Nenhum cliente para carregar.")
                _known_client_names = set()
                return

            # Para coleções muito grandes, 'get' com 'limit=count' pode ser ineficiente.
            # Uma alternativa seria iterar ou usar 'distinct' se suportado.
            # Para PoC e nosso volume, 'get' deve funcionar.
            results = collection.get(
                limit=count, 
                include=['metadatas']
            )
            
            unique_clients = set()
            for metadata_dict in results.get('metadatas', []):
                client_name = metadata_dict.get('client_name')
                if client_name and client_name.strip() and client_name.upper() != "UNKNOWN CLIENT":
                    unique_clients.add(client_name.upper())
            _known_client_names = unique_clients
            logging.info(f"Nomes de clientes carregados: {_known_client_names}")
        except Exception as e:
            logging.error(f"Falha ao carregar nomes de clientes do ChromaDB: {e}. Verifique a integridade da sua base ChromaDB ou os metadados.", exc_info=True)
            _known_client_names = set()
    else:
        logging.warning("ChromaDB não carregado. Não foi possível carregar nomes de clientes.")

if _vector_store:
    _load_known_client_names_from_chroma()


# --- FUNÇÃO DE DETECÇÃO DE CLIENTE/PROJETO/DOC_TYPE OTIMIZADA E COM FALLBACK ---
def _detect_client_or_project_in_question(question: str) -> Dict[str, str]:
    """
    Detecta se um nome de cliente, código de projeto ou tipo de documento está presente na pergunta do usuário.
    Prioriza a busca por nomes de clientes conhecidos carregados dinamicamente do ChromaDB.
    Retorna um dicionário com os filtros detectados, alinhados aos metadados (todos em UPPERCASE).
    """
    detected_filters = {}
    question_lower = question.lower()
    logging.debug(f"DEBUG (Detector): Questão recebida: '{question}'")

    if _known_client_names:
        for client_name in _known_client_names:
            client_name_lower = client_name.lower()
            search_pattern_regex = r'\b' + re.escape(client_name_lower) + r'\b'
            logging.debug(f"DEBUG (Detector): Tentando padrão regex '{search_pattern_regex}' para cliente '{client_name}'")
            if re.search(search_pattern_regex, question_lower):
                detected_filters["client_name"] = client_name
                logging.debug(f"DEBUG (Detector): Cliente detectado por regex e lista de nomes conhecidos: '{detected_filters['client_name']}'")
                break
            else:
                if client_name_lower in question_lower:
                    detected_filters["client_name"] = client_name
                    logging.debug(f"DEBUG (Detector): Cliente detectado por busca direta 'in' (fallback robusto): '{detected_filters['client_name']}'")
                    break
                else:
                    logging.debug(f"DEBUG (Detector): Cliente '{client_name}' não encontrado na questão (busca direta ou regex).")
    else:
        logging.warning("DEBUG (Detector): _known_client_names não carregada ou vazia. Não foi possível usar a detecção dinâmica de clientes.")

    project_code_match_d = re.search(r"D\d{9,15}", question, re.IGNORECASE)
    if project_code_match_d:
        detected_filters["project_code"] = project_code_match_d.group(0).upper()
        logging.debug(f"DEBUG (Detector): Código de Projeto detectado: '{detected_filters['project_code']}'")

    doc_type_match = re.search(r"(MIT\d{3})", question, re.IGNORECASE)
    if doc_type_match:
        detected_filters["doc_type"] = doc_type_match.group(1).upper()
        logging.debug(f"DEBUG (Detector): Tipo de Documento detectado: '{detected_filters['doc_type']}'")

    logging.debug(f"DEBUG (Detector): Filtros finais detectados: {detected_filters}")
    return detected_filters


# --- 3. Definição dos Nós (Nodes) ---

def retrieve_context_node(state: AgentState):
    """
    Nó para recuperar chunks relevantes da base de conhecimento com expansão de contexto.
    """
    global _vector_store

    if _vector_store is None:
        try:
            _vector_store = get_vector_store()
        except FileNotFoundError as e:
            raise RuntimeError(f"ChromaDB não disponível para recuperação: {e}. "
                               f"Certifique-se de que src/data_ingestion/knowledge_base_builder.py foi executado com sucesso.")

    logging.info(f"Executando retrieve_context_node para a pergunta: {state['question']}")
    question = state["question"]
    
    detected_filters_from_question = _detect_client_or_project_in_question(question)
    logging.debug(f"Filtros detectados da pergunta: {detected_filters_from_question}")
    
    chroma_filter_conditions = []
    expected_client_name = None
    if "client_name" in detected_filters_from_question:
        expected_client_name = detected_filters_from_question["client_name"]
        chroma_filter_conditions.append({"client_name": expected_client_name}) 
        logging.debug(f"  -> Filtro de cliente DETECTADO para ChromaDB (tentativa automática): '{expected_client_name}'")
    
    if "project_code" in detected_filters_from_question:
        project_code = detected_filters_from_question["project_code"]
        chroma_filter_conditions.append({"project_code": project_code})
        logging.debug(f"  -> Filtro de código de projeto DETECTADO para ChromaDB (tentativa automática): '{project_code}'")

    if "doc_type" in detected_filters_from_question:
        doc_type = detected_filters_from_question["doc_type"]
        chroma_filter_conditions.append({"doc_type": doc_type})
        logging.debug(f"  -> Filtro de tipo de documento DETECTADO para ChromaDB (tentativa automática): '{doc_type}'")

    top_N_chunks_initial = 15 # Aumentado para pegar mais "sementes"
    
    search_kwargs = {"k": top_N_chunks_initial}
    if chroma_filter_conditions:
        if len(chroma_filter_conditions) == 1:
            final_chroma_filter_clause = chroma_filter_conditions[0]
            logging.debug(f"  -> Aplicando filtro único no ChromaDB via retriever (tentativa automática): {final_chroma_filter_clause}")
        else:
            final_chroma_filter_clause = {"$and": chroma_filter_conditions}
            logging.debug(f"  -> Aplicando múltiplos filtros (AND) no ChromaDB via retriever (tentativa automática): {final_chroma_filter_clause}")
        search_kwargs["filter"] = final_chroma_filter_clause
    else:
        logging.debug("  -> Nenhum filtro específico detectado. Realizando busca geral via retriever.")

    retriever = _vector_store.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs_raw = retriever.invoke(question)

    logging.debug(f"{len(retrieved_docs_raw)} chunks RECUPERADOS BRUTOS (antes da filtragem manual e expansão). Verificando metadados:\n")
    
    retrieved_docs_filtered_and_expanded = []
    seen_chunks_identifiers = set() # Para evitar duplicatas após a expansão (document_hash, chunk_index)

    # Passo 1: Filtragem inicial baseada no client_name (já existente e funcionando)
    initial_relevant_chunks = []
    for i, doc in enumerate(retrieved_docs_raw):
        retrieved_client_name = doc.metadata.get('client_name', 'N/A')
        # AQUI: A filtragem já acontece com o cliente detectado. Se expected_client_name for None, ele não filtra.
        if expected_client_name and retrieved_client_name != expected_client_name:
            logging.debug(f"  -> Chunk {i+1} REJEITADO pela filtragem manual (não corresponde ao cliente esperado '{expected_client_name}').")
        else:
            initial_relevant_chunks.append(doc)
            logging.debug(f"  -> Chunk {i+1} ACEITO pela filtragem manual (corresponde ou não há cliente esperado).")

    # Passo 2: Expansão de contexto para os chunks mais relevantes
    header_keywords = [
        "demandas não aderentes", "premissas do escopo técnico", "considerações finais",
        "escopo administrativo", "escopo de leads", "escopo de clientes",
        "escopo de atividades", "escopo de produtos e serviços", "escopo de oportunidades",
        "escopo de ordem de venda", "escopo de documentos", "escopo de analytics",
        "módulos adicionais", "personalização de integração", "personalizações de crm",
        "definição do escopo", "detalhamento do escopo", "roteiro do projeto", "visão geral do projeto",
        "tecnologias-chave confirmadas", "fase 1: preparação do ambiente", "fase 2: construção da base",
        "fase 3: orquestração langgraph", "fase 4: teste, refinamento e avaliação", "definição dos nós"
    ]

    for seed_doc in initial_relevant_chunks:
        doc_hash = seed_doc.metadata.get("document_hash")
        chunk_index = seed_doc.metadata.get("chunk_index")
        original_filename = seed_doc.metadata.get("original_filename")

        if doc_hash is None or chunk_index is None:
            logging.warning(f"Chunk sem 'chunk_index' ou 'document_hash' em {original_filename}. Não será expandido.")
            if (hash(seed_doc.page_content), hash(frozenset(seed_doc.metadata.items()))) not in seen_chunks_identifiers: # fallback com hash do conteúdo
                retrieved_docs_filtered_and_expanded.append(seed_doc)
                seen_chunks_identifiers.add((hash(seed_doc.page_content), hash(frozenset(seed_doc.metadata.items()))))
            continue
        
        is_header_chunk = any(kw in seed_doc.page_content.lower() for kw in header_keywords)
        
        if is_header_chunk:
            logging.debug(f"DEBUG: Chunk {original_filename}[{chunk_index}] identificado como potencial cabeçalho de seção. Iniciando expansão.")
            
            # --- FIX CRÍTICO AQUI ---
            where_clause_for_expansion = {"document_hash": doc_hash}
            # Adiciona o filtro de client_name APENAS se um client_name foi detectado E NÃO É UM PLACEHOLDER GENÉRICO
            if expected_client_name is not None and expected_client_name != "UNKNOWN CLIENT":
                where_clause_for_expansion["client_name"] = expected_client_name

            all_chunks_from_doc_results = _vector_store._collection.get(
                where=where_clause_for_expansion, # AGORA ESTÁ SEGURO
                include=['documents', 'metadatas']
            )
            # --- FIM DO FIX CRÍTICO ---
            
            doc_chunks_by_index = sorted([
                Document(page_content=all_chunks_from_doc_results['documents'][j], 
                         metadata=all_chunks_from_doc_results['metadatas'][j])
                for j in range(len(all_chunks_from_doc_results['ids']))
            ], key=lambda x: x.metadata['chunk_index'])

            start_index_for_expansion = max(0, chunk_index - CONTEXT_EXPANSION_WINDOW)
            end_index_for_expansion = min(len(doc_chunks_by_index), chunk_index + CONTEXT_EXPANSION_WINDOW + 1)
            
            for k in range(start_index_for_expansion, end_index_for_expansion):
                if k < len(doc_chunks_by_index): # Proteção extra para garantir que o índice é válido
                    chunk_to_add = doc_chunks_by_index[k]
                    chunk_identifier = (chunk_to_add.metadata.get('document_hash'), chunk_to_add.metadata.get('chunk_index'))
                    if chunk_identifier not in seen_chunks_identifiers:
                        retrieved_docs_filtered_and_expanded.append(chunk_to_add)
                        seen_chunks_identifiers.add(chunk_identifier)
                        logging.debug(f"    -> Adicionado chunk vizinho {chunk_to_add.metadata.get('original_filename')}[{chunk_to_add.metadata.get('chunk_index')}] via expansão.")
        else:
            # Se não é um chunk de cabeçalho, apenas o adiciona se não foi adicionado pela expansão
            chunk_identifier = (doc_hash, chunk_index)
            if chunk_identifier not in seen_chunks_identifiers:
                retrieved_docs_filtered_and_expanded.append(seed_doc)
                seen_chunks_identifiers.add(chunk_identifier)
                logging.debug(f"DEBUG: Chunk {original_filename}[{chunk_index}] adicionado diretamente (não é cabeçalho ou já está coberto pela expansão).")

    # Ordenar os chunks finais por document_hash e chunk_index para garantir fluxo lógico
    retrieved_docs_filtered_and_expanded.sort(key=lambda x: (x.metadata.get('document_hash', 0), x.metadata.get('chunk_index', 0)))

    logging.debug(f"DEBUG: {len(retrieved_docs_filtered_and_expanded)} chunks FINAIS ENVIADOS para o LLM após filtragem e expansão. Conteúdos para debug:")
    for i, doc in enumerate(retrieved_docs_filtered_and_expanded):
        logging.debug(f"--- Chunk Final {i+1} ---")
        logging.debug(f"  Filename: {doc.metadata.get('original_filename')} | Index: {doc.metadata.get('chunk_index')}")
        logging.debug(f"  Content: {doc.page_content[:500]}...") # Exibe até 500 caracteres
        logging.debug("------------------------")

    # Preenchendo source_docs_metadata para uso posterior na citação
    source_docs_metadata = []
    for doc in retrieved_docs_filtered_and_expanded:
        source_docs_metadata.append({
            "source": doc.metadata.get('source', 'N/A'),
            "document_id": doc.metadata.get('original_filename', 'N/A'),
            "client_name": doc.metadata.get('client_name', 'N/A'),
            "doc_type": doc.metadata.get('doc_type', 'N/A'),
            "project_code_crm": doc.metadata.get('project_code', 'N/A'),
            "page_content": doc.page_content # Inclui o conteúdo para a citação, se necessário.
        })

    logging.info(f"DEBUG: {len(retrieved_docs_filtered_and_expanded)} chunks FINAIS ENVIADOS para o LLM após filtragem e expansão.")
    return {"context": retrieved_docs_filtered_and_expanded, "source_docs": source_docs_metadata, "filters": detected_filters_from_question}


def generate_response_node(state: AgentState):
    """
    Nó para gerar uma resposta usando o LLM com base na pergunta e no contexto recuperado.
    Prompt ajustado para melhor síntese e extração precisa de informações.
    """
    logging.info(f"Executando generate_response_node para a pergunta: {state['question']}")
    question = state["question"]
    context_docs = state["context"]

    if not context_docs:
        return {"answer": "Não encontrei informações suficientes no contexto fornecido para responder a esta pergunta."}

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Você é um assistente de IA interno da TOTVS, focado em automatizar tarefas e gerar inteligência de negócio, "
             "respondendo perguntas sobre documentos de projeto e informações de CRM. "
             "Sua tarefa é **extrair e sintetizar informações relevantes para responder à pergunta**, utilizando **APENAS o contexto fornecido**."
             "**INSTRUÇÃO CRÍTICA:** Se a pergunta for sobre um tópico específico como 'Considerações Finais', 'Objetivo', 'Premissas', 'Restrições', 'Customizações', 'Personalizações ou **'Não Aderências'**, "
             "procure pela seção ou subtítulo correspondente no contexto e **liste os pontos principais ou itens enumerados de forma EXATA**. "
             "Responda de forma **concisa, clara e profissional**. "
             "**Evite qualquer repetição de frases ou informações.** "
             "Se a informação necessária para formar uma resposta completa e adequada **não estiver presente ou for insuficiente** no contexto fornecido, "
             "declare 'Não encontrei informações suficientes no contexto fornecido para responder a esta pergunta.' "
             "**Não invente informações.** "
             f"Contexto: {context_text}"
            ),
            ("human", f"{question}"),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"question": question, "context": context_text})
    logging.info("Resposta generada pelo LLM.")

    # Garante que qualquer \\n vindo do LLM seja um \n real no Python, e não \\n literal.
    cleaned_response = response.strip().replace('\\n', '\n')
    logging.debug(f"Resposta antes da verificação de repetição: '{cleaned_response}'")

    response_to_return = cleaned_response

    # Lógica de detecção e remoção de repetição (mantida como está)
    half_len = len(cleaned_response) // 2
    if len(cleaned_response) % 2 == 0 and cleaned_response[:half_len] == cleaned_response[half_len:]:
        response_to_return = cleaned_response[:half_len].strip()
        logging.debug("Repetição exata da resposta completa detectada e removida.")
    else:
        match = re.match(r"^(.*?)(?:\s*\1\s*)+$", cleaned_response, re.DOTALL)
        if match:
            response_to_return = match.group(1).strip()
            logging.debug(f"Repetição de padrão detectada e removida pela regex. Original: '{cleaned_response}', Processada: '{response_to_return}'")
        else:
            logging.debug("Nenhuma repetição detectada pela regex ou por repetição exata.")

    return {"answer": response_to_return}


def format_citation_node(state: AgentState):
    """
    Nó para formatar a resposta final, incorporando uma citação concisa e bem formatada da fonte.
    Prioriza o chunk mais relevante, mas oferece fallback.
    """
    logging.info("Executando format_citation_node para referências exatas.")
    answer = state["answer"]
    all_source_docs = state["source_docs"]

    # Se o LLM respondeu que não encontrou informações, não há citação
    if "Não encontrei informações suficientes" in answer:
        logging.info("Resposta indica falta de informação, sem citação.")
        return {"answer": answer}

    selected_doc_meta_for_citation = None
    
    extracted_numbers = re.findall(r'\d+', answer)
    relevant_keywords = []

    common_keywords = [
        "licenças", "ids", "contratadas", "personalizações", "customizações",
        "campos", "integração", "filtros", "cliente", "clientes",
        "ordens de venda", "erp protheus", "ipass", "endereços", "cadastro"
    ]
    for kw in common_keywords:
        if kw in answer.lower():
            relevant_keywords.append(kw)
    
    # Priorizar o chunk com base no número de caracteres da resposta que ele contém
    best_match_count = -1
    for doc_meta in all_source_docs:
        current_snippet = doc_meta.get('page_content', '').strip()
        match_count = sum(1 for word in re.findall(r'\b\w+\b', answer.lower()) if word in current_snippet.lower())
        
        if match_count > best_match_count:
            best_match_count = match_count
            selected_doc_meta_for_citation = doc_meta
            logging.debug(f"DEBUG: Melhor chunk para citação atualizado: {doc_meta.get('document_id')}")

    if selected_doc_meta_for_citation is None and all_source_docs:
        selected_doc_meta_for_citation = all_source_docs[0]
        logging.debug("Nenhuma citação 'exata' encontrada (com match forte), usando o chunk mais relevante como fallback.")
    elif selected_doc_meta_for_citation is None:
        logging.warning("Nenhuma citação encontrada (lista de chunks vazia ou irrelevante).")

    final_citation_entry = None
    if selected_doc_meta_for_citation:
        citation_info = (
            f"**Documento:** {selected_doc_meta_for_citation['document_id']}\n" 
            f"**Cliente:** {selected_doc_meta_for_citation.get('client_name', 'N/A')}\n" 
            f"**Tipo:** {selected_doc_meta_for_citation.get('doc_type', 'N/A')}"
        )
        final_citation_entry = citation_info

    final_answer_text = answer
    if final_citation_entry:
        final_answer_text = final_answer_text.replace('\\n', '\n') 
        final_answer_text += "\n\n---\n**Fontes Consultadas:**\n" + final_citation_entry

    return {"answer": final_answer_text}
