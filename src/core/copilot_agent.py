# src/core/copilot_agent.py
import os
import re
import sys
import pathlib
from typing import TypedDict, List, Optional, Any, Dict 
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
import httpx
from chromadb.config import Settings
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do .env
load_dotenv()

# --- Constantes de Configuração (Devem ser as mesmas do knowledge_base_builder.py) ---
# Caminho para o diretório do ChromaDB, ALINHADO com knowledge_base_builder.py
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vector_db', 'chroma_db'))
CHROMA_COLLECTION_NAME = 'tcrm_copilot_kb' # <--- DEFINIÇÃO DO NOME DA COLEÇÃO
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

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

# Inicializa o LLM
def get_llm():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada.")

    # Configura o cliente HTTP para ignorar a verificação SSL, essencial para o proxy da TOTVS
    custom_http_client = httpx.Client(verify=False)

    return ChatOpenAI(
        model="gpt-4o-mini", # Ou qualquer outro modelo GPT que você queira usar
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        temperature=0,       # Menor temperatura para respostas mais determinísticas
        http_client=custom_http_client
    )

# Instância global do LLM para uso pelos nós
llm = get_llm()

# Inicializa o modelo de Embeddings
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

# Instância global do modelo de embeddings
embeddings_model = get_embeddings_model()

# Carrega o ChromaDB persistido
_vector_store: Optional[Chroma] = None # Variável privada para armazenar a instância do vector store

def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        if not os.path.exists(CHROMA_DB_DIRECTORY):
            # Levanta um erro claro se o diretório do DB não existir
            raise FileNotFoundError(f"ChromaDB directory not found at {CHROMA_DB_DIRECTORY}. "
                                    "Please run src/data_ingestion/knowledge_base_builder.py first.")
        # Se o diretório existe, carrega o ChromaDB
        _vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY, 
            embedding_function=embeddings_model,
            collection_name=CHROMA_COLLECTION_NAME # <--- USANDO O NOME DA COLEÇÃO
        )
        print(f"DEBUG: ChromaDB carregado de {CHROMA_DB_DIRECTORY} com coleção '{CHROMA_COLLECTION_NAME}'.", file=sys.stderr)
    return _vector_store

# Tenta carregar o vector store na inicialização do módulo.
# Se falhar (DB ainda não construído), _vector_store permanecerá None,
# e o erro será tratado em retrieve_context_node.
try:
    _vector_store = get_vector_store()
except FileNotFoundError as e:
    print(f"WARNING: {e}. O Copilot pode não funcionar corretamente até que a base de conhecimento seja construída.", file=sys.stderr)
    _vector_store = None # Garante que a variável está None se a carga falhar


# --- FUNÇÃO DE DETECÇÃO DE CLIENTE/PROJETO/DOC_TYPE (REVISADA E NÃO HARDCODED) ---
def _detect_client_or_project_in_question(question: str) -> Dict[str, str]:
    """
    Detecta se um nome de cliente, código de projeto ou tipo de documento está presente na pergunta do usuário.
    Retorna um dicionário com os filtros detectados, alinhados aos metadados (agora todos em UPPERCASE).
    """
    detected_filters = {}
    question_lower = question.lower()

    # Define uma lista abrangente de palavras comuns em português para exclusão.
    # Esta lista *NÃO* contém nomes de clientes, mas elementos linguísticos comuns.
    # É crucial para evitar falsos positivos como "Quais", "O", "De", etc.
    COMMON_PORTUGUESE_EXCLUSIONS = {
        "A", "AS", "O", "OS", "E", "DE", "DO", "DA", "DOS", "DAS", "UM", "UMA", "UNS", "UMAS",
        "EM", "NO", "NA", "NOS", "NAS", "POR", "PELO", "PELA", "PELOS", "PELAS", "COM", "SEM", "PARA",
        "QUAL", "QUAIS", "QUEM", "QUE", "COMO", "ONDE", "QUANDO", "PORQUE", "PRA", "PRO", "PRAS", "PROS",
        "SOBRE", "FALE", "ME", "SOBRE", "EXPLIQUE", "DETALHE", "DISCUTA", "APRESENTE", "MOSTRAR", "VER",
        "PROJETO", "DOCUMENTO", "ARQUIVO", "INFORMAÇÕES", "DADOS", "ESCOPO", "CONSIDERAÇÕES", "FINAIS",
        "OBJETIVO", "HISTÓRICO", "VERSÃO", "AMBIENTAÇÃO", "SUMÁRIO", "DEFINIÇÃO", "MÓDULOS", "ADICIONAIS",
        "PREMISSAS", "NEGÓCIO", "SOLUÇÃO", "SISTEMA", "SOFTWARE", "APLICAÇÃO", "PLATAFORMA", "INTEGRAÇÃO",
        "API", "APIS", "CLIENTE", "ID", "IDS", "LICENÇAS", "QUANTAS", "CONTRATADAS", "MIT", # "MIT" é um tipo de documento, não um cliente
        "CRM", "TOTVS", "ERP", "DATASUL", "PROTHEUS", "BX", "POC", "COPILOT" # Outros termos comuns do domínio
    }

    # 1. Detecção de nome de cliente entre colchetes (prioridade máxima, alta confiança)
    # Ex: "[KION]", "[SCENS]"
    bracket_match = re.search(r'\[([A-Z0-9\s-]+)\]', question, re.IGNORECASE)
    if bracket_match:
        # Pega o conteúdo dentro dos colchetes, remove espaços extras e converte para UPPER
        # AGORA VAI CORRESPONDER DIRETAMENTE AOS METADADOS EM UPPERCASE.
        client_name_candidate = bracket_match.group(1).strip().upper() 
        detected_filters["client_name"] = client_name_candidate
        print(f"DEBUG: Cliente detectado por colchetes: '{detected_filters['client_name']}'")
        
    # 2. Se não encontrou colchetes, tenta detectar nomes capitalizados não excluídos
    # Este é um fallback mais heurístico, mas com uma lista de exclusão mais robusta.
    if "client_name" not in detected_filters:
        # Busca por palavras que começam com maiúscula (ou são todas maiúsculas),
        # permitindo múltiplos termos em nomes compostos (ex: "SOUTH AMERICA").
        potential_client_candidates = re.findall(r'\b[A-Z][A-Z0-9]*(?:[\s-][A-Z][A-Z0-9]*)*\b', question)
        
        for candidate in potential_client_candidates:
            candidate_upper = candidate.upper()
            
            # Filtra por tamanho mínimo para evitar siglas indesejadas,
            # mas permite "ID" ou "MIT" se necessário para outros filtros.
            if len(candidate_upper) < 3 and candidate_upper not in {"ID", "MIT"}:
                continue
            
            # Se o candidato não está na lista de exclusão E não se parece com um código de projeto/documento.
            if candidate_upper not in COMMON_PORTUGUESE_EXCLUSIONS:
                if not re.fullmatch(r"D\d{9,15}", candidate_upper) and not re.fullmatch(r"MIT\d{3}", candidate_upper):
                    # AGORA VAI CORRESPONDER DIRETAMENTE AOS METADADOS EM UPPERCASE.
                    detected_filters["client_name"] = candidate.strip().upper() 
                    print(f"DEBUG: Cliente detectado por capitalização e exclusão: '{detected_filters['client_name']}'")
                    break # Assume apenas um nome de cliente por pergunta para simplificar a PoC

    # --- Detecção de Código de Projeto (D + dígitos) ---
    project_code_match_d = re.search(r"D\d{9,15}", question, re.IGNORECASE)
    if project_code_match_d:
        detected_filters["project_code"] = project_code_match_d.group(0).upper()
        print(f"DEBUG: Código de Projeto detectado: '{detected_filters['project_code']}'")

    # --- Detecção de Tipo de Documento (MIT + 3 dígitos) ---
    doc_type_match = re.search(r"(MIT\d{3})", question, re.IGNORECASE)
    if doc_type_match:
        detected_filters["doc_type"] = doc_type_match.group(1).upper()
        print(f"DEBUG: Tipo de Documento detectado: '{detected_filters['doc_type']}'")

    return detected_filters


# --- 3. Definição dos Nós (Nodes) ---

def retrieve_context_node(state: AgentState):
    """
    Nó para recuperar chunks relevantes da base de conhecimento.
    Recebe a pergunta do usuário e quaisquer filtros ativos do estado do agente.
    """
    global _vector_store

    if _vector_store is None:
        try:
            _vector_store = get_vector_store()
        except FileNotFoundError as e:
            raise RuntimeError(f"ChromaDB não disponível para recuperação: {e}. "
                               "Certifique-se de que src/data_ingestion/knowledge_base_builder.py foi executado com sucesso.")

    print(f"DEBUG: Executando retrieve_context_node para a pergunta: {state['question']}")
    question = state["question"]
    
    # Detecta filtros da pergunta atual
    detected_filters_from_question = _detect_client_or_project_in_question(question)
    
    # Prepara os filtros para o ChromaDB usando a sintaxe de $and para múltiplos
    chroma_filter_conditions = []
    
    if "client_name" in detected_filters_from_question:
        client_name = detected_filters_from_question["client_name"]
        # O nome do cliente detectado (já em UPPER) é usado diretamente, pois os metadados também estão em UPPER.
        chroma_filter_conditions.append({"client_name": client_name}) 
        print(f"  -> Filtro de cliente detectado: '{client_name}'")
    
    if "project_code" in detected_filters_from_question:
        project_code = detected_filters_from_question["project_code"]
        chroma_filter_conditions.append({"project_code": project_code})
        print(f"  -> Filtro de código de projeto detectado: '{project_code}'")

    if "doc_type" in detected_filters_from_question:
        doc_type = detected_filters_from_question["doc_type"]
        chroma_filter_conditions.append({"doc_type": doc_type})
        print(f"  -> Filtro de tipo de documento detectado: '{doc_type}'")

    # =====================================================================
    # >>>>> USANDO as_retriever COM search_kwargs PARA FILTRAGEM <<<<<

    top_N_chunks = 10 
    
    search_kwargs = {"k": top_N_chunks}
    if chroma_filter_conditions:
        if len(chroma_filter_conditions) == 1:
            final_chroma_filter_clause = chroma_filter_conditions[0]
            print(f"  -> Aplicando filtro único no ChromaDB via retriever: {final_chroma_filter_clause}")
        else:
            final_chroma_filter_clause = {"$and": chroma_filter_conditions}
            print(f"  -> Aplicando múltiplos filtros (AND) no ChromaDB via retriever: {final_chroma_filter_clause}")
        search_kwargs["filter"] = final_chroma_filter_clause
    else:
        print("  -> Nenhum filtro específico detectado. Realizando busca geral via retriever.")

    # Obtém o retriever e faz a busca
    retriever = _vector_store.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs = retriever.invoke(question)
    
    # =====================================================================
    
    print(f"DEBUG: {len(retrieved_docs)} chunks recuperados. Conteúdo dos chunks:")
    if not retrieved_docs:
        print("DEBUG: Nenhuma informação relevante foi recuperada.")
    
    # === CORREÇÃO: Inicialização e preenchimento de source_docs_metadata ===
    source_docs_metadata = [] # <--- Variável inicializada aqui!
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Original Filename: {doc.metadata.get('original_filename', 'N/A')}")
        print(f"Client Name: {doc.metadata.get('client_name', 'N/A')}")
        print(f"Doc Type: {doc.metadata.get('doc_type', 'N/A')}") 
        print(f"Project Code: {doc.metadata.get('project_code', 'N/A')}")
        print(f"Page Content (primeiros 300 chars): {doc.page_content[:300]}...")
        print(f"--- Fim Chunk {i+1} ---")

        # Preenchendo source_docs_metadata para uso posterior na citação
        source_docs_metadata.append({
            "source": doc.metadata.get('source', 'N/A'),
            "document_id": doc.metadata.get('original_filename', 'N/A'),
            "client_name": doc.metadata.get('client_name', 'N/A'),
            "doc_type": doc.metadata.get('doc_type', 'N/A'),
            "project_code_crm": doc.metadata.get('project_code', 'N/A'),
            "page_content": doc.page_content 
        })
    # ====================================================================

    print(f"DEBUG: {len(retrieved_docs)} chunks recuperados.")
    return {"context": retrieved_docs, "source_docs": source_docs_metadata, "filters": detected_filters_from_question}


def generate_response_node(state: AgentState):
    """
    Nó para gerar uma resposta usando o LLM com base na pergunta e no contexto recuperado.
    Prompt ajustado para melhor síntese de informações.
    """
    print(f"DEBUG: Executando generate_response_node para a pergunta: {state['question']}")
    question = state["question"]
    context_docs = state["context"]

    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Você é um assistente de IA interno da TOTVS, focado em automatizar tarefas e gerar inteligência de negócio, "
             "respondendo perguntas sobre documentos de projeto e informações de CRM. "
             "Sua tarefa é **extrair e sintetizar informações relevantes para responder à pergunta**, utilizando **APENAS o contexto fornecido**."
             "Se a pergunta for sobre um tópico específico como 'Considerações Finais', 'Objetivo', 'Premissas', 'Restrições', etc., "
             "procure pela seção correspondente e forneça o conteúdo principal dela. "
             "Responda de forma **concisa, clara e profissional**. "
             "**Evite qualquer repetição de frases ou informações.** "
             "Se a informação necessária para formar uma resposta completa e adequada **não estiver presente ou for insuficiente** no contexto fornecido, "
             "declare 'Não encontrei informações suficientes no contexto fornecido para responder a esta pergunta.' "
             "**Não invente informações.** "
             "Contexto: {context}"
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"question": question, "context": context_text})
    print("DEBUG: Resposta gerada pelo LLM.")

    cleaned_response = response.strip()
    print(f"DEBUG: Resposta antes da verificação de repetição: '{cleaned_response}'")

    response_to_return = cleaned_response

    # Lógica de detecção e remoção de repetição (mantida como está)
    half_len = len(cleaned_response) // 2
    if len(cleaned_response) % 2 == 0 and cleaned_response[:half_len] == cleaned_response[half_len:]:
        response_to_return = cleaned_response[:half_len].strip()
        print("DEBUG: Repetição exata da resposta completa detectada e removida.")
    else:
        match = re.match(r"^(.*?)(?:\s*\1\s*)+$", cleaned_response, re.DOTALL)
        if match:
            response_to_return = match.group(1).strip()
            print(f"DEBUG: Repetição de padrão detectada e removida pela regex. Original: '{cleaned_response}', Processada: '{response_to_return}'")
        else:
            print("DEBUG: Nenhuma repetição detectada pela regex ou por repetição exata.")

    return {"answer": response_to_return}


def format_citation_node(state: AgentState):
    """
    Nó para formatar a resposta final, incorporando uma citação concisa e exata da fonte.
    Tenta encontrar o chunk que contém a informação da resposta ou o mais relevante como fallback.
    """
    print("DEBUG: Executando format_citation_node para referências exatas.")
    answer = state["answer"]
    all_source_docs = state["source_docs"]

    selected_doc_meta_for_citation = None
    snippet_to_cite = None

    extracted_numbers = re.findall(r'\d+', answer)
    
    relevant_keywords = []
    if "licenças" in answer.lower():
        relevant_keywords.append("licenças")
    if "ids" in answer.lower():
        relevant_keywords.append("ids")
    if "contratadas" in answer.lower():
        relevant_keywords.append("contratadas")
    
    for doc_meta in all_source_docs:
        current_snippet = doc_meta.get('page_content', '').strip()
        current_snippet_lower = current_snippet.lower()

        is_strong_match = False
        
        if relevant_keywords and extracted_numbers:
             if any(kw in current_snippet_lower for kw in relevant_keywords) and \
                any(num in current_snippet for num in extracted_numbers):
                 is_strong_match = True
        elif relevant_keywords and not extracted_numbers: # Se não achou número, mas achou keywords
            if any(kw in current_snippet_lower for kw in relevant_keywords):
                is_strong_match = True
        elif extracted_numbers and not relevant_keywords: # Se achou número, mas não keywords
            if any(num in current_snippet for num in extracted_numbers):
                is_strong_match = True
        
        if is_strong_match:
            selected_doc_meta_for_citation = doc_meta
            snippet_to_cite = current_snippet
            print(f"DEBUG: Encontrado chunk 'exato' para citar (match forte): {doc_meta['document_id']}")
            break

    if selected_doc_meta_for_citation is None and all_source_docs:
        selected_doc_meta_for_citation = all_source_docs[0]
        snippet_to_cite = selected_doc_meta_for_citation.get('page_content', '').strip()
        print("DEBUG: Nenhuma citação 'exata' encontrada, usando o chunk mais similar como fallback.")
    elif selected_doc_meta_for_citation is None:
        print("DEBUG: Nenhuma citação encontrada (lista de chunks vazia).")

    final_citation_entry = None
    if selected_doc_meta_for_citation:
        citation_parts = [
            f"**Documento:** {selected_doc_meta_for_citation['document_id']} "
            f"(Cliente: {selected_doc_meta_for_citation.get('client_name', 'N/A')}, "
            f"Tipo: {selected_doc_meta_for_citation.get('doc_type', 'N/A')}, "
            f"Código Projeto CRM: {selected_doc_meta_for_citation.get('project_code_crm', 'N/A')})"
        ]
        if snippet_to_cite:
            if len(snippet_to_cite) > 300:
                snippet_to_cite = snippet_to_cite[:300] + "..."
            citation_parts.append(f"'{snippet_to_cite}'")
        final_citation_entry = "\n".join(citation_parts)

    final_answer_text = answer
    if final_citation_entry:
        final_answer_text += "\n\n**Fontes Consultadas:**\n\n" + final_citation_entry

    return {"answer": final_answer_text}
