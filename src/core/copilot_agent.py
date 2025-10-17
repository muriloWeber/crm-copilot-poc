# src/core/copilot_agent.py
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

# Carrega as variáveis de ambiente do .env
load_dotenv()

# --- Constantes de Configuração (Devem ser as mesmas do knowledge_base_builder.py) ---
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vector_db', 'chroma_db'))
CHROMA_COLLECTION_NAME = 'tcrm_copilot_kb'
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
                                    "Please run src/data_ingestion/knowledge_base_builder.py first.")
        _vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY, 
            embedding_function=embeddings_model,
            collection_name=CHROMA_COLLECTION_NAME
        )
        print(f"DEBUG: ChromaDB carregado de {CHROMA_DB_DIRECTORY} com coleção '{CHROMA_COLLECTION_NAME}'.", file=sys.stderr)
    return _vector_store

try:
    _vector_store = get_vector_store()
except FileNotFoundError as e:
    print(f"WARNING: {e}. O Copilot pode não funcionar corretamente até que a base de conhecimento seja construída.", file=sys.stderr)
    _vector_store = None

# --- FUNÇÃO PARA CARREGAR NOMES DE CLIENTES DINAMICAMENTE DO CHROMADB ---
_known_client_names: Set[str] = set()

def _load_known_client_names_from_chroma():
    global _known_client_names
    if _vector_store:
        print("DEBUG: Carregando nomes de clientes conhecidos do ChromaDB...")
        try:
            # Acessa o cliente ChromaDB subjacente para maior controle
            chroma_client = _vector_store._client
            collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
            
            # Obtém a contagem de documentos na coleção
            count = collection.count()
            
            if count == 0:
                print("DEBUG: Coleção ChromaDB vazia. Nenhum cliente para carregar.")
                _known_client_names = set()
                return

            # Obtém todos os metadados da coleção
            # Usamos limit=count para buscar todos os documentos existentes.
            # O `where` é omitido, mas a chamada `get` do cliente direto é mais tolerante.
            results = collection.get(
                limit=count, # Tenta pegar todos os documentos
                include=['metadatas']
            )
            
            unique_clients = set()
            for metadata_dict in results.get('metadatas', []):
                client_name = metadata_dict.get('client_name')
                # Filtra valores None, strings vazias e 'Unknown Client' em Python.
                if client_name and client_name.strip() and client_name.upper() != "UNKNOWN CLIENT":
                    unique_clients.add(client_name.upper()) # Garante que sejam todos em UPPERCASE
            _known_client_names = unique_clients
            print(f"DEBUG: Nomes de clientes carregados: {_known_client_names}")
        except Exception as e:
            print(f"ERROR: Falha ao carregar nomes de clientes do ChromaDB: {e}. "
                  f"Verifique a integridade da sua base ChromaDB ou os metadados.", file=sys.stderr)
            _known_client_names = set() # Define como vazio em caso de erro
    else:
        print("DEBUG: ChromaDB não carregado. Não foi possível carregar nomes de clientes.")

# Carrega os nomes dos clientes na inicialização do módulo, se o vector store estiver disponível
if _vector_store:
    _load_known_client_names_from_chroma()


# --- FUNÇÃO DE DETECÇÃO DE CLIENTE/PROJETO/DOC_TYPE (AGORA USA A LISTA DINÂMICA) ---
def _detect_client_or_project_in_question(question: str) -> Dict[str, str]:
    """
    Detecta se um nome de cliente, código de projeto ou tipo de documento está presente na pergunta do usuário.
    Prioriza a busca por nomes de clientes conhecidos carregados dinamicamente do ChromaDB.
    Retorna um dicionário com os filtros detectados, alinhados aos metadados (todos em UPPERCASE).
    """
    detected_filters = {}
    question_lower = question.lower()
    print(f"DEBUG (Detector): Questão recebida: '{question}'")

    # 1. Detecção de Nome de Cliente (usando a lista dinâmica de _known_client_names)
    # Esta é a forma mais robusta e dinâmica.
    if _known_client_names: # Certifica-se de que a lista foi carregada
        for client_name in _known_client_names:
            # Usar word boundaries (\b) para evitar falsos positivos (ex: "SCENS" em "ASCENSO").
            # re.escape() para tratar caracteres especiais em nomes de clientes, se houver.
            if re.search(r'\b' + re.escape(client_name.lower()) + r'\b', question_lower):
                detected_filters["client_name"] = client_name # Já está em UPPERCASE
                print(f"DEBUG (Detector): Cliente detectado por lista de nomes conhecidos: '{detected_filters['client_name']}'")
                break # Encontrou um cliente, não precisa procurar mais
            else:
                print(f"DEBUG (Detector): Cliente '{client_name}' não encontrado na questão (busca direta).")
    else:
        print("DEBUG (Detector): _known_client_names não carregada ou vazia. Não foi possível usar a detecção dinâmica de clientes.")

    # 2. Detecção de Código de Projeto (D + dígitos)
    project_code_match_d = re.search(r"D\d{9,15}", question, re.IGNORECASE)
    if project_code_match_d:
        detected_filters["project_code"] = project_code_match_d.group(0).upper()
        print(f"DEBUG (Detector): Código de Projeto detectado: '{detected_filters['project_code']}'")

    # 3. Detecção de Tipo de Documento (MIT + 3 dígitos)
    doc_type_match = re.search(r"(MIT\d{3})", question, re.IGNORECASE)
    if doc_type_match:
        detected_filters["doc_type"] = doc_type_match.group(1).upper()
        print(f"DEBUG (Detector): Tipo de Documento detectado: '{detected_filters['doc_type']}'")

    print(f"DEBUG (Detector): Filtros finais detectados: {detected_filters}")
    return detected_filters


# --- 3. Definição dos Nós (Nodes) ---

def retrieve_context_node(state: AgentState):
    """
    Nó para recuperar chunks relevantes da base de conhecimento.
    Recebe a pergunta do usuário e quaisquer filtros ativos do estado do agente.
    Adicionado log detalhado dos chunks recuperados para depuração.
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
    print(f"DEBUG: Filtros detectados da pergunta: {detected_filters_from_question}") # Log adicional aqui
    
    # Prepara os filtros para o ChromaDB usando a sintaxe de $and para múltiplos
    chroma_filter_conditions = []
    
    if "client_name" in detected_filters_from_question:
        client_name = detected_filters_from_question["client_name"]
        chroma_filter_conditions.append({"client_name": client_name}) 
        print(f"  -> Filtro de cliente DETECTADO para ChromaDB: '{client_name}'") # Log adicional aqui
    
    if "project_code" in detected_filters_from_question:
        project_code = detected_filters_from_question["project_code"]
        chroma_filter_conditions.append({"project_code": project_code})
        print(f"  -> Filtro de código de projeto DETECTADO para ChromaDB: '{project_code}'") # Log adicional aqui

    if "doc_type" in detected_filters_from_question:
        doc_type = detected_filters_from_question["doc_type"]
        chroma_filter_conditions.append({"doc_type": doc_type})
        print(f"  -> Filtro de tipo de documento DETECTADO para ChromaDB: '{doc_type}'") # Log adicional aqui

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
    
    # --- LOG DETALHADO PARA DIAGNÓSTICO DO FILTRO ---
    print(f"DEBUG: {len(retrieved_docs)} chunks RECUPERADOS APÓS APLICAÇÃO DE FILTRO. Verificando metadados:\n")
    if not retrieved_docs:
        print("DEBUG: Nenhuma informação relevante foi recuperada após filtragem.\n")
    
    source_docs_metadata = [] # Variável inicializada aqui!
    for i, doc in enumerate(retrieved_docs):
        retrieved_client_name = doc.metadata.get('client_name', 'N/A')
        retrieved_original_filename = doc.metadata.get('original_filename', 'N/A')
        retrieved_doc_type = doc.metadata.get('doc_type', 'N/A') # Adicionar tipo de documento

        print(f"--- Chunk {i+1} ---")
        print(f"  Client Name: {retrieved_client_name}")
        print(f"  Doc Type: {retrieved_doc_type}")
        print(f"  Original Filename: {retrieved_original_filename}")
        print(f"  Page Content (primeiros 150 chars): {doc.page_content[:150]}...")
        
        # Alerta se um chunk não corresponder ao cliente esperado
        expected_client = detected_filters_from_question.get('client_name')
        if expected_client and retrieved_client_name != expected_client:
            print(f"  !!! ALERTA: chunk {i+1} é do cliente '{retrieved_client_name}' e NÃO CORRESPONDE ao cliente esperado '{expected_client}' !!!")
        
        print(f"--- Fim Chunk {i+1} ---\n") # Adicionado \n para melhor legibilidade no console

        # Preenchendo source_docs_metadata para uso posterior na citação
        source_docs_metadata.append({
            "source": doc.metadata.get('source', 'N/A'),
            "document_id": retrieved_original_filename,
            "client_name": retrieved_client_name,
            "doc_type": retrieved_doc_type,
            "project_code_crm": doc.metadata.get('project_code', 'N/A'),
            "page_content": doc.page_content 
        })
    # --- FIM LOG DETALHADO ---

    print(f"DEBUG: {len(retrieved_docs)} chunks recuperados para o LLM.\n") # Adicionado \n para melhor legibilidade no console
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
    # Adicionado mais palavras-chave para melhor detecção da citação
    if "licenças" in answer.lower():
        relevant_keywords.append("licenças")
    if "ids" in answer.lower():
        relevant_keywords.append("ids")
    if "contratadas" in answer.lower():
        relevant_keywords.append("contratadas")
    if "personalizações" in answer.lower():
        relevant_keywords.append("personalizações")
    if "campos" in answer.lower():
        relevant_keywords.append("campos")
    
    for doc_meta in all_source_docs:
        current_snippet = doc_meta.get('page_content', '').strip()
        current_snippet_lower = current_snippet.lower()

        is_strong_match = False
        
        # Lógica de correspondência de citação aprimorada
        if relevant_keywords and extracted_numbers:
             if any(kw in current_snippet_lower for kw in relevant_keywords) and \
                any(num in current_snippet for num in extracted_numbers):
                 is_strong_match = True
        elif relevant_keywords and not extracted_numbers:
            if any(kw in current_snippet_lower for kw in relevant_keywords):
                is_strong_match = True
        elif extracted_numbers and not relevant_keywords:
            if any(num in current_snippet for num in extracted_numbers):
                is_strong_match = True
        
        # Também verifica se a resposta gerada está contida no snippet (match mais direto)
        # Atenção: LLMs podem parafrasear, então o match exato pode falhar.
        # Mas para snippets curtos de fatos, é útil.
        if answer.lower() in current_snippet_lower:
            is_strong_match = True

        if is_strong_match:
            selected_doc_meta_for_citation = doc_meta
            snippet_to_cite = current_snippet
            print(f"DEBUG: Encontrado chunk 'exato' para citar (match forte): {doc_meta['document_id']}")
            break

    if selected_doc_meta_for_citation is None and all_source_docs:
        # Fallback para o primeiro chunk se nenhum match forte for encontrado
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
            # Garante que o snippet exibido não seja excessivamente longo
            if len(snippet_to_cite) > 500: # Aumentado o limite para capturar mais contexto
                snippet_to_cite = snippet_to_cite[:500] + "..."
            citation_parts.append(f"'{snippet_to_cite}'")
        final_citation_entry = "\n".join(citation_parts)

    final_answer_text = answer
    if final_citation_entry:
        final_answer_text += "\n\n**Fontes Consultadas:**\n\n" + final_citation_entry

    return {"answer": final_answer_text}
