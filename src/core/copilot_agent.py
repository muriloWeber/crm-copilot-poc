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


# --- FUNÇÃO DE DETECÇÃO DE CLIENTE/PROJETO/DOC_TYPE OTIMIZADA E COM FALLBACK ---
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
    if _known_client_names: # Certifica-se de que a lista foi carregada
        for client_name in _known_client_names:
            client_name_lower = client_name.lower()
            
            # Tenta detecção com word boundaries (precisa)
            search_pattern_regex = r'\\b' + re.escape(client_name_lower) + r'\\b'
            print(f"DEBUG (Detector): Tentando padrão regex '{search_pattern_regex}' para cliente '{client_name}'")
            if re.search(search_pattern_regex, question_lower):
                detected_filters["client_name"] = client_name # Já está em UPPERCASE
                print(f"DEBUG (Detector): Cliente detectado por regex e lista de nomes conhecidos: '{detected_filters['client_name']}'")
                break # Encontrou um cliente, não precisa procurar mais
            else:
                # Fallback: Se a regex com word boundary falhou, tenta uma busca simples 'in'
                if client_name_lower in question_lower:
                    detected_filters["client_name"] = client_name # Já está em UPPERCASE
                    print(f"DEBUG (Detector): Cliente detectado por busca direta 'in' (fallback robusto): '{detected_filters['client_name']}'")
                    break # Encontrou um cliente, não precisa procurar mais
                else:
                    print(f"DEBUG (Detector): Cliente '{client_name}' não encontrado na questão (busca direta ou regex).")
    else:
        print("DEBUG (Detector): _known_client_names não carregada ou vazia. Não foi possível usar a detecção dinâmica de clientes.")

    # 2. Detecção de Código de Projeto (D + dígitos)
    project_code_match_d = re.search(r"D\\d{9,15}", question, re.IGNORECASE)
    if project_code_match_d:
        detected_filters["project_code"] = project_code_match_d.group(0).upper()
        print(f"DEBUG (Detector): Código de Projeto detectado: '{detected_filters['project_code']}'")

    # 3. Detecção de Tipo de Documento (MIT + 3 dígitos)
    doc_type_match = re.search(r"(MIT\\d{3})", question, re.IGNORECASE)
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
    Adicionado log detalhado dos chunks recuperados para depuração e filtragem manual.
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
    print(f"DEBUG: Filtros detectados da pergunta: {detected_filters_from_question}")
    
    # Prepara os filtros para o ChromaDB usando a sintaxe de $and para múltiplos
    chroma_filter_conditions = []
    
    # Captura o client_name detectado para uso na filtragem manual
    expected_client_name = None
    if "client_name" in detected_filters_from_question:
        expected_client_name = detected_filters_from_question["client_name"]
        chroma_filter_conditions.append({"client_name": expected_client_name}) 
        print(f"  -> Filtro de cliente DETECTADO para ChromaDB (tentativa automática): '{expected_client_name}'")
    
    if "project_code" in detected_filters_from_question:
        project_code = detected_filters_from_question["project_code"]
        chroma_filter_conditions.append({"project_code": project_code})
        print(f"  -> Filtro de código de projeto DETECTADO para ChromaDB (tentativa automática): '{project_code}'")

    if "doc_type" in detected_filters_from_question:
        doc_type = detected_filters_from_question["doc_type"]
        chroma_filter_conditions.append({"doc_type": doc_type})
        print(f"  -> Filtro de tipo de documento DETECTADO para ChromaDB (tentativa automática): '{doc_type}'")

    # =====================================================================
    # >>>>> USANDO as_retriever. Passando o filtro, mesmo sabendo que pode não ser totalmente eficaz <<<<<

    top_N_chunks = 10 
    
    search_kwargs = {"k": top_N_chunks}
    if chroma_filter_conditions:
        if len(chroma_filter_conditions) == 1:
            final_chroma_filter_clause = chroma_filter_conditions[0]
            print(f"  -> Aplicando filtro único no ChromaDB via retriever (tentativa automática): {final_chroma_filter_clause}")
        else:
            final_chroma_filter_clause = {"$and": chroma_filter_conditions}
            print(f"  -> Aplicando múltiplos filtros (AND) no ChromaDB via retriever (tentativa automática): {final_chroma_filter_clause}")
        search_kwargs["filter"] = final_chroma_filter_clause
    else:
        print("  -> Nenhum filtro específico detectado. Realizando busca geral via retriever.")

    # Obtém o retriever e faz a busca
    retriever = _vector_store.as_retriever(search_kwargs=search_kwargs)
    retrieved_docs_raw = retriever.invoke(question) # Renomeado para indicar que são os chunks brutos

    # =====================================================================
    
    # --- LOG DOS CHUNKS RECUPERADOS BRUTOS (antes da filtragem manual) ---
    print(f"DEBUG: {len(retrieved_docs_raw)} chunks RECUPERADOS BRUTOS (antes da filtragem manual). Verificando metadados:\\n")
    
    # --- FILTRAGEM MANUAL DE EMERGÊNCIA ---
    retrieved_docs_filtered = []
    
    # Itera sobre os chunks brutos e aplica a filtragem manual
    for i, doc in enumerate(retrieved_docs_raw):
        retrieved_client_name = doc.metadata.get('client_name', 'N/A')
        retrieved_original_filename = doc.metadata.get('original_filename', 'N/A')
        retrieved_doc_type = doc.metadata.get('doc_type', 'N/A')

        print(f"--- Chunk {i+1} (Bruto) ---")
        print(f"  Client Name: {retrieved_client_name}")
        print(f"  Doc Type: {retrieved_doc_type}")
        print(f"  Original Filename: {retrieved_original_filename}")
        print(f"  Page Content (primeiros 150 chars): {doc.page_content[:150]}...")
        
        # Lógica de filtragem: Se um client_name foi detectado na pergunta E o chunk não corresponde, REJEITAR.
        if expected_client_name and retrieved_client_name != expected_client_name:
            print(f"  -> Chunk REJEITADO pela filtragem manual (não corresponde ao cliente esperado '{expected_client_name}').")
        else:
            retrieved_docs_filtered.append(doc)
            print(f"  -> Chunk ACEITO pela filtragem manual (corresponde ou não há cliente esperado).")
        print(f"--- Fim Chunk {i+1} (Bruto) ---\\n")
    
    # Atribui os chunks filtrados para o processamento subsequente
    retrieved_docs = retrieved_docs_filtered
    
    if not retrieved_docs:
        print(f"WARNING: Após a filtragem manual, nenhum chunk restou para o cliente '{expected_client_name}' (se aplicável). "
              f"O LLM pode indicar que não encontrou informações suficientes.\\n")
        
    # Preenchendo source_docs_metadata para uso posterior na citação
    source_docs_metadata = []
    for doc in retrieved_docs:
        source_docs_metadata.append({
            "source": doc.metadata.get('source', 'N/A'),
            "document_id": doc.metadata.get('original_filename', 'N/A'),
            "client_name": doc.metadata.get('client_name', 'N/A'),
            "doc_type": doc.metadata.get('doc_type', 'N/A'),
            "project_code_crm": doc.metadata.get('project_code', 'N/A'),
            "page_content": doc.page_content 
        })
    # --- FIM LOG DETALHADO ---

    print(f"DEBUG: {len(retrieved_docs)} chunks FINAIS ENVIADOS para o LLM após filtragem manual.\\n")
    return {"context": retrieved_docs, "source_docs": source_docs_metadata, "filters": detected_filters_from_question}


def generate_response_node(state: AgentState):
    """
    Nó para gerar uma resposta usando o LLM com base na pergunta e no contexto recuperado.
    Prompt ajustado para melhor síntese de informações.
    """
    print(f"DEBUG: Executando generate_response_node para a pergunta: {state['question']}")
    question = state["question"]
    context_docs = state["context"]

    if not context_docs:
        return {"answer": "Não encontrei informações suficientes no contexto fornecido para responder a esta pergunta."}

    context_text = "\\n\\n".join([doc.page_content for doc in context_docs])

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
             f"Contexto: {context_text}"
            ),
            ("human", f"{question}"),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"question": question, "context": context_text})
    print("DEBUG: Resposta gerada pelo LLM.")

    # Garante que qualquer \\n vindo do LLM seja um \n real no Python, e não \\n literal.
    cleaned_response = response.strip().replace('\\\n', '\n')
    print(f"DEBUG: Resposta antes da verificação de repetição: '{cleaned_response}'")

    response_to_return = cleaned_response

    # Lógica de detecção e remoção de repetição (mantida como está)
    half_len = len(cleaned_response) // 2
    if len(cleaned_response) % 2 == 0 and cleaned_response[:half_len] == cleaned_response[half_len:]:
        response_to_return = cleaned_response[:half_len].strip()
        print("DEBUG: Repetição exata da resposta completa detectada e removida.")
    else:
        match = re.match(r"^(.*?)(?:\\s*\\1\\s*)+$", cleaned_response, re.DOTALL)
        if match:
            response_to_return = match.group(1).strip()
            print(f"DEBUG: Repetição de padrão detectada e removida pela regex. Original: '{cleaned_response}', Processada: '{response_to_return}'")
        else:
            print("DEBUG: Nenhuma repetição detectada pela regex ou por repetição exata.")

    return {"answer": response_to_return}


def format_citation_node(state: AgentState):
    """
    Nó para formatar a resposta final, incorporando uma citação concisa e bem formatada da fonte.
    Prioriza o chunk mais relevante, mas oferece fallback.
    """
    print("DEBUG: Executando format_citation_node para referências exatas.")
    answer = state["answer"]
    all_source_docs = state["source_docs"]

    # Se o LLM respondeu que não encontrou informações, não há citação
    if "Não encontrei informações suficientes" in answer:
        print("DEBUG: Resposta indica falta de informação, sem citação.")
        return {"answer": answer}

    selected_doc_meta_for_citation = None
    
    extracted_numbers = re.findall(r'\\d+', answer)
    relevant_keywords = []

    common_keywords = [
        "licenças", "ids", "contratadas", "personalizações", "customizações",
        "campos", "integração", "filtros", "cliente", "clientes",
        "ordens de venda", "erp protheus", "ipass", "endereços", "cadastro"
    ]
    for kw in common_keywords:
        if kw in answer.lower():
            relevant_keywords.append(kw)
    
    for doc_meta in all_source_docs:
        current_snippet = doc_meta.get('page_content', '').strip()
        current_snippet_lower = current_snippet.lower()

        is_strong_match = False
        
        # Lógica de correspondência de citação aprimorada
        # 1. Busca por tokens-chave da resposta no snippet (ignorando stopwords e tokens curtos)
        answer_tokens = [token for token in re.findall(r'\\b\\w+\\b', answer.lower()) if token not in {'de', 'da', 'do', 'e', 'o', 'a', 'os', 'as', 'em', 'um', 'uma'} and len(token) > 2]
        snippet_tokens = [token for token in re.findall(r'\\b\\w+\\b', current_snippet_lower) if token not in {'de', 'da', 'do', 'e', 'o', 'a', 'os', 'as', 'em', 'um', 'uma'} and len(token) > 2]

        if all(token in snippet_tokens for token in answer_tokens):
            is_strong_match = True

        if not is_strong_match: # Fallback se o match de tokens não foi suficiente
             if (relevant_keywords and any(kw in current_snippet_lower for kw in relevant_keywords)) and \
                (extracted_numbers and any(num in current_snippet for num in extracted_numbers)):
                 is_strong_match = True
             elif relevant_keywords and not extracted_numbers: 
                 if any(kw in current_snippet_lower for kw in relevant_keywords):
                     is_strong_match = True
             elif extracted_numbers and not relevant_keywords: 
                 if any(num in current_snippet for num in extracted_numbers):
                     is_strong_match = True
        
        if is_strong_match:
            selected_doc_meta_for_citation = doc_meta
            print(f"DEBUG: Encontrado chunk 'exato' para citar (match forte): {doc_meta['document_id']}")
            break

    if selected_doc_meta_for_citation is None and all_source_docs:
        # Fallback para o primeiro chunk se nenhum match forte for encontrado
        selected_doc_meta_for_citation = all_source_docs[0]
        print("DEBUG: Nenhuma citação 'exata' encontrada, usando o chunk mais similar como fallback.")
    elif selected_doc_meta_for_citation is None:
        print("DEBUG: Nenhuma citação encontrada (lista de chunks vazia).")

    final_citation_entry = None
    if selected_doc_meta_for_citation:
        # Usando \n para cada quebra de linha desejada na citação.
        # Isto é o que o Streamlit espera para Markdown.
        citation_info = (
            f"**Documento:** {selected_doc_meta_for_citation['document_id']}\n" 
            f"**Cliente:** {selected_doc_meta_for_citation.get('client_name', 'N/A')}\n" 
            f"**Tipo:** {selected_doc_meta_for_citation.get('doc_type', 'N/A')}"
        )
        final_citation_entry = citation_info

    final_answer_text = answer
    if final_citation_entry:
        # Garante que a resposta principal também esteja livre de \\n literal.
        final_answer_text = final_answer_text.replace('\\n', '\n') 
        # Usa \n\n para criar um espaçamento de parágrafo antes das Fontes Consultadas.
        final_answer_text += "\n\n---\n**Fontes Consultadas:**\n" + final_citation_entry

    return {"answer": final_answer_text}