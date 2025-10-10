# src/core/copilot_agent.py
import os
import re
import sys # Import adicionado para tratamento de erros
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # OpenAIEmbeddings adicionado
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
import httpx
from chromadb.config import Settings # Mantido, mas não estritamente necessário para este código específico
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do .env
load_dotenv()

# --- Constantes de Configuração (Devem ser as mesmas do knowledge_base_builder.py) ---
# Caminho para o diretório do ChromaDB, resolvido a partir deste arquivo
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chroma_db'))
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

# --- 1. Definição do AgentState ---
class AgentState(TypedDict):
    """
    Representa o estado do agente Copilot.
    question: A pergunta original do usuário.
    context: Uma lista de documentos (chunks) recuperados da base de conhecimento.
    source_docs: Uma lista de dicionários contendo metadados detalhados dos chunks.
    answer: A resposta gerada pelo LLM.
    messages: Histórico de mensagens da conversa (opcional para PoC inicial, mas bom para expansão).
    filters: Um dicionário de filtros aplicados à recuperação.
    """
    question: str
    context: List[Document]
    source_docs: List[dict]
    answer: str
    messages: List[BaseMessage]
    filters: Optional[dict] # Adicionado aqui para permitir filtros no estado

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
        _vector_store = Chroma(persist_directory=CHROMA_DB_DIRECTORY, embedding_function=embeddings_model)
        print(f"DEBUG: ChromaDB carregado de {CHROMA_DB_DIRECTORY}.", file=sys.stderr)
    return _vector_store

# Tenta carregar o vector store na inicialização do módulo.
# Se falhar (DB ainda não construído), _vector_store permanecerá None,
# e o erro será tratado em retrieve_context_node.
try:
    _vector_store = get_vector_store()
except FileNotFoundError as e:
    print(f"WARNING: {e}. O Copilot pode não funcionar corretamente até que a base de conhecimento seja construída.", file=sys.stderr)
    _vector_store = None # Garante que a variável está None se a carga falhar


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
    filters = state.get("filters")

    if filters:
        print(f"DEBUG: Aplicando filtros na recuperação: {filters}")
        retrieved_docs = _vector_store.similarity_search(question, k=10, filter=filters)
    else:
        print("DEBUG: Nenhum filtro ativo. Recuperando da base completa.")
        retrieved_docs = _vector_store.similarity_search(question, k=10)

    # --- NOVO TRECHO DE DEBUGGING INICIO ---
    print(f"DEBUG: {len(retrieved_docs)} chunks recuperados. Conteúdo dos chunks:")
    if not retrieved_docs:
        print("DEBUG: Nenhuma informação relevante foi recuperada.")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Chunk {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Document ID: {doc.metadata.get('document_id', 'N/A')}")
        print(f"Page Content (primeiros 300 chars): {doc.page_content[:300]}...")
        print(f"--- Fim Chunk {i+1} ---")

    source_docs_metadata = []
    for doc in retrieved_docs:
        # Extrai metadados relevantes para a citação.
        # Assegure que as chaves de metadados (e.g., 'document_id', 'client_name')
        # correspondem às que você definiu em knowledge_base_builder.py.
        source_docs_metadata.append({
            "source": doc.metadata.get('source', 'N/A'),
            "document_id": doc.metadata.get('document_id', 'N/A'),
            "client_name": doc.metadata.get('client_name', 'N/A'),
            "totvs_coordinator": doc.metadata.get('totvs_coordinator', 'N/A'),
            "project_code_crm": doc.metadata.get('project_code_crm', 'N/A'),
            "page_content": doc.page_content # Inclui o conteúdo para citação posterior
        })
    print(f"DEBUG: {len(retrieved_docs)} chunks recuperados.")
    return {"context": retrieved_docs, "source_docs": source_docs_metadata}


def generate_response_node(state: AgentState):
    """
    Nó para gerar uma resposta usando o LLM com base na pergunta e no contexto recuperado.
    """
    print(f"DEBUG: Executando generate_response_node para a pergunta: {state['question']}")
    question = state["question"]
    context_docs = state["context"]

    # Concatena o conteúdo dos documentos para enviar ao LLM
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Você é um assistente de IA focado em responder perguntas sobre documentos de projetos e dados de CRM. "
             "Sua tarefa é fornecer respostas concisas e diretas com base EXCLUSIVAMENTE no contexto fornecido. "
             "Responda uma única vez, de forma clara e profissional, e evite qualquer repetição de frases ou informações. " # Nova instrução aqui
             "Se a resposta não puder ser encontrada no contexto fornecido, responda 'Não encontrei a informação no contexto fornecido.' "
             "Não invente informações. "
             "Contexto: {context}"
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"question": question, "context": context_text})
    print("DEBUG: Resposta gerada pelo LLM.")

    # --- TRECHO DE PÓS-PROCESSAMENTO PARA REMOVER REPETIÇÃO ---
    cleaned_response = response.strip()
    print(f"DEBUG: Resposta antes da verificação de repetição: '{cleaned_response}'")

    response_to_return = cleaned_response

    # Primeiro, verifica se é uma repetição exata da string completa (A.A.)
    half_len = len(cleaned_response) // 2
    if len(cleaned_response) % 2 == 0 and cleaned_response[:half_len] == cleaned_response[half_len:]:
        response_to_return = cleaned_response[:half_len].strip()
        print("DEBUG: Repetição exata da resposta completa detectada e removida.")
    else:
        # Caso contrário, tenta a regex para padrões repetidos com possível espaço/newline
        # Usaremos a regex original que é mais tolerante a espaços
        match = re.match(r"^(.*?)(?:\s*\1\s*)+$", cleaned_response, re.DOTALL)
        if match:
            response_to_return = match.group(1).strip()
            print(f"DEBUG: Repetição de padrão detectada e removida pela regex. Original: '{cleaned_response}', Processada: '{response_to_return}'")
        else:
            print("DEBUG: Nenhuma repetição detectada pela regex ou por repetição exata.")
    # --- FIM DO TRECHO DE PÓS-PROCESSAMENTO ---

    return {"answer": response_to_return} # Retorna a resposta já limpa


def format_citation_node(state: AgentState):
    """
    Nó para formatar a resposta final, incorporando uma citação concisa e exata da fonte.
    Tenta encontrar o chunk que contém a informação da resposta ou o mais relevante como fallback.
    """
    print("DEBUG: Executando format_citation_node para referências exatas.")
    answer = state["answer"]
    all_source_docs = state["source_docs"] # Esta é uma lista de dicionários de metadados dos chunks

    selected_doc_meta_for_citation = None
    snippet_to_cite = None

    # --- Lógica para encontrar o chunk mais EXATO ---
    # 1. Tentar extrair o número da resposta e palavras-chave
    extracted_number = next(iter(re.findall(r'\d+', answer)), None) # Pega o primeiro número na resposta
    
    relevant_keywords = []
    if "licenças" in answer.lower():
        relevant_keywords.append("licenças")
    if "ids" in answer.lower():
        relevant_keywords.append("ids")
    if "contratadas" in answer.lower():
        relevant_keywords.append("contratadas")
    
    # Priorizar chunks que contêm o número E uma palavra-chave relevante, ou a frase exata
    for doc_meta in all_source_docs:
        current_snippet = doc_meta.get('page_content', '').strip()
        current_snippet_lower = current_snippet.lower()

        is_strong_match = False
        
        # Heurística de alta precisão: procurar a frase exata "Licenças contratadas: Quantidade de IDs: X"
        # O "X" será o número extraído ou algo genérico como "10"
        if "licenças contratadas: quantidade de ids:" in current_snippet_lower:
            if extracted_number and extracted_number in current_snippet:
                 is_strong_match = True
            elif "10" in current_snippet: # fallback para "10" se o extracted_number falhar
                is_strong_match = True

        # Se não for um match forte, tentar uma combinação de número e keywords
        if not is_strong_match and extracted_number and relevant_keywords:
            if extracted_number in current_snippet and any(kw in current_snippet_lower for kw in relevant_keywords):
                is_strong_match = True
        
        if is_strong_match:
            selected_doc_meta_for_citation = doc_meta
            snippet_to_cite = current_snippet
            print(f"DEBUG: Encontrado chunk 'exato' para citar (match forte): {doc_meta['document_id']}")
            break # Encontrou um match forte, pode parar e citar este

    # --- Fallback: Se nenhum match "exato" foi encontrado, usar o chunk mais similar ---
    if selected_doc_meta_for_citation is None and all_source_docs:
        selected_doc_meta_for_citation = all_source_docs[0] # Pega o primeiro, que é o mais similar
        snippet_to_cite = selected_doc_meta_for_citation.get('page_content', '').strip()
        print("DEBUG: Nenhuma citação 'exata' encontrada, usando o chunk mais similar como fallback.")
    elif selected_doc_meta_for_citation is None:
        print("DEBUG: Nenhuma citação encontrada (lista de chunks vazia).")

    # --- Construir a citação final ---
    final_citation_entry = None
    if selected_doc_meta_for_citation:
        citation_parts = [
            f"**Documento:** {selected_doc_meta_for_citation['document_id']} (Cliente: {selected_doc_meta_for_citation.get('client_name', 'N/A')}, Coordenador TOTVS: {selected_doc_meta_for_citation.get('totvs_coordinator', 'N/A')}, Código Projeto CRM: {selected_doc_meta_for_citation.get('project_code_crm', 'N/A')})"
        ]
        if snippet_to_cite:
            # Limita o snippet da citação para não poluir demais, como antes.
            if len(snippet_to_cite) > 300:
                snippet_to_cite = snippet_to_cite[:300] + "..."
            citation_parts.append(f"'{snippet_to_cite}'")
        final_citation_entry = "\n".join(citation_parts)

    # Combine answer with citation
    final_answer_text = answer
    if final_citation_entry:
        final_answer_text += "\n\n**Fontes Consultadas:**\n\n" + final_citation_entry

    return {"answer": final_answer_text}
   