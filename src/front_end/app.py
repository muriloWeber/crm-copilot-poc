import streamlit as st
import os
# Mudança AQUI: Importar Chroma do novo pacote
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import httpx # Necessário para o cliente HTTP personalizado se usando proxy
import subprocess # Para o botão de re-ingestão
import sys # Adicionar para pegar o executável Python do ambiente

# --- Configuração (Adaptada do knowledge_base_builder.py) ---
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
# Caminho para o diretório do ChromaDB (relativo a src/front_end)
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/chroma_db'))

# --- Função Auxiliar para Inicializar o Modelo de Embeddings ---
# Usamos st.cache_resource para evitar carregar o modelo múltiplas vezes
@st.cache_resource
def get_embedding_model():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        st.error("Variável de ambiente OPENAI_API_KEY não configurada. Por favor, configure-a para prosseguir.")
        st.stop() # Interrompe a execução do Streamlit

    # Cliente HTTP personalizado para lidar com proxies ou certificados SSL (se necessário)
    # ATENÇÃO: verify=False desabilita a verificação de certificados SSL e deve ser usado com cautela em produção.
    # É útil em ambientes de desenvolvimento ou com proxies que interceptam SSL.
    custom_http_client = httpx.Client(verify=False)

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        http_client=custom_http_client
    )

# --- Função para Carregar o ChromaDB ---
# CORREÇÃO: Argumento _embeddings_model com underscore para evitar UnhashableParamError
@st.cache_resource
def get_vector_store(_embeddings_model):
    if not os.path.exists(CHROMA_DB_DIRECTORY):
        st.error(f"ChromaDB não encontrado em: {CHROMA_DB_DIRECTORY}. Por favor, execute "
                 "`python -m src.data_ingestion.knowledge_base_builder` no terminal primeiro.")
        st.stop()

    try:
        # Carrega o ChromaDB existente
        return Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=_embeddings_model
        )
    except Exception as e:
        st.error(f"Erro ao carregar o ChromaDB: {e}. Verifique se o diretório existe e está íntegro.")
        st.stop()


# --- Streamlit UI ---
st.set_page_config(page_title="TCRM Copilot PoC - RAG Demo", layout="wide")

st.title("TCRM Copilot PoC - Demonstração RAG (Recuperação de Chunks)")
st.markdown("Bem-vindo, Murilo! Aqui você pode testar a capacidade de recuperação de informação da nossa base de conhecimento RAG, utilizando filtros.")

# Inicializa o modelo de embeddings e o vector store (ChromaDB)
embeddings_model = get_embedding_model()
vector_store = get_vector_store(embeddings_model)


# --- Sidebar para Configuração de Filtros ---
st.sidebar.header("Filtros de Busca RAG")
document_id_filter = st.sidebar.text_input("Filtrar por ID do Documento (e.g., MIT041)", key="doc_id_filter")
client_name_filter = st.sidebar.text_input("Filtrar por Nome do Cliente (e.g., SCENS INDUSTRIA E COMERCIO DE FRAGRANCIAS LTDA)", key="client_name_filter")
totvs_coordinator_filter = st.sidebar.text_input("Filtrar por Coordenador TOTVS (e.g., Vanessa Holdefer)", key="totvs_coordinator_filter")

# Constrói o dicionário de filtros para o ChromaDB
active_filters = {}
filter_clauses = []
if document_id_filter:
    filter_clauses.append({"document_id": document_id_filter})
if client_name_filter:
    filter_clauses.append({"client_name": client_name_filter})
if totvs_coordinator_filter:
    filter_clauses.append({"totvs_coordinator": totvs_coordinator_filter})

if filter_clauses:
    if len(filter_clauses) > 1:
        # Para múltiplos filtros, usa o operador "$and" (ChromaDB requer isso)
        active_filters = {"$and": filter_clauses}
    else:
        # Se houver apenas um filtro, não precisa do "$and"
        active_filters = filter_clauses[0]

# --- Interface Principal do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada para a pergunta do usuário
user_query = st.chat_input("Pergunte algo sobre os documentos do projeto...")

if user_query:
    # Adiciona a pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        st.markdown("Buscando na base de conhecimento...")
        
        try:
            # Realiza a busca de similaridade com ou sem filtros
            if active_filters:
                st.info(f"Aplicando filtros: {active_filters}")
                retrieved_docs = vector_store.similarity_search(user_query, k=5, filter=active_filters)
            else:
                st.info("Nenhum filtro aplicado. Buscando em toda a base.")
                retrieved_docs = vector_store.similarity_search(user_query, k=5)
            
            if retrieved_docs:
                st.success(f"Foram recuperados {len(retrieved_docs)} chunks relevantes:")
                
                # Exibe cada chunk com seus metadados
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source', 'N/A')
                    doc_id = doc.metadata.get('document_id', 'N/A')
                    client_name = doc.metadata.get('client_name', 'N/A')
                    totvs_coord = doc.metadata.get('totvs_coordinator', 'N/A')
                    project_code_crm = doc.metadata.get('project_code_crm', 'N/A')


                    st.markdown(f"**--- Chunk {i+1} ---**")
                    st.text(f"Source: {source}")
                    st.text(f"Document ID: {doc_id}")
                    st.text(f"Client Name: {client_name}")
                    st.text(f"TOTVS Coordinator: {totvs_coord}")
                    st.text(f"Project Code CRM: {project_code_crm}")
                    st.code(doc.page_content, language='markdown') # Exibe o conteúdo do chunk formatado
                    st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": "Chunks recuperados e exibidos acima."})
            else:
                st.warning("Nenhum chunk relevante encontrado com os critérios de busca e filtro.")
                st.session_state.messages.append({"role": "assistant", "content": "Não encontrei informações relevantes na base de conhecimento com os filtros aplicados."})

        except Exception as e:
            st.error(f"Ocorreu um erro durante a busca: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Desculpe, ocorreu um erro durante a busca: {e}"})

# --- Botão de Re-ingestão na Sidebar ---
if st.sidebar.button("Re-Ingest (Rebuild Knowledge Base)", type="primary"):
    st.sidebar.info("Iniciando a reconstrução da base de conhecimento... Por favor, aguarde.")
    try:
        # Pega o caminho do executável Python do ambiente virtual atual
        python_executable = sys.executable
        # Define o diretório de trabalho como a raiz do projeto (importante para resolver módulos)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        result = subprocess.run(
            [python_executable, "-m", "src.data_ingestion.knowledge_base_builder"],
            cwd=project_root, # Executa a partir da raiz do projeto
            capture_output=True, text=True, check=True
        )
        st.sidebar.success("Base de conhecimento reconstruída com sucesso!")
        st.sidebar.text("Log de Re-ingestão:")
        st.sidebar.code(result.stdout)
        if result.stderr:
            st.sidebar.error("Erros durante a re-ingestão:")
            st.sidebar.code(result.stderr)
        
        # Limpa os caches para forçar o Streamlit a recarregar o modelo e o ChromaDB
        st.cache_resource.clear()
        st.experimental_rerun() # Recarrega a aplicação Streamlit
    except subprocess.CalledProcessError as e:
        st.sidebar.error(f"Erro ao reconstruir a base de conhecimento: {e.cmd} retornou status {e.returncode}. Erro: {e.stderr}")
        st.sidebar.code(e.output) # Mostra o stdout e stderr combinados do processo
    except Exception as e:
        st.sidebar.error(f"Erro inesperado durante a re-ingestão: {e}")