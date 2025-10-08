import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import httpx
import time
from chromadb.config import Settings
from dotenv import load_dotenv


# --- Configuração ---
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/chroma_db'))
RAW_DOCUMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw_documents'))

@st.cache_resource
def get_embedding_model():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        st.error("Variável de ambiente OPENAI_API_KEY não configurada. Por favor, configure-a para prosseguir.")
        st.stop()

    custom_http_client = httpx.Client(verify=False)
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        http_client=custom_http_client
    )

@st.cache_resource
def get_vector_store(_embeddings_model):
    # Verifica se há documentos brutos. Se não, não há o que indexar.
    if not os.path.exists(RAW_DOCUMENTS_DIR) or not os.listdir(RAW_DOCUMENTS_DIR):
        st.warning(f"O diretório de documentos raw '{RAW_DOCUMENTS_DIR}' está vazio ou não existe. "
                   "Adicione documentos em '/data/raw_documents' e execute o script de ingestão para construir a base de conhecimento.")
        return None

    # Verifica se o diretório do ChromaDB existe. Se não, avisa para rodar o builder.
    if not os.path.exists(CHROMA_DB_DIRECTORY) or not os.listdir(CHROMA_DB_DIRECTORY):
        st.warning(f"ChromaDB não encontrado em: {CHROMA_DB_DIRECTORY}. "
                   "Execute o script `python -m src.data_ingestion.knowledge_base_builder` no terminal "
                   "para popular a base de conhecimento antes de usar o Streamlit.")
        return None

    try:
        # Carrega o ChromaDB existente
        return Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=_embeddings_model,
            client_settings=Settings(allow_reset=True)
        )
    except Exception as e:
        st.error(f"Erro ao carregar o ChromaDB: {e}. Verifique se o diretório existe e está íntegro.")
        return None


# --- Streamlit UI ---
st.set_page_config(page_title="TCRM Copilot PoC - RAG Demo", layout="wide")

st.title("TCRM Copilot PoC - Demonstração RAG (Recuperação de Chunks)")
st.markdown("Bem-vindo, Murilo! Aqui você pode testar a capacidade de recuperação de informação da nossa base de conhecimento RAG, utilizando filtros.")

embeddings_model = get_embedding_model()
vector_store = get_vector_store(embeddings_model) # AQUI o ChromaDB é carregado

# --- Bloco de Instruções e Status na Sidebar ---
st.sidebar.markdown("### Status e Instruções")
if vector_store:
    st.sidebar.success("Base de conhecimento ChromaDB carregada com sucesso!")
else:
    st.sidebar.error("Base de conhecimento ChromaDB NÃO carregada!")
    st.sidebar.info("Para construir ou atualizar a base de conhecimento, siga estes passos:")
    st.sidebar.code(
        "1. Abra um terminal na pasta raiz do projeto.\n"
        "2. Ative o ambiente virtual: `.\venv\Scripts\Activate.ps1`\n"
        "3. Execute o script de ingestão: `python -m src.data_ingestion.knowledge_base_builder`\n"
        "4. Após a conclusão, reinicie este aplicativo Streamlit."
    )
    # Se não há vector_store, não há por que continuar com a UI de busca
    if not st.session_state.get('show_empty_db_warning', False):
        st.session_state['show_empty_db_warning'] = True
        st.experimental_rerun() # Reruns para exibir a mensagem e parar.

# --- Contador de Chunks no ChromaDB ---
chunks_count = 0
try:
    if vector_store: # Só tenta contar se o vector_store foi carregado com sucesso
        num_chunks_ids = vector_store.get(include=[])['ids']
        chunks_count = len(num_chunks_ids)
        st.sidebar.markdown(f"**Chunks na Base:** `{chunks_count}`")
        if chunks_count == 0:
            st.sidebar.warning("A base de conhecimento está vazia. Adicione documentos e siga as instruções acima.")
    else:
        st.sidebar.markdown(f"**Chunks na Base:** `{chunks_count}`")

except Exception as e:
    st.sidebar.error(f"Erro ao contar chunks no ChromaDB: {e}")


st.sidebar.header("Filtros de Busca RAG")
document_id_filter = st.sidebar.text_input("Filtrar por ID do Documento (e.g., MIT041)", key="doc_id_filter")
client_name_filter = st.sidebar.text_input("Filtrar por Nome do Cliente (e.g., SCENS INDUSTRIA E COMERCIO DE FRAGRANCIAS LTDA)", key="client_name_filter")
totvs_coordinator_filter = st.sidebar.text_input("Filtrar por Coordenador TOTVS (e.g., Vanessa Holdefer)", key="totvs_coordinator_filter")

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
        active_filters = {"$and": filter_clauses}
    else:
        active_filters = filter_clauses[0]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Pergunte algo sobre os documentos do projeto...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        st.markdown("Buscando na base de conhecimento...")
        
        if not vector_store:
            st.warning("ChromaDB não carregado. Não é possível realizar a busca. Por favor, siga as instruções na sidebar para construir/atualizar a base.")
            st.session_state.messages.append({"role": "assistant", "content": "ChromaDB não carregado. Não é possível realizar a busca."})
            st.stop() # Interrompe a execução aqui para evitar erros de vector_store None

        try:
            if active_filters:
                st.info(f"Aplicando filtros: {active_filters}")
                retrieved_docs = vector_store.similarity_search(user_query, k=5, filter=active_filters)
            else:
                st.info("Nenhum filtro aplicado. Buscando em toda a base.")
                retrieved_docs = vector_store.similarity_search(user_query, k=5)
            
            if retrieved_docs:
                st.success(f"Foram recuperados {len(retrieved_docs)} chunks relevantes:")
                
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
                    st.code(doc.page_content, language='markdown')
                    st.markdown("---")
                
                st.session_state.messages.append({"role": "assistant", "content": "Chunks recuperados e exibidos acima."})
            else:
                st.warning("Nenhum chunk relevante encontrado com os critérios de busca e filtro.")
                st.session_state.messages.append({"role": "assistant", "content": "Não encontrei informações relevantes na base de conhecimento com os filtros aplicados."})

        except Exception as e:
            st.error(f"Ocorreu um erro durante a busca: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"Desculpe, ocorreu um erro durante a busca: {e}"})

# Opcional: Adiciona um aviso se o vector_store não for carregado, e se o Streamlit foi recarregado por isso
if st.session_state.get('show_empty_db_warning', False) and not vector_store:
    st.info("Por favor, construa a base de conhecimento conforme as instruções na sidebar para começar a interagir.")
    st.session_state['show_empty_db_warning'] = False # Reseta para não entrar em loop de rerun.