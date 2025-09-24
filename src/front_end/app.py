# src/front_end/app.py

import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_ingestion.knowledge_base_builder import (
    initialize_embedding_model,
    CHROMA_DB_DIRECTORY,
    build_knowledge_base_from_documents
)
from langchain_community.vectorstores import Chroma

# --- Configurações Iniciais ---
# Verifique se as variáveis de ambiente estão configuradas
if "OPENAI_API_KEY" not in os.environ:
    st.error("A variável de ambiente OPENAI_API_KEY não está configurada.")
    st.stop()
if "OPENAI_API_BASE" not in os.environ:
    st.warning("A variável de ambiente OPENAI_API_BASE não está configurada. Usando o endpoint padrão da OpenAI.")

COLLECTION_NAME = "tcrm_copilot_poc_kb"
# Ajuste o RAW_DOCUMENTS_PATH para ser relativo à raiz do projeto
RAW_DOCUMENTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw_documents'))

# --- Funções de Ingestão e Carregamento da KB ---
@st.cache_resource
def get_embedding_model():
    """Inicializa e cacheia o modelo de embedding."""
    return initialize_embedding_model()

@st.cache_resource
def get_knowledge_base(_embedding_model): # LINHA ALTERADA: Adicionado underscore antes de 'embedding_model'
    """Carrega a base de conhecimento ChromaDB existente."""

    if not _embedding_model: 
        st.error("Modelo de embeddings não inicializado. Não é possível carregar a base de conhecimento.")
        return None
    
    db_path = os.path.join(CHROMA_DB_DIRECTORY, COLLECTION_NAME)
    if not os.path.exists(db_path):
        st.warning(f"Base de conhecimento não encontrada em {db_path}. Por favor, ingira os documentos primeiro.")
        return None
        
    try:
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=_embedding_model, 
            collection_name=COLLECTION_NAME
        )
        st.success("Base de conhecimento carregada com sucesso!")
        return vector_store
    except Exception as e:
        st.error(f"Erro ao carregar a base de conhecimento: {e}")
        return None

def ingest_documents_to_kb():
    """Função para re-ingestão de documentos."""
    st.info("Iniciando ingestão de documentos. Isso pode levar alguns minutos...")
    doc_paths = []
    for file_name in os.listdir(RAW_DOCUMENTS_PATH):
        if file_name.endswith(('.pdf', '.docx', '.txt')):
            doc_paths.append(os.path.join(RAW_DOCUMENTS_PATH, file_name))
    
    if not doc_paths:
        st.warning("Nenhum documento encontrado na pasta 'data/raw_documents' para ingestão.")
        return None
        
    kb = build_knowledge_base_from_documents(doc_paths, collection_name=COLLECTION_NAME)
    if kb:
        st.success("Ingestão concluída e base de conhecimento atualizada!")
    else:
        st.error("Falha na ingestão de documentos.")
    return kb

# --- UI Streamlit ---
st.set_page_config(page_title="TCRM Copilot", layout="wide")
st.title("🤖 TCRM Copilot")

# Sidebar para configurações e ações
with st.sidebar:
    st.header("Configurações e Ferramentas")
    
    # Botão para re-ingestão de documentos
    if st.button("Re-ingestão de Documentos"):
        # Limpa o cache para garantir que a nova KB seja carregada
        st.cache_resource.clear()
        kb = ingest_documents_to_kb()
        if kb:
            st.session_state['knowledge_base'] = kb
            st.rerun() # Reinicia a aplicação para carregar a nova KB
    
    st.markdown("---")
    st.info("O TCRM Copilot responde a perguntas sobre documentos de projeto e dados do TCRM (futuramente).")
    st.caption("Desenvolvido por Murilo Weber")

# Inicializa o modelo de embeddings uma vez
embedding_model = get_embedding_model()

# Carrega ou cria a base de conhecimento. Usamos st.session_state para manter o estado.
if 'knowledge_base' not in st.session_state or st.session_state['knowledge_base'] is None:
    st.session_state['knowledge_base'] = get_knowledge_base(embedding_model)
    # Se a KB ainda não existir após a tentativa de carregamento, informa o usuário
    if st.session_state['knowledge_base'] is None and not os.path.exists(os.path.join(CHROMA_DB_DIRECTORY, COLLECTION_NAME)):
        st.warning("A base de conhecimento não foi encontrada. Por favor, clique em 'Re-ingestão de Documentos' na barra lateral para começar.")

# Inicializa o histórico do chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Lógica de Chat ---
if prompt := st.chat_input("Pergunte algo ao Copilot..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Pensando..."):
        if st.session_state['knowledge_base']:
            try:
                # Simulação da etapa de retrieve_context_node
                results = st.session_state['knowledge_base'].similarity_search(prompt, k=3)
                context = "\n\n".join([doc.page_content for doc in results])
                sources = [f"* {doc.metadata.get('source', 'Desconhecido')} (chunk: {doc.page_content[:50]}...)" for doc in results]

                response_content = f"Simulação de resposta baseada no contexto recuperado:\n\n{context}\n\n**Fontes:**\n" + "\n".join(sources)
                
                with st.chat_message("assistant"):
                    st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})

            except Exception as e:
                st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Desculpe, ocorreu um erro: {e}"})
        else:
            response_content = "A base de conhecimento não está disponível. Por favor, ingira os documentos primeiro."
            with st.chat_message("assistant"):
                st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})
