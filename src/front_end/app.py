# src/front_end/app.py

import sys
import os
import re

# Desativa a telemetria do ChromaDB (evita erros de SSL com posthog.com)
os.environ['CHROMA_ANALYTICS'] = 'false'

import streamlit as st
import time
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
# Removido 'StateGraph, END' daqui pois build_graph já importa
# from langgraph.graph import StateGraph, END

# Configura o logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Mantenha INFO para evitar muita verbosidade da rede
logger = logging.getLogger(__name__)

# Adiciona a raiz do projeto ao sys.path para que o Python encontre o pacote 'src'
# independentemente de como o Streamlit é executado.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insere no início para dar prioridade

# Carrega as variáveis do .env no início do script
load_dotenv()

# --- VERIFICAÇÃO DE AMBIENTE: ADICIONADO PARA DEBUG ---
openai_api_key_status = os.getenv("OPENAI_API_KEY")
if openai_api_key_status:
    logger.info("OPENAI_API_KEY foi carregada com sucesso (valor não exibido por segurança).")
else:
    logger.error("FALHA: OPENAI_API_KEY NÃO FOI CARREGADA. Verifique seu arquivo .env e seu posicionamento.")
    st.error("Aviso: OPENAI_API_KEY não foi carregada. Verifique o console para mais detalhes.")
# --- FIM VERIFICAÇÃO DE AMBIENTE ---

# --- Importar as definições do agente que criamos ---
# Agora importamos a função build_graph diretamente, que já tem o grafo completo
from src.core.copilot_agent import AgentState, build_graph # <--- IMPORTAÇÃO CORRIGIDA!
from src.data_ingestion.incremental_ingestor import add_document_to_vector_store, get_chroma_instance, get_embeddings_model
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importar text_splitter para inicialização


# --- Configuração do Agente LangGraph ---
# REMOVIDA A FUNÇÃO create_agent_workflow() DAQUI, POIS USAMOS build_graph() DO copilot_agent.py

# Inicializa o agente e componentes na primeira execução do Streamlit
if "copilot_agent" not in st.session_state:
    try:
        st.session_state.copilot_agent = build_graph() # <--- AGORA CHAMAMOS build_graph() CORRETAMENTE!
        st.session_state.llm_initialized = True
        logger.info("Copilot Agent inicializado com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao inicializar o Copilot Agent: {e}. Verifique as variáveis de ambiente e a base de conhecimento.", exc_info=True)
        st.error(f"Erro ao inicializar o Copilot: {e}. Certifique-se de que a base de conhecimento foi construída e as variáveis de ambiente estão corretas.")
        st.session_state.llm_initialized = False

# Inicializa o text_splitter e a instância do ChromaDB
if "text_splitter" not in st.session_state:
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

if "vector_store" not in st.session_state:
    try:
        st.session_state.vector_store = get_chroma_instance()
        logger.info("ChromaDB instance loaded.")
    except Exception as e:
        logger.error(f"Erro ao carregar instância do ChromaDB: {e}", exc_info=True)
        st.error(f"Erro ao carregar base de vetores: {e}")

# --- TCRM Copilot Streamlit UI ---
st.set_page_config(page_title="TCRM Copilot", page_icon="🤖")

st.title("🤖 TCRM Copilot")
st.markdown("Seu assistente de IA para projetos TOTVS CRM.")

with st.sidebar:
    st.header("⚙️ Gerenciamento da Base de Conhecimento")
    st.subheader("Upload de Novo Documento")

    uploaded_file = st.file_uploader(
        "Selecione um documento (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        help="Faça o upload de um novo documento para adicioná-lo à base de conhecimento do Copilot."
    )

    # Campos para metadados adicionais
    client_name_for_upload = st.text_input(
        "Nome do Cliente (Opcional)",
        key="upload_client_name", # Adicionado key para evitar problemas de re-render
        help="Ex: KION, MARSON, SCENS. Use para filtrar em futuras consultas. Deixe em branco se não aplicável."
    )
    doc_type_for_upload = st.text_input(
        "Tipo de Documento (Opcional)",
        key="upload_doc_type", # Adicionado key
        help="Ex: Escopo Técnico, Ordem de Serviço. Ajuda a categorizar o documento."
    )
    project_code_for_upload = st.text_input(
        "Código do Projeto (Opcional)",
        key="upload_project_code", # Adicionado key
        help="Ex: D000071597001. Para vincular a um projeto específico."
    )

    if st.button("Adicionar Documento à Base"):
        if uploaded_file is not None:
            st.info("Processando documento para adicionar à base de conhecimento...")
            try:
                # Salvar o arquivo temporariamente para que o loader possa acessá-lo
                temp_file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Construir o dicionário detected_metadata
                metadata_for_ingestion = {
                    "client_name": client_name_for_upload if client_name_for_upload else None,
                    "doc_type": doc_type_for_upload if doc_type_for_upload else None,
                    "project_code": project_code_for_upload if project_code_for_upload else None
                }
                
                # Chamar a função de ingestão com o caminho do arquivo temporário e os metadados
                add_document_to_vector_store(
                    file_path=temp_file_path,
                    vector_store=st.session_state.vector_store,
                    text_splitter=st.session_state.text_splitter,
                    detected_metadata=metadata_for_ingestion
                )
                st.success(f"Documento '{uploaded_file.name}' adicionado com sucesso!")
                st.toast("Upload concluído!", icon="✅")
                
                # Opcional: remover o arquivo temporário após o processamento
                os.remove(temp_file_path)
                logger.info(f"Arquivo temporário {temp_file_path} removido.")

            except Exception as e:
                logger.error(f"Erro inesperado ao adicionar documento: {e}", exc_info=True)
                st.error(f"Erro inesperado ao adicionar documento: {e}")
                st.toast("Erro no upload!", icon="🚨")
        else:
            st.warning("Por favor, selecione um arquivo para fazer o upload.")
            st.toast("Nenhum arquivo selecionado!", icon="⚠️")

# Inicializar histórico de chat na sessão do Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []
# Ensure llm_initialized is set even if not initially found, to prevent KeyError
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False # Default to False if not set by try/except


# Display de mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de chat
if prompt := st.chat_input("Pergunte ao Copilot sobre seu projeto..."):
    if not st.session_state.llm_initialized:
        st.warning("O Copilot não foi inicializado corretamente. Verifique os logs.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Variável para acumular a simulação de digitação
        accumulated_text_for_display = ""
        # Variável para guardar a resposta FINAL do agente
        agent_final_response_content = ""

        try:
            # --- CONVERSÃO DAS MENSAGENS E ESTADO INICIAL COMPLETO ---
            lc_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

            # --- REMOVIDA A EXTRAÇÃO DE METADADOS DA PERGUNTA DO USUÁRIO AQUI ---
            # Essa extração agora é feita pelo 'extract_filters_node' no grafo.
            
            initial_agent_state = AgentState(
                question=prompt,
                context=[],         # Será populado por retrieve_context_node
                source_docs=[],     # Será populado por retrieve_context_node
                answer="",          # Será populado por generate_response_node
                messages=lc_messages, # Mensagens convertidas para o formato LangChain
                filters={} # <--- FILTROS INICIALIZADOS VAZIOS. extract_filters_node irá populá-los.
            )

            # Invocar o agente LangGraph
            response = st.session_state.copilot_agent.invoke(initial_agent_state)
            
            # A resposta final formatada já deve estar em response['answer']
            agent_final_response_content = response.get("answer", "Não consegui gerar uma resposta para isso. Tente refazer a pergunta ou fornecer mais contexto.")

            # Simular digitação usando a variável temporária
            for chunk in agent_final_response_content.split(" "):
                accumulated_text_for_display += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(accumulated_text_for_display + "▌")
            
            # Exiba a resposta COMPLETA e FINAL (sem o cursor piscando)
            message_placeholder.markdown(agent_final_response_content)

        except Exception as e:
            logger.error(f"Ocorreu um erro ao processar sua pergunta: {e}", exc_info=True)
            st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
            agent_final_response_content = "Ops! Parece que algo deu errado. Por favor, tente novamente."
            message_placeholder.markdown(agent_final_response_content) # Exibe a mensagem de erro

    # Apenas UMA VEZ, adicione a resposta final (limpa e completa) ao histórico da sessão
    st.session_state.messages.append({"role": "assistant", "content": agent_final_response_content})