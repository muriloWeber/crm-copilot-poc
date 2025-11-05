import sys
import os
import re
import time
import logging

# Desativa a telemetria do ChromaDB (evita erros de SSL com posthog.com em ambientes restritos)
# Essencial para PoCs e ambientes internos, onde a verificação de certificado pode falhar.
os.environ['CHROMA_ANALYTICS'] = 'false'

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Adiciona a raiz do projeto ao sys.path para que o Python encontre o pacote 'src'
# independentemente de como o Streamlit é executado. Garante que as importações internas funcionem.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insere no início para dar prioridade

# Carrega as variáveis do .env no início do script, garantindo que estejam disponíveis globalmente.
load_dotenv()

# --- Configuração de Logging ---
# Configura o logger para toda a aplicação. Nível INFO é um bom balanço entre verbosidade e informação.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- VERIFICAÇÃO CRÍTICA DE AMBIENTE ---
# Valida a presença da chave da API da OpenAI, crucial para o funcionamento do LLM.
openai_api_key_status = os.getenv("OPENAI_API_KEY")
if openai_api_key_status:
    logger.info("OPENAI_API_KEY foi carregada com sucesso (valor não exibido por segurança).")
else:
    logger.error("FALHA CRÍTICA: OPENAI_API_KEY NÃO FOI CARREGADA. Verifique seu arquivo .env e seu posicionamento.")
    st.error("Aviso: OPENAI_API_KEY não foi carregada. Verifique o console para mais detalhes.")
    # Considerar st.stop() aqui se o funcionamento do LLM for mandatório e não puder prosseguir
    # st.stop() # Descomentar para interromper a execução do Streamlit se a chave não estiver presente.

# --- Importações do Core do Copilot e Ingestão ---
# Importa o estado do agente e a função de construção do grafo principal.
from src.core.copilot_agent import AgentState, build_graph
# Importa funções para ingestão incremental de documentos e acesso ao ChromaDB.
from src.data_ingestion.incremental_ingestor import add_document_to_vector_store, get_chroma_instance, get_embeddings_model
# Importa o text splitter, necessário para o processamento de documentos.
from langchain.text_splitter import RecursiveCharacterTextSplitter


# --- Funções Auxiliares para o Streamlit UI ---

def initialize_copilot_components():
    """
    Inicializa o agente LangGraph e a base de vetores (ChromaDB) na sessão do Streamlit.
    Garante que estes componentes sejam configurados apenas uma vez.
    """
    if "copilot_agent" not in st.session_state:
        try:
            st.session_state.copilot_agent = build_graph()
            st.session_state.llm_initialized = True
            logger.info("Copilot Agent inicializado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao inicializar o Copilot Agent: {e}. Verifique as variáveis de ambiente e a base de conhecimento.", exc_info=True)
            st.error(f"Erro ao inicializar o Copilot: {e}. Certifique-se de que a base de conhecimento foi construída e as variáveis de ambiente estão corretas.")
            st.session_state.llm_initialized = False

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

    # Garante que llm_initialized seja sempre definido, mesmo que a inicialização falhe.
    if "llm_initialized" not in st.session_state:
        st.session_state.llm_initialized = False # Default para False se não for setado.

def handle_document_upload(uploaded_file, client_name, doc_type, project_code):
    """
    Processa o upload de um documento, salvando-o temporariamente e adicionando-o à base de vetores.
    Fornece feedback ao usuário via Streamlit.
    """
    if uploaded_file is None:
        st.warning("Por favor, selecione um arquivo para fazer o upload.")
        st.toast("Nenhum arquivo selecionado!", icon="⚠️")
        return

    st.info("Processando documento para adicionar à base de conhecimento...")
    temp_file_path = None # Inicializa para garantir que esteja acessível no finally
    try:
        # Salva o arquivo temporariamente para processamento.
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Prepara os metadados para a ingestão.
        metadata_for_ingestion = {
            "client_name": client_name if client_name else None,
            "doc_type": doc_type if doc_type else None,
            "project_code": project_code if project_code else None
        }
        
        # Adiciona o documento à base de vetores.
        add_document_to_vector_store(
            file_path=temp_file_path,
            vector_store=st.session_state.vector_store,
            text_splitter=st.session_state.text_splitter,
            detected_metadata=metadata_for_ingestion
        )
        st.success(f"Documento '{uploaded_file.name}' adicionado com sucesso!")
        st.toast("Upload e ingestão concluídos!", icon="✅")

    except Exception as e:
        logger.error(f"Erro inesperado ao adicionar documento: {e}", exc_info=True)
        st.error(f"Erro inesperado ao adicionar documento: {e}")
        st.toast("Erro no upload ou ingestão!", icon="🚨")
    finally:
        # Garante que o arquivo temporário seja removido mesmo em caso de erro.
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Arquivo temporário {temp_file_path} removido.")

def convert_st_messages_to_lc_messages(st_messages):
    """
    Converte o histórico de mensagens do formato Streamlit para o formato LangChain.
    """
    lc_messages = []
    for msg in st_messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages

def display_typing_simulation(message_placeholder, full_response):
    """
    Simula o efeito de digitação para a resposta do assistente no Streamlit.
    """
    accumulated_text = ""
    # Simulação básica: exibe a resposta palavra por palavra.
    # Pode ser ajustado para caracteres ou para simular um tempo de resposta mais natural.
    for chunk in full_response.split(" "):
        accumulated_text += chunk + " "
        time.sleep(0.02) # Ajuste a velocidade de digitação aqui (menor = mais rápido)
        message_placeholder.markdown(accumulated_text + "▌") # Adiciona um cursor piscando

    message_placeholder.markdown(full_response) # Exibe a resposta completa e final.


# --- Configuração da Página do Streamlit ---
st.set_page_config(page_title="TCRM Copilot", page_icon="🤖", layout="centered")

st.title("🤖 TCRM Copilot")
st.markdown("Seu assistente de IA para projetos TOTVS CRM.")


# --- Inicialização dos Componentes do Copilot (Agente e DB) ---
initialize_copilot_components()


# --- Sidebar para Gerenciamento da Base de Conhecimento ---
with st.sidebar:
    st.header("⚙️ Gerenciamento da Base de Conhecimento")
    st.subheader("Upload de Novo Documento")

    uploaded_file = st.file_uploader(
        "Selecione um documento (.pdf, .docx, .txt)",
        type=["pdf", "docx", "txt"],
        help="Faça o upload de um novo documento para adicioná-lo à base de conhecimento do Copilot."
    )

    client_name_for_upload = st.text_input(
        "Nome do Cliente (Opcional)",
        key="upload_client_name",
        help="Ex: KION, MARSON, SCENS. Use para filtrar em futuras consultas."
    )
    doc_type_for_upload = st.text_input(
        "Tipo de Documento (Opcional)",
        key="upload_doc_type",
        help="Ex: Escopo Técnico, Ordem de Serviço, Roteiro. Ajuda a categorizar o documento."
    )
    project_code_for_upload = st.text_input(
        "Código do Projeto (Opcional)",
        key="upload_project_code",
        help="Ex: D000071597001. Para vincular a um projeto específico."
    )

    if st.button("Adicionar Documento à Base", help="Clique para processar e adicionar o arquivo selecionado à base de conhecimento."):
        handle_document_upload(uploaded_file, client_name_for_upload, doc_type_for_upload, project_code_for_upload)


# --- Seção Principal do Chat ---

# Inicializa o histórico de chat na sessão do Streamlit se ainda não existir.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe mensagens anteriores do histórico.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Processa a entrada do usuário no chat.
if prompt := st.chat_input("Pergunte ao Copilot sobre seu projeto..."):
    # Verifica se o LLM foi inicializado com sucesso antes de processar a pergunta.
    if not st.session_state.llm_initialized:
        st.warning("O Copilot não foi inicializado corretamente. Verifique os logs de inicialização.")
        st.stop() # Interrompe a execução para evitar erros.

    # Adiciona a pergunta do usuário ao histórico.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepara para exibir a resposta do assistente.
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        agent_final_response_content = "" # Variável para armazenar a resposta final do agente.

        try:
            # Converte o histórico de mensagens para o formato do LangChain.
            lc_messages = convert_st_messages_to_lc_messages(st.session_state.messages)

            # Prepara o estado inicial do agente com a pergunta e o histórico.
            initial_agent_state = AgentState(
                question=prompt,
                context=[],         # Populado pelo nó 'retrieve_context_node'.
                source_docs=[],     # Populado pelo nó 'retrieve_context_node'.
                answer="",          # Populado pelo nó 'generate_response_node'.
                messages=lc_messages, # Histórico de mensagens para contexto do LLM.
                filters={}          # Populado pelo nó 'extract_filters_node'.
            )

            # Invoca o agente LangGraph para processar a pergunta.
            response = st.session_state.copilot_agent.invoke(initial_agent_state)
            
            # Extrai a resposta final do resultado do agente.
            agent_final_response_content = response.get(
                "answer",
                "Não consegui gerar uma resposta detalhada para isso. Tente reformular a pergunta ou fornecer mais contexto."
            )

            # Exibe a resposta com simulação de digitação.
            display_typing_simulation(message_placeholder, agent_final_response_content)

        except Exception as e:
            logger.error(f"Ocorreu um erro ao processar sua pergunta: {e}", exc_info=True)
            st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
            agent_final_response_content = "Ops! Parece que algo deu errado ao processar sua pergunta. Por favor, tente novamente."
            message_placeholder.markdown(agent_final_response_content) # Exibe a mensagem de erro no Streamlit.

    # Adiciona a resposta final do assistente ao histórico da sessão.
    st.session_state.messages.append({"role": "assistant", "content": agent_final_response_content})
