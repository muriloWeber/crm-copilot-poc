# src/front_end/app.py

import sys
import os
import streamlit as st
import time
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage # <--- NOVA IMPORTAÇÃO AQUI
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO) 

# Adiciona a raiz do projeto ao sys.path para que o Python encontre o pacote 'src'
# independentemente de como o Streamlit é executado.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insere no início para dar prioridade

# --- Importar as definições do agente que criamos ---
# Estes imports agora funcionarão porque a raiz do projeto está no sys.path
from src.core.copilot_agent import AgentState, retrieve_context_node, generate_response_node, format_citation_node

# Carrega as variáveis do .env no início do script
load_dotenv()

# --- Configuração do Agente LangGraph ---
def create_agent_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("format_citation", format_citation_node)

    workflow.set_entry_point("retrieve_context")
    workflow.add_edge("retrieve_context", "generate_response")
    workflow.add_edge("generate_response", "format_citation")
    workflow.add_edge("format_citation", END)

    return workflow.compile()

# Instanciar o workflow uma vez
# Isso também inicializa o modelo de embeddings e o ChromaDB
# Certifique-se que o build_knowledge_base já foi executado!
try:
    copilot_agent = create_agent_workflow()
    st.session_state.llm_initialized = True
except Exception as e:
    st.error(f"Erro ao inicializar o Copilot: {e}. Certifique-se de que a base de conhecimento foi construída e as variáveis de ambiente estão corretas.")
    st.session_state.llm_initialized = False


# --- TCRM Copilot Streamlit UI ---
st.set_page_config(page_title="TCRM Copilot", page_icon="��")

st.title("🤖 TCRM Copilot")
st.markdown("Seu assistente de IA para projetos TOTVS CRM.")

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
            # --- CONVERSÃO DAS MENSAGENS E ESTADO INICIAL COMPLETO (PONTO CRÍTICO CORRIGIDO!) ---
            lc_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    lc_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    lc_messages.append(AIMessage(content=msg["content"]))

            # Prepare initial state for the agent, ensuring all AgentState fields are initialized
            initial_agent_state = AgentState(
                question=prompt,
                context=[],         # Será populado por retrieve_context_node
                source_docs=[],     # Será populado por retrieve_context_node
                answer="",          # Será populado por generate_response_node
                messages=lc_messages, # Mensagens convertidas para o formato LangChain
                filters={}          # Inicializado como dicionário vazio
            )

            # Invocar o agente LangGraph
            response = copilot_agent.invoke(initial_agent_state) # <--- Passando o estado inicial completo
            
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
            st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
            agent_final_response_content = "Ops! Parece que algo deu errado. Por favor, tente novamente."
            message_placeholder.markdown(agent_final_response_content) # Exibe a mensagem de erro

    # Apenas UMA VEZ, adicione a resposta final (limpa e completa) ao histórico da sessão
    st.session_state.messages.append({"role": "assistant", "content": agent_final_response_content})