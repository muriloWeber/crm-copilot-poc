# src/core/copilot_agent.py
import os
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma
import httpx
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

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

# --- 2. Configurações e Inicialização do LLM ---
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

llm = get_llm()


# --- 3. Definição dos Nós (Nodes) ---

def retrieve_context_node(state: AgentState, vector_store: Chroma, filters: Optional[dict] = None):
    """
    Nó para recuperar chunks relevantes da base de conhecimento.
    Recebe a pergunta do usuário e quaisquer filtros ativos.
    """
    print(f"DEBUG: Executando retrieve_context_node para a pergunta: {state['question']}")
    question = state["question"]

    # Usa os filtros diretamente como são passados.
    # Assumimos que as chaves em 'filters' correspondem às chaves nos metadados do ChromaDB
    # definidos durante a ingestão pelo knowledge_base_builder.py.
    # Ex: filters = {"document_id": "document.pdf", "client_name": "SCENS"}
    if filters:
        print(f"DEBUG: Aplicando filtros na recuperação: {filters}")
        retrieved_docs = vector_store.similarity_search(question, k=5, filter=filters)
    else:
        print("DEBUG: Nenhuns filtros ativos. Recuperando da base completa.")
        retrieved_docs = vector_store.similarity_search(question, k=5)

    source_docs_metadata = []
    for doc in retrieved_docs:
        source_docs_metadata.append({
            "source": doc.metadata.get('source', 'N/A'), # O caminho completo do arquivo
            "document_id": doc.metadata.get('document_id', 'N/A'), # O nome do arquivo (ex: TCRM_Copilot_Projeto.pdf)
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
             "Se a resposta não puder ser encontrada no contexto fornecido, responda 'Não encontrei a informação no contexto fornecido.' "
             "Não invente informações. Formate sua resposta de forma clara e profissional. "
             "Contexto: {context}"
            ),
            ("human", "{question}"),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()
    response = chain.invoke({"question": question, "context": context_text})
    print("DEBUG: Resposta gerada pelo LLM.")
    return {"answer": response}


def format_citation_node(state: AgentState):
    """
    Nó para formatar a resposta final, incorporando citações detalhadas das fontes.
    """
    print("DEBUG: Executando format_citation_node.")
    answer = state["answer"]
    source_docs = state["source_docs"]

    citations = []
    seen_sources = set() # Para evitar duplicar citações da mesma fonte

    for doc_meta in source_docs:
        # Usamos 'document_id' para o nome do arquivo, que é mais amigável para o usuário.
        # Adicione outros metadados para enriquecer a citação.
        citation_text = f"**{doc_meta['document_id']}**"
        
        # Adiciona detalhes apenas se existirem e forem diferentes de 'N/A'
        details = []
        if doc_meta.get('client_name') and doc_meta['client_name'] != 'N/A':
            details.append(f"Cliente: {doc_meta['client_name']}")
        if doc_meta.get('totvs_coordinator') and doc_meta['totvs_coordinator'] != 'N/A':
            details.append(f"Coordenador TOTVS: {doc_meta['totvs_coordinator']}")
        if doc_meta.get('project_code_crm') and doc_meta['project_code_crm'] != 'N/A':
            details.append(f"Código Projeto CRM: {doc_meta['project_code_crm']}")

        if details:
            citation_text += f" ({', '.join(details)})"

        if citation_text not in seen_sources:
            citations.append(citation_text)
            seen_sources.add(citation_text)

    if citations:
        formatted_citations = "\n".join([f"- {c}" for c in citations])
        final_answer = f"{answer}\n\n**Fontes Consultadas:**\n{formatted_citations}"
    else:
        final_answer = answer # Se não houver citações, apenas retorna a resposta

    print("DEBUG: Citações formatadas.")
    return {"answer": final_answer}