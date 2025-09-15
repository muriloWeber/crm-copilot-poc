# TCRM Copilot - Prova de Conceito (PoC)

## Visão Geral
Este repositório contém o código-fonte para a Prova de Conceito (PoC) do **TCRM Copilot**, um agente de IA interno focado em otimizar a fase "Durante o Projeto" do TOTVS CRM. O objetivo principal é automatizar tarefas e gerar inteligência de negócio, respondendo perguntas sobre documentos de projeto e dados provenientes da API do TOTVS CRM, sempre fornecendo contexto e citando as fontes originais.

## Objetivos da PoC
*   Validar a capacidade do Copilot em responder perguntas complexas usando **Recuperação e Geração Aumentada (RAG)** a partir de múltiplas fontes.
*   Demonstrar a integração com documentos de projeto (`.pdf`, `.docx`, `.txt`) e dados em tempo real da API do TOTVS CRM.
*   Estabelecer a base para funcionalidades de alto valor, como a geração assistida de documentos de handover.

## Tecnologias Chave
*   **Orquestração de Agentes:** LangGraph
*   **Framework de LLM:** LangChain
*   **Modelo de Linguagem:** OpenAI GPT Models (gpt-4o-mini para testes iniciais)
*   **Metodologia:** RAG (Retrieval-Augmented Generation)
*   **Banco de Vetores:** ChromaDB
*   **Interface de Usuário (Front-end):** Streamlit
*   **Linguagem:** Python

## Estrutura do Projeto
crm-copilot-poc/ 
├── .env.example 
├── .gitignore 
├── README.md 
├── requirements.txt 
├── src/ 
│   ├── core/ 
│   ├── data_ingestion/ 
│   ├── frontend/ 
│   ├── rag_system/ 
│   └── utils/ 
|── data/
│   ├── processed_data/
│   ├── raw_documents/ 
│   └── api_sample_responses/ 
├── config/ 
├── tests/ 
└── docs/

## Configuração do Ambiente de Desenvolvimento

Para configurar o ambiente e começar a desenvolver o TCRM Copilot, siga os passos abaixo:

### 1. Clonar o Repositório

Primeiro, clone este repositório para sua máquina local.

```bash
git clone https://github.com/SeuUsuario/crm-copilot-poc.git
cd crm-copilot-poc
(Lembre-se de substituir SeuUsuario pelo seu nome de usuário do GitHub.)