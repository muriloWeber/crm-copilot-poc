# TCRM Copilot: Agente de IA para Inteligência de Negócio e Otimização de Projetos

## Visão Geral

O **TCRM Copilot** é um agente de Inteligência Artificial interno, desenvolvido para atuar como um assistente inteligente e estratégico focado na fase **"Durante o Projeto"**. Seu propósito primordial é revolucionar a forma como interagimos com as informações do projeto, automatizando tarefas repetitivas e gerando inteligência de negócio acionável.

A Prova de Conceito (PoC) do TCRM Copilot valida a capacidade central do sistema de:
*   **Responder a perguntas complexas** sobre documentos de projeto (escopos, atas, etc.).
*   **Fornecer contexto e citar as fontes** de suas respostas, sejam elas provenientes de documentos ou de informações acessadas via API do TOTVS CRM (TCRM), garantindo rastreabilidade e confiabilidade.

Este projeto visa não apenas otimizar a recuperação de informações, mas também estabelecer uma base sólida para futuras expansões em automação inteligente e geração de insights de alto valor.

## Objetivos da PoC

*   Validar a capacidade do Copilot em responder perguntas complexas usando **Recuperação e Geração Aumentada (RAG)** a partir de múltiplas fontes.
*   Demonstrar a integração com documentos de projeto (`.pdf`, `.docx`, `.txt`) e dados em tempo real da API do TOTVS CRM.
*   Estabelecer a base para funcionalidades de alto valor, como a geração assistida de documentos de handover.

## Tecnologias Chave

A arquitetura do TCRM Copilot é construída sobre um conjunto robusto de tecnologias modernas, garantindo escalabilidade, flexibilidade e poder de processamento:

*   **Orquestração de Agentes**: **LangGraph**
    *   Define a estrutura e o fluxo de estados e transições do agente, permitindo a criação de lógicas complexas e reativas.
*   **Framework de LLM**: **LangChain**
    *   Oferece um ecossistema completo para desenvolvimento de aplicações baseadas em Large Language Models (LLMs), incluindo componentes para carregamento de documentos (loaders), chunking de texto e geração de embeddings.
*   **Modelo de Linguagem**: **OpenAI GPT Models** (Especificamente `gpt-4o-mini` para a PoC inicial via API)
    *   Aproveitamos sua eficiência e custo-benefício para processamento de linguagem natural e geração de respostas.
*   **Metodologia**: **RAG (Retrieval-Augmented Generation)**
    *   Metodologia central que garante que o agente recupere informações relevantes de uma base de conhecimento externa antes de gerar uma resposta, proporcionando maior precisão e evitando alucinações.
*   **Banco de Vetores**: **ChromaDB**
    *   Utilizado para armazenamento eficiente e recuperação rápida de embeddings de texto, formando a base de conhecimento vetorial persistente do Copilot.
*   **Interface de Usuário (Front-end)**: **Streamlit**
    *   Permite a rápida prototipagem e desenvolvimento de uma interface de chat interativa e amigável para o usuário final.
*   **Linguagem**: **Python**
    *   A linguagem de programação principal, escolhida por sua vasta gama de bibliotecas para IA, Data Science e desenvolvimento web.

## Status e Decisões Tomadas na PoC

A fase de Prova de Conceito trouxe insights valiosos e definiu o caminho para o desenvolvimento, com os seguintes pontos de progresso e decisões estratégicas:

*   **Validação da Conectividade do LLM**: Testes bem-sucedidos com o modelo `gpt-4o-mini` via proxy `https://proxy.dta.totvs.ai` confirmaram a operacionalidade da API e a capacidade do "cérebro" do Copilot. O custo por token foi analisado, e a otimização em escala será crucial.
*   **Funcionalidade de Upload de Documentos no Streamlit**: Confirmada a implementação direta de `st.file_uploader` para permitir que o usuário alimente o Copilot com novos documentos de forma prática. Esta funcionalidade já está operacional.
*   **Habilidade e Orientação do Desenvolvedor**: Reconhecido o forte background em Python e Data Science, com foco em desmistificar LangGraph e construção de agentes através de exemplos práticos e uma abordagem modular.
*   **Geração do Documento de Handover**: Identificada como uma "killer feature" de alto valor. Para a PoC, a prioridade é que o Copilot consiga responder a perguntas individuais do handover com suas respectivas fontes. A montagem estruturada do conteúdo (texto/lista) é um *stretch goal* para esta fase, sem foco na geração de um DOCX perfeitamente formatado.
*   **Discussão sobre o DTA da TOTVS**: Reconhecido como um potencial acelerador para integração e automação de fluxos de dados externos. No entanto, sua dependência para as próximas etapas imediatas do PoC foi adiada devido à falta de clareza sobre suas capacidades e integração imediatas. O core do Copilot (RAG, LangGraph, Streamlit) segue desenvolvimento independentemente.

## Próximos Passos Prioritários

Com base nas decisões da PoC e na situação atual do projeto, os próximos focos de desenvolvimento incluem:

1.  **Configuração do Ambiente e Ingestão de Documentos Locais**: Continuar com a configuração do ambiente Python e o desenvolvimento dos scripts para extrair texto de documentos (PDF, DOCX, TXT) e preparar para chunking. (Concluído: `incremental_ingestor.py` em operação).
2.  **Construção da Base de Conhecimento RAG**: Focar na estratégia de chunking, geração de embeddings e população do ChromaDB. (Concluído: Lógica implementada em `incremental_ingestor.py`).
3.  **Integração Básica com TCRM API (se possível)**: Entender a estrutura dos dados da API e como transformá-los para ingestão no ChromaDB, mesmo que por scripts Python diretos.
4.  **Esboço da Orquestração LangGraph**: Começar a definir o `AgentState` e os nós básicos (`retrieve_context`, `generate_response`, `format_citation`). (Implementado e refinado em `copilot_agent.py`).
5.  **Desenvolvimento da Interface Streamlit**: Implementar a UI básica, incluindo a funcionalidade de upload de documentos. (Implementado em `front_end/app.py`).
6.  **Visualização da Arquitetura**: Gerar um fluxograma de alto nível para alinhamento.

### Configuração do Ambiente de Desenvolvimento

Para configurar o ambiente e começar a desenvolver/utilizar o TCRM Copilot, siga os passos abaixo:

### 1. Clonar o Repositório

Primeiro, clone este repositório para sua máquina local.

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd tcrm-copilot
# Lembre-se de substituir <URL_DO_SEU_REPOSITORIO> pela URL real do seu repositório Git.
```

### 2. Criar e Ativar o Ambiente Virtual

Recomenda-se fortemente o uso de um ambiente virtual para gerenciar as dependências do projeto e evitar conflitos.

```bash
python -m venv venv
# Para ativar no Linux/macOS:
source venv/bin/activate
# Para ativar no Windows:
# venv\Scripts\activate
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 4. Configurar Variáveis de Ambiente

Crie um arquivo .env na raiz do projeto (.) copiando o .env.example e preenchendo com suas chaves e URLs (substitua pelos seus valores reais). Estas variáveis são cruciais para a comunicação com a API da OpenAI.

```ini
OPENAI_API_KEY="sua_chave_api_openai_aqui"
OPENAI_API_BASE="sua_URL_de_proxy_DTA"
```

## Como Executar o TCRM Copilot

Após configurar o ambiente, você pode iniciar a aplicação Streamlit:

```bash
streamlit run src/front_end/app.py
```

Isso abrirá a interface do TCRM Copilot em seu navegador padrão, geralmente em 'http://localhost:8501'.

## Entendendo a Pipeline do TCRM Copilot

### Processo de Ingestão de Dados (RAG)

O coração da nossa base de conhecimento reside no módulo `src/data_ingestion/incremental_ingestor.py`. Este módulo é responsável por:

* **Carregamento de Documentos:** Suporta arquivos PDFs, DOCX e TXT.

* **Extração de Metadados Inteligente:** Automaticamente extrai `client_name`, `doc_type` (e.g., MIT041) e `project_code` a partir de heurísticas no nome do arquivo e/ou dados fornecidos na interface.

* **Chunking Adaptativo:** Divide documentos longos em pedaços menores (chunks) usando `RecursiveCharacterTextSplitter` para processamento eficiente e contextualizado.

* **Geração de Embeddings:** Converte cada chunk em um vetor numérico de alta dimensionalidade usando `OpenAIEmbeddings`.

* **Armazenamento no ChromaDB:** Persiste os chunks e seus embeddings em nossa base de vetores, permitindo buscas semânticas.

* **Ingestão Incremental e Resiliente:** Verifica a duplicidade de documentos (via hash) para evitar re-ingestão e trata `RateLimitError` da API da OpenAI com retries e backoff exponencial, garantindo robustez e eficiência.

Este processo garante que o Copilot sempre tenha acesso às informações mais atualizadas e relevantes, com uma base de conhecimento em constante evolução.

### Orquestração do Agente (LangGraph)

O `src/core/copilot_agent.py` define a lógica de orquestração do agente usando LangGraph. Ele gerencia o fluxo da interação, desde a recuperação de contexto relevante no ChromaDB (`retrieve_context`), passando pela geração da resposta com o LLM (`generate_response`), até a formatação das citações (`format_citation`), garantindo que o agente se comporte de maneira coerente e útil em cada etapa da conversação.

### Interface do Usuário (Streamlit)

O `src/front_end/app.py` provê a interface interativa do Copilot. É através dela que os usuários podem fazer upload de documentos, interagir com o agente via chat e visualizar as respostas e suas fontes de forma clara e intuitiva.

### Módulos Legados (`src/data_ingestion/legacy_components/`)

O diretório `legacy_components` abriga as versões iniciais dos módulos de extração e construção da base de conhecimento (`document_parser.py` e `knowledge_base_builder.py`). Estes arquivos foram cruciais na fase inicial de prototipagem, mas suas funcionalidades foram posteriormente absorvidas, otimizadas e consolidadas no `incremental_ingestor.py`. Eles são mantidos para fins de referência histórica e para documentar a evolução das estratégias de engenharia do projeto. Representam um valioso registro de aprendizado e insights sobre as escolhas de design e refatorações realizadas.

## Roadmap Futuro (Visão Visionária)

O TCRM Copilot está apenas começando. As futuras iterações e melhorias incluem:

* **Integração Aprofundada com TCRM API:** Ampliar a capacidade do agente de buscar e interpretar dados em tempo real diretamente do TOTVS CRM, enriquecendo suas respostas com informações operacionais.

* **Geração Estruturada de Documentos:** Evoluir a funcionalidade de handover para montar documentos completos e formatados automaticamente, com base nas informações do projeto.

* **Exploração do DTA da TOTVS:** Avaliar a integração com o DTA para orquestração de fluxos de dados externos e automação mais ampla de processos.

* **Conversas Multi-turn Avançadas:** Aprimorar a capacidade do agente de manter o contexto em longas conversas e responder a perguntas de acompanhamento de forma mais natural.

* **Feedback Loops e Melhoria Contínua:** Implementar mecanismos para coletar feedback do usuário sobre a qualidade das respostas e usar essas informações para refinar o desempenho do agente e a base de conhecimento.

* **Funcionalidades Inteligentes Adicionais:** Explorar a inclusão de sumarização automática de longos documentos, identificação de riscos e oportunidades em documentos de projeto, sugestão proativa de próximos passos baseada no contexto atual do projeto, e muito mais.