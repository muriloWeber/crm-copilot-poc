"""
Módulo responsável pela ingestão incremental de documentos (PDF, DOCX, TXT)
para a base de dados vetorial ChromaDB.

Inclui funcionalidades para carregamento de arquivos, chunking de texto,
geração de embeddings, persistência no ChromaDB e tratamento de metadados,
além de lidar com a verificação de duplicidade e rate limits da API da OpenAI.
"""

import os
import hashlib
import logging
import time
import datetime
import re
from typing import List, Dict, Union, Any, Optional

import openai # Para capturar openai.RateLimitError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import httpx # Para configurar o cliente HTTP com verificação SSL

# --- Configuração de Logging ---
# Configura o logger para este módulo. O nível INFO é adequado para operações gerais,
# mas pode ser ajustado para DEBUG para uma depuração mais granular durante o desenvolvimento.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes de Configuração ---
# O diretório do ChromaDB é definido de forma absoluta para garantir persistência
# e evitar problemas de caminho relativo em diferentes ambientes de execução.
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vector_db', 'chroma_db'))
CHROMA_COLLECTION_NAME = 'tcrm_copilot_kb'
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

# --- Funções de Apoio ---

def get_file_hash(file_path: str) -> str:
    """
    Calcula o hash SHA256 de um arquivo para fins de verificação de integridade
    e detecção de duplicidade.

    Args:
        file_path (str): O caminho completo para o arquivo.

    Returns:
        str: O hash SHA256 hexadecimal do arquivo.
    """
    hasher = hashlib.sha256()
    # Abre o arquivo em modo binário para leitura em chunks, evitando carregar
    # arquivos muito grandes na memória de uma vez.
    with open(file_path, 'rb') as f:
        # Lê o arquivo em blocos de 64KB
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_document_loader(file_path: str) -> Optional[Union[PyPDFLoader, Docx2txtLoader, TextLoader]]:
    """
    Retorna o loader apropriado da LangChain com base na extensão do arquivo.
    Suporta PDF, DOCX e TXT.

    Args:
        file_path (str): O caminho completo para o arquivo.

    Returns:
        Optional[Union[PyPDFLoader, Docx2txtLoader, TextLoader]]: Uma instância do loader
                                                                  correspondente, ou None
                                                                  se a extensão não for suportada.
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        logger.debug(f"Utilizando PyPDFLoader para {file_path}")
        return PyPDFLoader(file_path)
    elif file_extension == '.docx':
        logger.debug(f"Utilizando Docx2txtLoader para {file_path}")
        return Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        logger.debug(f"Utilizando TextLoader para {file_path}")
        return TextLoader(file_path)
    else:
        logger.warning(f"Extensão de arquivo não suportada para ingestão: {file_extension}. Arquivo: {file_path}")
        return None

# Variável global para armazenar a instância do modelo de embeddings (singleton pattern).
_embeddings_model: Optional[OpenAIEmbeddings] = None

def get_embeddings_model() -> OpenAIEmbeddings:
    """
    Inicializa e retorna uma instância única do modelo de embeddings da OpenAI.
    Utiliza um padrão Singleton para garantir que o modelo seja instanciado apenas uma vez.

    Raises:
        ValueError: Se a variável de ambiente OPENAI_API_KEY não estiver configurada.

    Returns:
        OpenAIEmbeddings: Uma instância configurada do modelo de embeddings.
    """
    global _embeddings_model
    if _embeddings_model is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")

        if not openai_api_key:
            logger.error("Variável de ambiente OPENAI_API_KEY não configurada para embeddings. O modelo não será inicializado.")
            raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada para embeddings.")
        
        # Configura o cliente HTTP para ignorar a verificação SSL.
        # ATENÇÃO: 'verify=False' desabilita a verificação de certificados SSL, o que pode
        # ser um RISCO DE SEGURANÇA em ambientes de produção. É usado aqui para PoC/testes
        # com proxies internos que podem não ter certificados confiáveis.
        # Para produção, configure a verificação SSL corretamente ou use certificados CA.
        custom_http_client = httpx.Client(verify=False)

        _embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_key=openai_api_key,
            base_url=openai_api_base if openai_api_base else None,
            http_client=custom_http_client
        )
        logger.info(f"Modelo de embeddings '{EMBEDDING_MODEL_NAME}' inicializado com sucesso.")
    return _embeddings_model

# Variável global para armazenar a instância do ChromaDB (singleton pattern).
_chroma_instance: Optional[Chroma] = None

def get_chroma_instance() -> Chroma:
    """
    Retorna uma instância única do ChromaDB.
    Se o diretório de persistência não existir, ele será criado.
    Utiliza um padrão Singleton para garantir uma única conexão com o DB.

    Returns:
        Chroma: Uma instância configurada do cliente ChromaDB.
    """
    global _chroma_instance
    if _chroma_instance is None:
        embeddings = get_embeddings_model() # Garante que o modelo de embeddings esteja inicializado
        
        if not os.path.exists(CHROMA_DB_DIRECTORY):
            logger.info(f"Diretório ChromaDB não encontrado em {CHROMA_DB_DIRECTORY}. Criando...")
            os.makedirs(CHROMA_DB_DIRECTORY, exist_ok=True)
            # Ao criar o diretório, inicializa uma nova coleção.
            _chroma_instance = Chroma(
                persist_directory=CHROMA_DB_DIRECTORY,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME
            )
            logger.info(f"Nova instância ChromaDB criada em {CHROMA_DB_DIRECTORY} com coleção '{CHROMA_COLLECTION_NAME}'.")
        else:
            logger.info(f"ChromaDB carregado do diretório: {CHROMA_DB_DIRECTORY} para a coleção '{CHROMA_COLLECTION_NAME}'.")
            # Carrega a coleção existente do diretório.
            _chroma_instance = Chroma(
                persist_directory=CHROMA_DB_DIRECTORY,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME
            )
    return _chroma_instance

# --- Função Principal de Ingestão Incremental ---

def add_document_to_vector_store(
    file_path: str,
    vector_store: Chroma,
    text_splitter: RecursiveCharacterTextSplitter,
    detected_metadata: Dict[str, str], # Metadados passados pelo Streamlit (podem ser None)
    max_retries: int = 5
) -> None:
    """
    Adiciona um documento (PDF/DOCX/TXT) ao ChromaDB de forma incremental.
    A função verifica a duplicidade do documento pelo hash e lida com
    possíveis Rate Limits da API da OpenAI, re-tentando a operação.

    Args:
        file_path (str): O caminho completo para o arquivo a ser ingerido.
        vector_store (Chroma): A instância do ChromaDB onde o documento será adicionado.
        text_splitter (RecursiveCharacterTextSplitter): O splitador de texto configurado
                                                        para dividir o conteúdo do documento em chunks.
        detected_metadata (Dict[str, str]): Um dicionário de metadados extraídos ou fornecidos
                                            pelo usuário (e.g., via UI), que podem incluir
                                            'client_name', 'doc_type', 'project_code'.
        max_retries (int): O número máximo de tentativas em caso de RateLimitError.

    Raises:
        Exception: Em caso de falha na ingestão do documento após todas as tentativas
                   ou outros erros inesperados.
    """
    file_name = os.path.basename(file_path)
    document_hash = get_file_hash(file_path)
    logger.info(f"Iniciando processamento do arquivo: '{file_name}' (hash: {document_hash[:8]})")

    try:
        base_name_no_ext = os.path.splitext(file_name)[0]

        # --- Lógica de extração e fallback para metadados ---
        # Prioriza metadados fornecidos via 'detected_metadata' (e.g., UI),
        # depois tenta extrair do nome do arquivo, e por último um valor padrão 'UNKNOWN'.
        
        # 1. client_name
        client_name_from_file_match = re.match(r'\[(.*?)\]', base_name_no_ext)
        client_name_from_file = client_name_from_file_match.group(1).strip() if client_name_from_file_match else None
        final_client_name = (detected_metadata.get('client_name') or client_name_from_file or 'UNKNOWN CLIENT').upper()
        logger.debug(f"[METADATA] 'client_name' para '{file_name}': {final_client_name}")

        # 2. doc_type (e.g., MIT041)
        doc_type_from_file_match = re.search(r'(MIT\d{3})', base_name_no_ext, re.IGNORECASE)
        doc_type_from_file = doc_type_from_file_match.group(1).strip().upper() if doc_type_from_file_match else None
        final_doc_type = (detected_metadata.get('doc_type') or doc_type_from_file or 'UNKNOWN DOC TYPE').upper()
        logger.debug(f"[METADATA] 'doc_type' para '{file_name}': {final_doc_type}")

        # 3. project_code (e.g., D000071597001)
        project_code_from_file_match = re.search(r'(D\d{9,15})', base_name_no_ext, re.IGNORECASE)
        project_code_from_file = project_code_from_file_match.group(1).strip().upper() if project_code_from_file_match else None
        final_project_code = (detected_metadata.get('project_code') or project_code_from_file or 'UNKNOWN PROJECT').upper()
        logger.debug(f"[METADATA] 'project_code' para '{file_name}': {final_project_code}")

        # --- Verificação de Duplicidade ---
        # Consulta o ChromaDB para verificar se um documento com o mesmo hash já existe.
        existing_docs = vector_store._collection.get(
            where={"document_hash": document_hash},
            include=[] # Não precisamos do conteúdo, apenas dos IDs e metadados para verificar a existência
        )
        if existing_docs and existing_docs['ids']:
            logger.info(f"Documento '{file_name}' (hash: {document_hash[:8]}) já existe no ChromaDB. Ingestão ignorada.")
            return

        # --- Carregamento do Documento ---
        loader = get_document_loader(file_path)
        if not loader:
            logger.error(f"Não foi possível obter um loader para o arquivo '{file_name}'. Ingestão falhou.")
            return # Aviso já logado em get_document_loader

        pages: List[Document] = loader.load()
        logger.info(f"Carregadas {len(pages)} páginas/seções de '{file_name}' usando {type(loader).__name__}.")

        if not pages:
            logger.warning(f"Nenhum conteúdo extraído de '{file_name}'. Ingestão ignorada.")
            return

        # --- Adição de Metadados Padrão às Páginas ---
        # Metadados são adicionados aqui e serão propagados para os chunks.
        for i, page in enumerate(pages):
            page.metadata["source"] = file_name
            page.metadata["document_hash"] = document_hash
            page.metadata["original_filename"] = file_name # Mantém o nome original completo do arquivo
            
            # Normaliza o 'page_number'. Para PDFs geralmente já vem, para outros adicionamos.
            page_number = page.metadata.get("page", page.metadata.get("page_number", i + 1))
            page.metadata["page_number"] = page_number
            
            # Adiciona os metadados de cliente, tipo e projeto
            page.metadata["client_name"] = final_client_name
            page.metadata["doc_type"] = final_doc_type
            page.metadata["project_code"] = final_project_code

            logger.debug(f"[METADATA_PAGE] Página {page.metadata['page_number']} de '{file_name}' com metadados: {page.metadata}")

        # --- Chunking do Documento ---
        chunks: List[Document] = text_splitter.split_documents(pages)
        # Adicionar um 'chunk_index' a cada chunk para referência e debugging
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            # Garante que os metadados específicos estejam em CADA CHUNK.
            # Isso é uma redundância segura para o ChromaDB, pois os metadados
            # das 'pages' são copiados para os 'chunks' durante o split.
            # Contudo, verificar e garantir é uma boa prática.
            chunk.metadata["client_name"] = final_client_name
            chunk.metadata["doc_type"] = final_doc_type
            chunk.metadata["project_code"] = final_project_code
            logger.debug(f"[METADATA_CHUNK] Chunk {i} de '{file_name}' (Pág {chunk.metadata.get('page_number', 'N/A')}) com metadados: {chunk.metadata}")

        logger.info(f"Documento chunkizado em {len(chunks)} partes para '{file_name}'.")

        if not chunks:
            logger.warning(f"Nenhum chunk gerado para '{file_name}' após o split. Ingestão ignorada.")
            return

        # --- Geração de Embeddings e Adição ao ChromaDB (com re-tentativa) ---
        logger.info(f"Gerando embeddings e adicionando {len(chunks)} chunks de '{file_name}' ao ChromaDB...")
        
        attempt = 0
        while attempt < max_retries:
            try:
                vector_store.add_documents(chunks)
                logger.info(f"Documento '{file_name}' (hash: {document_hash[:8]}) adicionado com sucesso ao ChromaDB.")
                return # Sai da função após ingestão bem-sucedida
            except openai.RateLimitError as e:
                attempt += 1
                logger.warning(f"Rate limit excedido ao gerar embeddings para '{file_name}' (Tentativa {attempt}/{max_retries}). Erro: {e}")
                
                wait_time = 60 # Tempo de espera padrão (1 minuto)
                
                # Tenta extrair o tempo de reset da mensagem de erro para uma espera mais precisa
                reset_time_match = re.search(r'Limit resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC', str(e))
                if reset_time_match:
                    reset_dt_utc = datetime.datetime.strptime(reset_time_match.group(1), '%Y-%m-%d %H:%M:%S')
                    current_dt_utc = datetime.datetime.utcnow()
                    
                    # Calcula a diferença em segundos, adiciona um buffer de 5 segundos e garante que seja positivo
                    calculated_wait_time = (reset_dt_utc - current_dt_utc).total_seconds() + 5 
                    if calculated_wait_time > 0:
                        wait_time = calculated_wait_time
                        logger.info(f"Aguardando até {reset_dt_utc} UTC (aproximadamente {wait_time:.0f} segundos) antes de re-tentar...")
                    else:
                        logger.info(f"Tempo de reset já passou ou é insignificante ({calculated_wait_time:.0f}s). Aguardando o tempo padrão de {wait_time} segundos.")
                else:
                    logger.info(f"Não foi possível extrair o tempo de reset da mensagem de erro. Aguardando o tempo padrão de {wait_time} segundos.")
                
                time.sleep(wait_time)
                
            except Exception as e: # Captura outros erros inesperados durante add_documents
                logger.error(f"Erro inesperado ao adicionar chunks de '{file_name}' ao ChromaDB: {e}", exc_info=True)
                raise # Re-lança o erro para a camada superior lidar

        # Se o loop terminar sem 'break', significa que o número máximo de re-tentativas foi atingido.
        logger.error(f"Falha persistente ao adicionar documento '{file_name}' após {max_retries} tentativas devido a RateLimitError.")
        raise Exception(f"Falha ao adicionar documento '{file_name}' após {max_retries} tentativas devido a RateLimitError.")

    except Exception as e:
        logger.error(f"Falha crítica no processamento do arquivo '{file_name}': {e}", exc_info=True)
        raise # Re-lança a exceção original para a UI ou outro manipulador de erro.

def query_chroma_metadata(client_name: Optional[str] = None, project_code: Optional[str] = None, doc_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Consulta o ChromaDB para listar IDs e metadados de documentos, aplicando filtros opcionais.
    Útil para depuração e verificação do conteúdo da base de conhecimento.

    Args:
        client_name (Optional[str]): Nome do cliente para filtrar.
        project_code (Optional[str]): Código do projeto para filtrar.
        doc_type (Optional[str]): Tipo de documento para filtrar (e.g., MIT041).

    Returns:
        Dict[str, Any]: Um dicionário contendo os IDs e metadados dos documentos encontrados.
    """
    vector_store = get_chroma_instance()
    
    where_clause = {}
    if client_name:
        where_clause["client_name"] = client_name.upper()
    if project_code:
        where_clause["project_code"] = project_code.upper()
    if doc_type:
        where_clause["doc_type"] = doc_type.upper()

    logger.info(f"Consultando ChromaDB com filtro: {where_clause if where_clause else 'Nenhum filtro'}")
    
    # Se não houver filtro, limita o número de resultados para evitar sobrecarga em bases grandes.
    # Inclui apenas os metadados para a consulta.
    if not where_clause:
        results = vector_store._collection.get(limit=100, include=['metadatas'])
    else:
        results = vector_store._collection.get(where=where_clause, include=['metadatas'])

    if results and results.get('ids'):
        logger.info(f"Encontrados {len(results['ids'])} resultados.")
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            logger.info(f"  ID: {doc_id[:8]}... Metadados: {metadata}")
    else:
        logger.info("Nenhum documento encontrado para a consulta especificada.")
    
    return results

def get_known_entities() -> Dict[str, List[str]]:
    """
    Consulta o ChromaDB para extrair e retornar uma lista de todos os
    nomes de clientes, códigos de projeto e tipos de documento únicos
    presentes nos metadados dos documentos ingeridos.
    Essas entidades são usadas para o pré-filtragem no agente Copilot.

    Advertência: Para coleções ChromaDB muito grandes (milhões de chunks),
    esta operação pode ser custosa. Em cenários de alta escala,
    considerar um mecanismo de cache ou persistência dedicado para essas entidades.

    Returns:
        Dict[str, List[str]]: Um dicionário com listas ordenadas de
                              'client_names', 'project_codes' e 'doc_types' únicos.
    """
    vector_store = get_chroma_instance()
    
    try:
        # Recupera todos os metadados da coleção.
        # Inclui apenas 'metadatas' para otimizar o uso de memória.
        results = vector_store._collection.get(include=['metadatas'])
    except Exception as e:
        logger.error(f"Erro ao obter metadados do ChromaDB para extrair entidades conhecidas: {e}", exc_info=True)
        return {"client_names": [], "project_codes": [], "doc_types": []}

    client_names = set()
    project_codes = set()
    doc_types = set()

    if results and results.get('metadatas'):
        for metadata in results['metadatas']:
            # Adiciona entidades se existirem e não forem vazias, convertendo para maiúsculas para padronização.
            if 'client_name' in metadata and metadata['client_name']:
                client_names.add(metadata['client_name'].upper())
            if 'project_code' in metadata and metadata['project_code']:
                project_codes.add(metadata['project_code'].upper())
            if 'doc_type' in metadata and metadata['doc_type']:
                doc_types.add(metadata['doc_type'].upper())

    logger.debug(f"Entidades Conhecidas Extraídas - Clientes: {list(client_names)}, Projetos: {list(project_codes)}, Tipos Doc: {list(doc_types)}")
    return {
        "client_names": sorted(list(client_names)), # Retorna listas ordenadas alfabeticamente
        "project_codes": sorted(list(project_codes)),
        "doc_types": sorted(list(doc_types))
    }

# --- Bloco de Execução Principal (para testes locais) ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Carrega as variáveis de ambiente do arquivo .env

    # Aumenta o nível de log para DEBUG para depuração detalhada durante testes locais.
    logger.setLevel(logging.DEBUG) 

    logger.info("Iniciando testes do módulo de ingestão incremental...")

    # --- Teste de Ingestão de Documentos (Opcional) ---
    # Descomente o bloco abaixo para realizar a ingestão de documentos de teste.
    # Certifique-se de que os arquivos existam nos caminhos especificados.
    
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    # chroma_instance = get_chroma_instance()
    # sample_docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'documents'))
    # files_to_ingest = [
    #     os.path.join(sample_docs_dir, "[INOVA] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 02-01-2025.docx"),
    #     os.path.join(sample_docs_dir, "[KION] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041.docx"),
    #     os.path.join(sample_docs_dir, "[MARSON] Escopo de Customização de Integração - TOTVS iPaaS - MIT041 - V1.0 10-04-2024.docx"),
    #     os.path.join(sample_docs_dir, "[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx"),
    #     os.path.join(sample_docs_dir, "Roteiro do Projeto_ TCRM Copilot - Prova de Conceito (PoC).pdf"),
    # ]
    # for f_path in files_to_ingest:
    #     if os.path.exists(f_path):
    #         try:
    #             # Metadados para ingestão (vazio para extração do nome do arquivo, ou preencher manualmente)
    #             metadata_for_ingestion = {} 
    #             logger.info(f"Iniciando ingestão de {os.path.basename(f_path)}...")
    #             add_document_to_vector_store(f_path, chroma_instance, text_splitter, metadata_for_ingestion)
    #             time.sleep(1) # Pequena pausa para evitar sobrecarregar o sistema/API
    #         except Exception as e:
    #             logger.error(f"Falha ao ingerir '{os.path.basename(f_path)}': {e}", exc_info=True)
    #     else:
    #         logger.warning(f"Arquivo de teste não encontrado: {f_path}")
    # logger.info("Processo de ingestão incremental de documentos de teste concluído.")

    # --- Testes de Consulta de Metadados ---
    logger.setLevel(logging.INFO) # Volta para INFO para os testes de consulta, menos verboso

    logger.info("\n--- Consultando TODAS as entidades conhecidas (clientes, projetos, tipos de doc) ---")
    known_entities = get_known_entities()
    logger.info(f"Entidades Conhecidas: {known_entities}")

    # Consultas dinâmicas baseadas nas entidades conhecidas
    if known_entities.get("client_names"):
        for client_name in known_entities["client_names"]:
            logger.info(f"\n--- Consultando ChromaDB para documentos do cliente '{client_name}' ---")
            query_chroma_metadata(client_name=client_name)
    else:
        logger.info("\n--- Nenhuma entidade de cliente conhecida para consulta dinâmica. ---")

    if known_entities.get("doc_types"):
        for doc_type in known_entities["doc_types"]:
            logger.info(f"\n--- Consultando ChromaDB para documentos do tipo '{doc_type}' ---")
            query_chroma_metadata(doc_type=doc_type)
    else:
        logger.info("\n--- Nenhuma entidade de tipo de documento conhecida para consulta dinâmica. ---")
    
    if known_entities.get("project_codes"):
        for project_code in known_entities["project_codes"]:
            logger.info(f"\n--- Consultando ChromaDB para documentos do projeto '{project_code}' ---")
            query_chroma_metadata(project_code=project_code)
    else:
        logger.info("\n--- Nenhuma entidade de código de projeto conhecida para consulta dinâmica. ---")

    logger.info("\n--- Consultando ChromaDB para TODOS os documentos (limitado aos 100 primeiros) ---")
    query_chroma_metadata() # Sem filtros, pega os primeiros 100

    logger.info("Testes do módulo de ingestão incremental concluídos.")