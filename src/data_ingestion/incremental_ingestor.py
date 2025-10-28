# src/data_ingestion/incremental_ingestor.py

import os
import hashlib
import logging
import time
import datetime
import re
from typing import List, Dict, Union, Any, Optional

import openai # Importar a biblioteca openai para pegar a exceção específica
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import httpx # Para o cliente HTTP customizado com verify=False

# Configura o logger para a ingestão
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes de Configuração ---
# O diretório do ChromaDB deve ser absoluto e persistente
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vector_db', 'chroma_db'))
CHROMA_COLLECTION_NAME = 'tcrm_copilot_kb'
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

# --- Funções de Apoio ---

def get_file_hash(file_path: str) -> str:
    """Calcula o hash SHA256 de um arquivo."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_document_loader(file_path: str) -> Optional[Union[PyPDFLoader, Docx2txtLoader, TextLoader]]:
    """Retorna o loader apropriado para o tipo de arquivo."""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.pdf':
        return PyPDFLoader(file_path)
    elif file_extension == '.docx':
        return Docx2txtLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path)
    else:
        logger.warning(f"Extensão de arquivo não suportada: {file_extension}")
        return None

def get_embeddings_model() -> OpenAIEmbeddings:
    """Inicializa e retorna o modelo de embeddings da OpenAI."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")

    if not openai_api_key:
        raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada para embeddings.")
    
    # Configura o cliente HTTP para ignorar a verificação SSL
    custom_http_client = httpx.Client(verify=False)

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        http_client=custom_http_client
    )

_embeddings_model: Optional[OpenAIEmbeddings] = None
def get_chroma_instance() -> Chroma:
    """
    Retorna uma instância do ChromaDB.
    Se o diretório não existir, ele será criado.
    """
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = get_embeddings_model()

    if not os.path.exists(CHROMA_DB_DIRECTORY):
        logger.info(f"Diretório ChromaDB não encontrado em {CHROMA_DB_DIRECTORY}. Criando...")
        os.makedirs(CHROMA_DB_DIRECTORY, exist_ok=True)
        # O ChromaDB será inicializado como vazio e populado na primeira ingestão
        vector_store = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=_embeddings_model,
            collection_name=CHROMA_COLLECTION_NAME
        )
        logger.info(f"Nova instância ChromaDB criada em {CHROMA_DB_DIRECTORY} com coleção '{CHROMA_COLLECTION_NAME}'.")
        return vector_store
    else:
        logger.info(f"ChromaDB carregado do diretório: {CHROMA_DB_DIRECTORY}")
        return Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=_embeddings_model,
            collection_name=CHROMA_COLLECTION_NAME
        )

# --- Função Principal de Ingestão Incremental ---

def add_document_to_vector_store(
    file_path: str,
    vector_store: Chroma,
    text_splitter: RecursiveCharacterTextSplitter,
    detected_metadata: Dict[str, str],
    max_retries: int = 5
):
    """
    Adiciona um documento (PDF/DOCX/TXT) ao ChromaDB de forma incremental,
    evitando duplicações e tratando rate limits da API.
    """
    file_name = os.path.basename(file_path)
    logger.info(f"Processando arquivo: {file_name} (hash: {get_file_hash(file_path)[:8]})")

    try:
        # Extrair metadados do nome do arquivo
        base_name = os.path.splitext(file_name)[0]

        # --- Processamento de client_name ---
        client_name_from_file = None
        client_match = re.match(r'\[(.*?)\]', base_name)
        if client_match:
            client_name_from_file = client_match.group(1) # Valor capturado do regex

        client_name_from_metadata = detected_metadata.get('client_name') # Pode ser None

        # Prioridade: 1. Do arquivo, 2. Do detected_metadata, 3. Default
        # Usamos 'or' para pegar o primeiro valor 'truthy' (não-None, não-vazio)
        final_client_name = (client_name_from_file or 
                             client_name_from_metadata or 
                             'UNKNOWN CLIENT')
        client_name = final_client_name.upper()

        # --- Processamento de doc_type ---
        doc_type_from_file = None
        doc_type_match = re.search(r'(MIT\d{3})', base_name, re.IGNORECASE)
        if doc_type_match:
            doc_type_from_file = doc_type_match.group(1)

        doc_type_from_metadata = detected_metadata.get('doc_type')

        final_doc_type = (doc_type_from_file or
                          doc_type_from_metadata or
                          'UNKNOWN DOC TYPE')
        doc_type = final_doc_type.upper()
        
        # --- Processamento de project_code ---
        project_code_from_file = None
        project_code_match = re.search(r'(D\d{9,15})', base_name, re.IGNORECASE)
        if project_code_match:
            project_code_from_file = project_code_match.group(1)

        project_code_from_metadata = detected_metadata.get('project_code')

        final_project_code = (project_code_from_file or
                              project_code_from_metadata or
                              'UNKNOWN PROJECT')
        project_code = final_project_code.upper()


        # Hash do documento para verificação de duplicidade
        document_hash = get_file_hash(file_path)

        # Verificar se o documento já existe no vetor store
        existing_docs = vector_store._collection.get(
            where={"document_hash": document_hash},
            include=['metadatas']
        )
        if existing_docs and existing_docs['ids']:
            logger.info(f"Documento '{file_name}' (hash: {document_hash[:8]}) já existe no ChromaDB. Ignorando ingestão.")
            return

        loader = get_document_loader(file_path)
        if not loader:
            return

        pages = loader.load()
        logger.info(f"Carregados {len(pages)} páginas/seções do documento.")

        # Adicionar metadados adicionais aos documentos antes de chunkar
        for i, page in enumerate(pages):
            page.metadata["source"] = file_name
            page.metadata["document_hash"] = document_hash
            page.metadata["original_filename"] = file_name # Nome original completo do arquivo
            page.metadata["page_number"] = i + 1 # Adiciona o número da página (começando em 1)
            page.metadata["client_name"] = client_name
            page.metadata["doc_type"] = doc_type
            page.metadata["project_code"] = project_code


        chunks = text_splitter.split_documents(pages)
        # Adicionar um chunk_index a cada chunk para referência
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["client_name"] = client_name # Assegurar metadados no chunk também
            chunk.metadata["doc_type"] = doc_type
            chunk.metadata["project_code"] = project_code

        logger.info(f"Documento chunkizado em {len(chunks)} partes.")

        if not chunks:
            logger.warning(f"Nenhum conteúdo extraído ou chunk gerado para {file_name}. Ignorando.")
            return

        logger.info(f"Gerando embeddings e adicionando {len(chunks)} chunks ao ChromaDB...")
        
        attempt = 0
        while attempt < max_retries:
            try:
                vector_store.add_documents(chunks)
                logger.info(f"Documento '{file_name}' adicionado com sucesso ao ChromaDB.")
                break # Sai do loop se o upload for bem-sucedido
            except openai.RateLimitError as e:
                attempt += 1
                logger.warning(f"Limite de requisições excedido ao gerar embeddings para '{file_name}' (Tentativa {attempt}/{max_retries}). Erro: {e}")
                
                wait_time = 60 # Tempo de espera padrão (1 minuto)
                
                # Tenta extrair o tempo de reset do erro para uma espera mais precisa
                reset_time_match = re.search(r'Limit resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC', str(e))
                if reset_time_match:
                    reset_time_str = reset_time_match.group(1)
                    try:
                        # Parse a string de data/hora para um objeto datetime
                        reset_dt_utc = datetime.datetime.strptime(reset_time_str, '%Y-%m-%d %H:%M:%S')
                        current_dt_utc = datetime.datetime.utcnow()
                        
                        # Calcula a diferença em segundos, adiciona um buffer e garante que seja positivo
                        calculated_wait_time = (reset_dt_utc - current_dt_utc).total_seconds() + 5 
                        if calculated_wait_time > 0:
                            wait_time = calculated_wait_time
                            logger.info(f"Aguardando até {reset_time_str} UTC (aproximadamente {wait_time:.0f} segundos) antes de tentar novamente...")
                        else: # Se o tempo de reset já passou ou é insignificante
                            logger.info(f"O tempo de reset já passou ou está muito próximo ({calculated_wait_time:.0f}s). Aguardando o tempo padrão de {wait_time} segundos.")
                    except ValueError:
                        logger.warning(f"Não foi possível analisar o tempo de reset '{reset_time_str}'. Aguardando o tempo padrão de {wait_time} segundos.")
                else: # Se não encontrou a string de reset
                    logger.info(f"Não foi possível encontrar o tempo de reset na mensagem de erro. Aguardando o tempo padrão de {wait_time} segundos.")
                
                time.sleep(wait_time)
                
            except Exception as e: # Captura outros erros inesperados durante add_documents
                logger.error(f"Erro inesperado ao adicionar chunks de '{file_name}' ao ChromaDB: {e}", exc_info=True)
                raise # Relança o erro para que a UI possa lidar com ele
        else: # Este bloco é executado se o loop 'while' terminar sem um 'break' (ou seja, max_retries atingido)
            logger.error(f"Falha ao adicionar documento '{file_name}' após {max_retries} tentativas devido a RateLimitError.")
            raise Exception(f"Falha ao adicionar documento '{file_name}' após {max_retries} tentativas devido a RateLimitError.")

    except Exception as e:
        logger.error(f"Erro ao processar o arquivo '{file_name}': {e}", exc_info=True)
        raise # Relança a exceção original para a UI

# --- Exemplo de Uso (para teste standalone) ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Carrega as variáveis de ambiente

    logger.setLevel(logging.DEBUG) # Aumenta o nível de log para debug em teste local

    # Exemplo de configuração do text_splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # Obter instância do ChromaDB
    chroma_instance = get_chroma_instance()

    # Caminho para o diretório de arquivos de exemplo (ajuste conforme sua estrutura)
    sample_docs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'documents'))
    
    # Lista de arquivos para ingestão
    files_to_ingest = [
        os.path.join(sample_docs_dir, "[INOVA] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 02-01-2025.docx"),
        os.path.join(sample_docs_dir, "[KION] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041.docx"),
        os.path.join(sample_docs_dir, "[MARSON] Escopo de Customização de Integração - TOTVS iPaaS - MIT041 - V1.0 10-04-2024.docx"),
        os.path.join(sample_docs_dir, "[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx"),
        # Adicione outros arquivos de teste aqui
    ]

    for f_path in files_to_ingest:
        if os.path.exists(f_path):
            try:
                # detected_metadata pode ser extraído dinamicamente ou passado como um dicionário vazio se não for estritamente necessário para o seu caso de uso
                # Aqui, para exemplo, extraímos do nome do arquivo
                base_name_f = os.path.splitext(os.path.basename(f_path))[0]
                client_match_f = re.match(r'\[(.*?)\]', base_name_f)
                client_name_f = client_match_f.group(1).upper() if client_match_f else 'UNKNOWN CLIENT'
                doc_type_match_f = re.search(r'(MIT\d{3})', base_name_f, re.IGNORECASE)
                doc_type_f = doc_type_match_f.group(1).upper() if doc_type_match_f else 'UNKNOWN DOC TYPE'
                project_code_match_f = re.search(r'(D\d{9,15})', base_name_f, re.IGNORECASE)
                project_code_f = project_code_match_f.group(1).upper() if project_code_match_f else 'UNKNOWN PROJECT'

                metadata_for_ingestion = {
                    "client_name": client_name_f,
                    "doc_type": doc_type_f,
                    "project_code": project_code_f
                }
                
                add_document_to_vector_store(f_path, chroma_instance, text_splitter, metadata_for_ingestion)
                time.sleep(1) # Pequena pausa para evitar rate limit em cascata
            except Exception as e:
                logger.error(f"Falha ao ingerir {os.path.basename(f_path)}: {e}")
        else:
            logger.warning(f"Arquivo não encontrado: {f_path}")

    logger.info("Processo de ingestão incremental concluído.")

    #Para verificar o conteúdo do ChromaDB após a ingestão
    # if chroma_instance._collection.count() > 0:
    #     logger.info(f"Total de documentos no ChromaDB: {chroma_instance._collection.count()}")
    #     # Exemplo de recuperação de um item (pode ser pesado para muitos itens)
    #     # all_ids = chroma_instance._collection.get(limit=1, include=['metadatas'])
    #     # logger.debug(f"Exemplo de metadata: {all_ids['metadatas'][0]}")
