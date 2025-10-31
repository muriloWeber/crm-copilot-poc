import os
import hashlib
import logging
import time
import datetime
import re
from typing import List, Dict, Union, Any, Optional

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import httpx

# Configura o logger para a ingestão
# OBS: O nível de log pode ser ajustado para DEBUG se necessário para depuração detalhada
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

_embeddings_model: Optional[OpenAIEmbeddings] = None
def get_embeddings_model() -> OpenAIEmbeddings:
    """Inicializa e retorna o modelo de embeddings da OpenAI."""
    global _embeddings_model # Garante que a instância seja única e persistente
    if _embeddings_model is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_api_base = os.getenv("OPENAI_API_BASE")

        if not openai_api_key:
            raise ValueError("Variável de ambiente OPENAI_API_KEY não configurada para embeddings.")
        
        # Configura o cliente HTTP para ignorar a verificação SSL
        custom_http_client = httpx.Client(verify=False)

        _embeddings_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            openai_api_key=openai_api_key,
            base_url=openai_api_base if openai_api_base else None,
            http_client=custom_http_client
        )
    return _embeddings_model

_chroma_instance: Optional[Chroma] = None
def get_chroma_instance() -> Chroma:
    """
    Retorna uma instância do ChromaDB.
    Se o diretório não existir, ele será criado.
    """
    global _chroma_instance
    if _chroma_instance is None:
        embeddings = get_embeddings_model()
        if not os.path.exists(CHROMA_DB_DIRECTORY):
            logger.info(f"Diretório ChromaDB não encontrado em {CHROMA_DB_DIRECTORY}. Criando...")
            os.makedirs(CHROMA_DB_DIRECTORY, exist_ok=True)
            # O ChromaDB será inicializado como vazio e populado na primeira ingestão
            _chroma_instance = Chroma(
                persist_directory=CHROMA_DB_DIRECTORY,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME
            )
            logger.info(f"Nova instância ChromaDB criada em {CHROMA_DB_DIRECTORY} com coleção '{CHROMA_COLLECTION_NAME}'.")
        else:
            logger.info(f"ChromaDB carregado do diretório: {CHROMA_DB_DIRECTORY}")
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
):
    """
    Adiciona um documento (PDF/DOCX/TXT) ao ChromaDB de forma incremental,
    evitando duplicações e tratando rate limits da API.
    """
    file_name = os.path.basename(file_path)
    logger.info(f"Processando arquivo: {file_name} (hash: {get_file_hash(file_path)[:8]})")

    try:
        base_name = os.path.splitext(file_name)[0]

        # --- Lógica de extração e fallback para metadados ---
        
        # client_name
        client_name_from_file = None
        client_match = re.match(r'\[(.*?)\]', base_name)
        if client_match:
            client_name_from_file = client_match.group(1).strip() # Remove espaços em branco
        client_name_from_detected = detected_metadata.get('client_name')
        
        final_client_name = (client_name_from_detected or client_name_from_file or 'UNKNOWN CLIENT').upper()
        logger.debug(f"[METADATA] Final Client Name para {file_name}: {final_client_name}")

        # doc_type
        doc_type_from_file = None
        doc_type_match = re.search(r'(MIT\d{3})', base_name, re.IGNORECASE)
        if doc_type_match:
            doc_type_from_file = doc_type_match.group(1).strip()
        doc_type_from_detected = detected_metadata.get('doc_type')

        final_doc_type = (doc_type_from_detected or doc_type_from_file or 'UNKNOWN DOC TYPE').upper()
        logger.debug(f"[METADATA] Final Doc Type para {file_name}: {final_doc_type}")

        # project_code
        project_code_from_file = None
        project_code_match = re.search(r'(D\d{9,15})', base_name, re.IGNORECASE)
        if project_code_match:
            project_code_from_file = project_code_match.group(1).strip()
        project_code_from_detected = detected_metadata.get('project_code')

        final_project_code = (project_code_from_detected or project_code_from_file or 'UNKNOWN PROJECT').upper()
        logger.debug(f"[METADATA] Final Project Code para {file_name}: {final_project_code}")

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
        logger.info(f"Carregados {len(pages)} páginas/seções do documento por {type(loader).__name__}.")

        # Adicionar metadados adicionais aos documentos antes de chunkar
        for i, page in enumerate(pages):
            page.metadata["source"] = file_name
            page.metadata["document_hash"] = document_hash
            page.metadata["original_filename"] = file_name # Nome original completo do arquivo
            
            # Para PDFs, o loader já pode ter um page_number. Mantemos ou adicionamos.
            # Para DOCX/TXT, adicionamos um 'índice' de página
            if "page" in page.metadata: # Algumas vezes é 'page' ou 'page_number'
                page.metadata["page_number"] = page.metadata["page"]
            elif "page_number" not in page.metadata: 
                page.metadata["page_number"] = i + 1 # Adiciona o número da página (começando em 1)
            
            page.metadata["client_name"] = final_client_name
            page.metadata["doc_type"] = final_doc_type
            page.metadata["project_code"] = final_project_code

            logger.debug(f"[METADATA_PAGE] Página {page.metadata.get('page_number', i+1)} de {file_name} com metadados: {page.metadata}")


        chunks = text_splitter.split_documents(pages)
        # Adicionar um chunk_index a cada chunk para referência
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            # Garante que os metadados específicos estejam em CADA CHUNK, mesmo que já estejam nas 'pages'
            # Isso é redundante mas seguro e garante que o ChromaDB os receba
            chunk.metadata["client_name"] = final_client_name
            chunk.metadata["doc_type"] = final_doc_type
            chunk.metadata["project_code"] = final_project_code
            logger.debug(f"[METADATA_CHUNK] Chunk {i} de {file_name} com metadados: {chunk.metadata}")


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
                    reset_dt_utc = datetime.datetime.strptime(reset_time_match.group(1), '%Y-%m-%d %H:%M:%S')
                    current_dt_utc = datetime.datetime.utcnow()
                    
                    # Calcula a diferença em segundos, adiciona um buffer e garante que seja positivo
                    calculated_wait_time = (reset_dt_utc - current_dt_utc).total_seconds() + 5 
                    if calculated_wait_time > 0:
                        wait_time = calculated_wait_time
                        logger.info(f"Aguardando até {reset_dt_utc} UTC (aproximadamente {wait_time:.0f} segundos) antes de tentar novamente...")
                    else: # Se o tempo de reset já passou ou é insignificante
                        logger.info(f"O tempo de reset já passou ou está muito próximo ({calculated_wait_time:.0f}s). Aguardando o tempo padrão de {wait_time} segundos.")
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

def query_chroma_metadata(client_name: Optional[str] = None, project_code: Optional[str] = None, doc_type: Optional[str] = None):
    """
    Consulta o ChromaDB para listar IDs e metadados de documentos, com filtros opcionais.
    Útil para depuração.
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
    
    # Se não houver filtro, pegamos um número limitado para não sobrecarregar em bases grandes
    if not where_clause:
        results = vector_store._collection.get(limit=100, include=['metadatas'])
    else:
        results = vector_store._collection.get(where=where_clause, include=['metadatas'])

    if results and results['ids']:
        logger.info(f"Encontrados {len(results['ids'])} resultados.")
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            logger.info(f"  ID: {doc_id[:8]}... Metadata: {metadata}")
    else:
        logger.info("Nenhum documento encontrado para a consulta.")
    
    return results

def get_known_entities() -> Dict[str, List[str]]:
    """
    Consulta o ChromaDB para retornar todos os nomes de clientes e códigos de projeto únicos.
    """
    vector_store = get_chroma_instance()
    
    # Isso é um pouco custoso, mas aceitável para um pequeno número de metadados
    # Em um cenário de produção com milhões de documentos, talvez buscar de um cache otimizado.
    try:
        results = vector_store._collection.get(include=['metadatas'])
    except Exception as e:
        logger.error(f"Erro ao obter metadados do ChromaDB: {e}")
        return {"client_names": [], "project_codes": [], "doc_types": []}

    client_names = set()
    project_codes = set()
    doc_types = set()

    if results and results.get('metadatas'):
        for metadata in results['metadatas']:
            if 'client_name' in metadata and metadata['client_name']:
                client_names.add(metadata['client_name'].upper())
            if 'project_code' in metadata and metadata['project_code']:
                project_codes.add(metadata['project_code'].upper())
            if 'doc_type' in metadata and metadata['doc_type']:
                doc_types.add(metadata['doc_type'].upper())

    logger.debug(f"Entidades Conhecidas - Clientes: {list(client_names)}, Projetos: {list(project_codes)}, Tipos Doc: {list(doc_types)}")
    return {
        "client_names": sorted(list(client_names)),
        "project_codes": sorted(list(project_codes)),
        "doc_types": sorted(list(doc_types))
    }

# --- Bloco de Execução Principal (consolidado) ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv() # Carrega as variáveis de ambiente

    logger.setLevel(logging.DEBUG) # Aumenta o nível de log para debug em teste local

    # --- Teste de Ingestão de Documentos (Opcional) ---
    # Descomente o bloco abaixo para realizar a ingestão de documentos de teste
    #
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
    #     os.path.join(sample_docs_dir, "Roteiro do Projeto_ TCRM Copilot - Prova de Conceito (PoC).pdf"), # Adicionado o PDF da PoC
    # ]
    # for f_path in files_to_ingest:
    #     if os.path.exists(f_path):
    #         try:
    #             metadata_for_ingestion = {} 
    #             logger.info(f"Iniciando ingestão de {os.path.basename(f_path)}...")
    #             add_document_to_vector_store(f_path, chroma_instance, text_splitter, metadata_for_ingestion)
    #             time.sleep(1)
    #         except Exception as e:
    #             logger.error(f"Falha ao ingerir {os.path.basename(f_path)}: {e}")
    #     else:
    #         logger.warning(f"Arquivo não encontrado: {f_path}")
    # logger.info("Processo de ingestão incremental concluído.")

    # --- Testes de Consulta de Metadados ---
    logger.setLevel(logging.INFO) # Volta para INFO para os testes de consulta, menos verboso

    print("\n--- Querying ChromaDB for KION ---")
    query_chroma_metadata(client_name="KION")

    print("\n--- Querying ChromaDB for SCENS ---")
    query_chroma_metadata(client_name="SCENS")
    
    print("\n--- Querying ChromaDB for INOVA ---")
    query_chroma_metadata(client_name="INOVA")
    
    print("\n--- Querying ChromaDB for MARSON ---")
    query_chroma_metadata(client_name="MARSON")

    print("\n--- Querying ChromaDB for Documents with MIT041 ---")
    query_chroma_metadata(doc_type="MIT041")
    
    print("\n--- Querying ChromaDB for ALL documents (first 100) ---")
    query_chroma_metadata() # Sem filtros, pega os primeiros 100
    
    # --- Teste de Entidades Conhecidas (Este é o crucial para o copilot_agent) ---
    print("\n--- Querying ALL known entities ---")
    known_entities = get_known_entities()
    print(known_entities)