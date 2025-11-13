"""
Módulo LEGADO para construção inicial da base de conhecimento vetorial (ChromaDB).

Este módulo foi responsável por uma das primeiras implementações da pipeline
de ingestão de documentos no projeto TCRM Copilot. Ele carrega documentos
(PDF, DOCX, TXT) de um diretório específico, extrai seu conteúdo,
identifica metadados através de heurísticas baseadas em nomes de arquivo
e conteúdo (como 'Código do Projeto' dentro do texto), divide o texto em chunks,
gera embeddings usando a OpenAI e os persiste no ChromaDB.

ATENÇÃO: Este arquivo é considerado LEGADO e não é mais utilizado na arquitetura principal
do sistema. Sua funcionalidade foi substituída e aprimorada pelo módulo
`src/data_ingestion/incremental_ingestor.py`. O `incremental_ingestor.py`
apresenta uma abordagem mais sofisticada para:
- Ingestão incremental (evitando re-processamento de documentos existentes).
- Tratamento de Rate Limits da API da OpenAI.
- Detecção e fallback de metadados mais robusta.
- Utilização de `logging` padronizado.
- Integração mais direta com os loaders da LangChain.

Este arquivo é mantido para fins de referência histórica e para ilustrar
a evolução das estratégias de ingestão de dados e construção da base de
conhecimento ao longo do desenvolvimento do projeto.
"""

import os
import re
import sys
import glob
import logging
from pathlib import Path
import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import httpx
# from chromadb.utils import embedding_functions # Removido, pois não é utilizado diretamente.
from langchain_core.documents import Document
import hashlib # Importado para gerar o hash do documento

# Carrega as variáveis de ambiente do .env
load_dotenv()

# --- Configuração de Logging ---
# Configura o logger para este módulo.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes de Configuração ---
# Caminhos relativos e nomes de coleção/modelo para a construção da base.
# `Path` é usado para manipulação de caminhos de forma mais robusta e independente de SO.
DATA_DIR = Path("data")
RAW_DOCUMENTS_DIR = DATA_DIR / "raw_documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db" / "chroma_db"
CHROMA_COLLECTION_NAME = 'tcrm_copilot_kb'
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 150

# Garante que os diretórios necessários existam.
RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Diretórios verificados/criados: {RAW_DOCUMENTS_DIR}, {VECTOR_DB_DIR}")


# --- Inicialização do Modelo de Embeddings ---
# Configura o cliente HTTP para ignorar a verificação SSL.
# ATENÇÃO: 'verify=False' desabilita a verificação de certificados SSL, o que pode
# ser um RISCO DE SEGURANÇA em ambientes de produção. É usado aqui para PoC/testes
# com proxies internos que podem não ter certificados confiáveis.
# Para produção, configure a verificação SSL corretamente ou use certificados CA.
custom_http_client = httpx.Client(verify=False)
logger.info("Cliente HTTP configurado para ignorar verificação SSL (httpx.Client(verify=False)).")


# Inicializa o modelo de Embeddings da OpenAI.
# As chaves e URLs base são carregadas das variáveis de ambiente.
embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    http_client=custom_http_client
)
logger.info(f"Modelo de embeddings '{EMBEDDING_MODEL_NAME}' inicializado.")


def load_and_chunk_documents() -> List[Document]:
    """
    Carrega documentos do diretório 'RAW_DOCUMENTS_DIR', extrai o texto,
    identifica metadados relevantes a partir do nome do arquivo e do conteúdo,
    divide o texto em chunks e associa esses metadados a cada chunk.

    Esta função representa a lógica central da construção inicial da base de conhecimento,
    demonstrando as heurísticas adotadas para enriquecer os dados antes da vetorização.

    Returns:
        List[Document]: Uma lista de objetos LangChain Document, onde cada Document
                        representa um chunk de texto com metadados enriquecidos.
    """
    documents_with_metadata: List[Document] = []
    
    # Inicializa o splitador de texto com tamanho e sobreposição definidos.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Adiciona um índice de caractere inicial para cada chunk.
    )
    logger.info(f"Splitador de texto inicializado (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")

    # Itera sobre todos os arquivos no diretório de documentos brutos.
    for file_path_obj in RAW_DOCUMENTS_DIR.iterdir():
        file_path = str(file_path_obj)
        filename = file_path_obj.name
        logger.info(f"Processando documento: {filename}")

        loader = None
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            # Para arquivos TXT, detecta a codificação para evitar erros de leitura.
            # `chardet` é usado para uma detecção robusta, com fallback para 'utf-8'.
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
            loader = TextLoader(file_path, encoding=encoding)
            logger.debug(f"Detectada codificação '{encoding}' para o arquivo TXT: {filename}")
        else:
            logger.warning(f"Tipo de arquivo não suportado, pulando: {filename}")
            continue

        try:
            # Carrega o conteúdo do documento usando o loader apropriado.
            # O conteúdo de todas as páginas é concatenado para facilitar a busca de metadados no texto.
            loaded_pages = loader.load()
            doc_content = "".join([page.page_content for page in loaded_pages])
            logger.debug(f"Conteúdo de {len(loaded_pages)} páginas carregado de {filename}.")
            
            # --- Geração de Hash para Identificação Única ---
            # Um hash SHA256 do conteúdo completo do documento é gerado para identificar
            # univocamente o documento original, útil para detecção de duplicidade.
            document_hash = hashlib.sha256(doc_content.encode('utf-8')).hexdigest()
            logger.debug(f"Hash do documento '{filename}': {document_hash[:8]}...")

            # --- Extração de Metadados Heurística ---
            # Esta seção implementa a lógica para extrair metadados importantes
            # a partir do nome do arquivo e do conteúdo textual, padronizando-os.

            # 1. client_name: Extrai o nome do cliente de colchetes no nome do arquivo (e.g., "[NOME_CLIENTE] - ...").
            client_name_match = re.search(r'\[([A-Za-z0-9_\s]+)\]', filename)
            client_name = client_name_match.group(1).strip() if client_name_match else 'Unknown Client'
            if client_name != 'Unknown Client':
                client_name = client_name.upper() # Padroniza para maiúsculas.
            logger.debug(f"Metadado 'client_name' extraído para {filename}: {client_name}")
            
            # 2. doc_type: Extrai o tipo de documento, e.g., "MIT041", do nome do arquivo.
            doc_type_match = re.search(r'(MIT\d{3})', filename, re.IGNORECASE)
            doc_type = doc_type_match.group(1).upper() if doc_type_match else 'Generic Doc'
            logger.debug(f"Metadado 'doc_type' extraído para {filename}: {doc_type}")

            # 3. project_code: Tenta extrair o código do projeto do *conteúdo* do documento,
            #    procurando por padrões como "Código do Projeto: DXXXXXXXXXXXX".
            project_code_match = re.search(r'Código do Projeto: (D\d{9,15})', doc_content)
            project_code = project_code_match.group(1).strip() if project_code_match else 'Unknown Project'
            logger.debug(f"Metadado 'project_code' extraído para {filename}: {project_code}")

            # --- Chunking e Adição de Metadados ---
            # O conteúdo é dividido em chunks, e os metadados extraídos são anexados a cada um.
            chunks_content: List[str] = text_splitter.split_text(doc_content)
            logger.debug(f"Documento '{filename}' dividido em {len(chunks_content)} chunks.")
            
            for i, chunk_content in enumerate(chunks_content):
                metadata = {
                    "source": os.path.join("data", "raw_documents", filename), # Caminho relativo para o arquivo original.
                    "original_filename": filename,
                    "client_name": client_name,
                    "doc_type": doc_type,
                    "project_code": project_code,
                    "chunk_number": i + 1,      # Número sequencial do chunk (base 1).
                    "chunk_index": i,           # Índice do chunk (base 0) - útil para ordenação.
                    "document_hash": document_hash # Hash do documento original para rastreamento.
                }
                documents_with_metadata.append(Document(page_content=chunk_content, metadata=metadata))
                logger.debug(f"Chunk {i+1} de {filename} preparado com metadados: {metadata}")

        except Exception as e:
            logger.error(f"Erro ao processar o arquivo '{filename}': {e}", exc_info=True)
            # Continua para o próximo arquivo, em vez de parar todo o processo.

    return documents_with_metadata


# --- Bloco de Execução Principal (para testes locais e reconstrução da base) ---
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG) # Aumenta o nível de log para DEBUG para teste local.
    logger.info("--- Iniciando a construção da base de conhecimento (LEGADO knowledge_base_builder.py) ---")
    
    # Remove a pasta do ChromaDB existente para garantir uma reconstrução limpa da base.
    # Esta operação é DESTRUTIVA e apaga todos os embeddings previamente persistidos.
    # Isso reflete a natureza 'batch' e não incremental desta versão do construtor.
    if VECTOR_DB_DIR.exists():
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)
        logger.warning(f"Diretório ChromaDB existente '{VECTOR_DB_DIR}' removido para reconstrução completa.")
    else:
        logger.info(f"Diretório ChromaDB '{VECTOR_DB_DIR}' não encontrado. Criando novo...")
        
    docs_to_ingest = load_and_chunk_documents()

    if docs_to_ingest:
        logger.info(f"Total de {len(docs_to_ingest)} chunks preparados para ingestão no ChromaDB.")
        # Cria e persiste a nova base de dados no ChromaDB a partir dos documentos processados.
        db = Chroma.from_documents(
            docs_to_ingest,
            embeddings_model,
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=CHROMA_COLLECTION_NAME
        )
        logger.info(f"Ingestão concluída e ChromaDB persistido em '{VECTOR_DB_DIR}' com coleção '{CHROMA_COLLECTION_NAME}'.")
    else:
        logger.warning("Nenhum documento para ingestão encontrado. Verifique se a pasta 'data/raw_documents' contém arquivos suportados.")
    
    logger.info("--- Conclusão da execução do LEGADO knowledge_base_builder.py ---")
