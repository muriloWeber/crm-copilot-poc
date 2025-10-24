# src/data_ingestion/knowledge_base_builder.py

import os
import re
import sys
import glob
from pathlib import Path
import chardet
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import httpx
from chromadb.utils import embedding_functions
from langchain_core.documents import Document
import hashlib # Importar hashlib para gerar o hash do documento

# Carrega as variáveis de ambiente do .env
load_dotenv()

# --- Configurações ---
DATA_DIR = Path("data")
RAW_DOCUMENTS_DIR = DATA_DIR / "raw_documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db" / "chroma_db"
CHROMA_COLLECTION_NAME = 'tcrm_copilot_kb' 
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
CHUNK_SIZE = 750 # Mantendo seu CHUNK_SIZE
CHUNK_OVERLAP = 150 # Mantendo seu CHUNK_OVERLAP

# Garante que os diretórios existam
RAW_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# Inicializa o modelo de Embeddings da OpenAI
# Configura o cliente HTTP para ignorar a verificação SSL, essencial para o proxy da TOTVS
custom_http_client = httpx.Client(verify=False)

embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    http_client=custom_http_client
)

def load_and_chunk_documents():
    documents_with_metadata = [] # Renomeado para evitar confusão com 'documents' na função
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    for file_path_obj in RAW_DOCUMENTS_DIR.iterdir():
        file_path = str(file_path_obj)
        filename = file_path_obj.name
        print(f"Processando documento: {filename}")

        loader = None
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif filename.endswith(".txt"):
            # Para TXT, detecta a codificação para evitar erros
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] if result['encoding'] else 'utf-8' # Fallback para utf-8

            loader = TextLoader(file_path, encoding=encoding)
        else:
            print(f"Tipo de arquivo não suportado, pulando: {filename}")
            continue

        try:
            # Carrega o conteúdo do documento
            doc_content = "".join([page.page_content for page in loader.load()])
            
            # === NOVO: Gerar um hash único para o conteúdo original do documento ===
            document_hash = hashlib.sha256(doc_content.encode('utf-8')).hexdigest()
            # ======================================================================

            # MODIFICAÇÃO: Extrai client_name e doc_type do filename ou content
            client_name_match = re.search(r'\[([A-Za-z0-9_\s]+)\]', filename)
            # Prioriza o nome dentro de colchetes, se não encontrar, tenta por heurística simples ou Unknown
            client_name = client_name_match.group(1).strip() if client_name_match else 'Unknown Client'
            
            # AGORA, TODOS OS NOMES DE CLIENTE VÁLIDOS SÃO PADRONIZADOS PARA UPPERCASE NO METADADO.
            if client_name != 'Unknown Client':
                client_name = client_name.upper() 
            
            doc_type_match = re.search(r'(MIT\d{3})', filename, re.IGNORECASE)
            doc_type = doc_type_match.group(1).upper() if doc_type_match else 'Generic Doc'

            # MODIFICAÇÃO: Extrai project_code.
            project_code_match = re.search(r'Código do Projeto: (D\d{9,15})', doc_content)
            project_code = project_code_match.group(1).strip() if project_code_match else 'Unknown Project'

            # Cria chunks e adiciona metadados enriquecidos
            chunks = text_splitter.split_text(doc_content)
            
            for i, chunk_content in enumerate(chunks):
                metadata = {
                    "source": os.path.join("data", "raw_documents", filename), # Caminho relativo
                    "original_filename": filename,
                    "client_name": client_name, 
                    "doc_type": doc_type,       
                    "project_code": project_code,
                    "chunk_number": i + 1,      # Mantém seu chunk_number (1-based)
                    "chunk_index": i,           # NOVO: chunk_index (0-based) para lógica do agente
                    "document_hash": document_hash # NOVO: Hash para identificar o documento original
                }
                documents_with_metadata.append(Document(page_content=chunk_content, metadata=metadata))

        except Exception as e:
            print(f"Erro ao processar {filename}: {e}")

    return documents_with_metadata


if __name__ == "__main__":
    print("Iniciando a ingestão de documentos para o ChromaDB...")
    
    # Remove a pasta do ChromaDB existente para garantir uma reconstrução limpa
    if VECTOR_DB_DIR.exists():
        import shutil
        shutil.rmtree(VECTOR_DB_DIR)
        print(f"Diretório ChromaDB existente '{VECTOR_DB_DIR}' removido para reconstrução.")

    docs_to_ingest = load_and_chunk_documents()

    if docs_to_ingest:
        print(f"Total de {len(docs_to_ingest)} chunks preparados para ingestão.")
        # Cria a nova base de dados no ChromaDB
        db = Chroma.from_documents(
            docs_to_ingest,
            embeddings_model,
            persist_directory=str(VECTOR_DB_DIR),
            collection_name=CHROMA_COLLECTION_NAME 
        )
        print("Ingestão concluída e ChromaDB persistido.")
    else:
        print("Nenhum documento para ingestão. Verifique a pasta 'data/raw_documents'.")
