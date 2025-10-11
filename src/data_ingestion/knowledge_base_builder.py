import os
import re
import shutil
import pathlib
from typing import List, Dict, Any

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

# Carrega variáveis de ambiente (como OPENAI_API_KEY)
load_dotenv()

# --- Configurações ---
# Diretório onde os documentos crus (PDF, DOCX, TXT) estão localizados
RAW_DOCS_DIR = pathlib.Path("data/raw_documents")
# Diretório onde o banco de vetores Chroma será armazenado
VECTOR_DB_DIR = pathlib.Path("data/vector_db/chroma_db")

# Tamanho de cada pedaço de texto (chunk) para o LLM
CHUNK_SIZE = 1000
# Sobreposição entre chunks para manter o contexto
CHUNK_OVERLAP = 200
# Modelo de embedding da OpenAI
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- Funções de Ajuda ---

def extract_metadata_from_document_content(doc_content: str, filename: str) -> Dict[str, str]:
    """
    Extrai o nome do cliente e o código do projeto do conteúdo do documento
    com maior precisão, priorizando o nome do arquivo para o cliente.
    """
    client_name = "Unknown Client"
    project_code = "Unknown Project Code"

    # 1. EXTRAÇÃO DO NOME DO CLIENTE: Prioridade máxima para o nome do arquivo
    # Ex: "[KION] - Escopo Técnico..." -> "KION"
    filename_client_match = re.match(r"\[(.*?)\]", filename)
    if filename_client_match:
        client_name = filename_client_match.group(1).strip()
    # NUNCA sobrescrever o client_name se encontrado no nome do arquivo.

    # 2. EXTRAÇÃO DO CÓDIGO DO PROJETO: Buscar no conteúdo, com regex mais específica
    # Ex: "Código do Projeto: D000071597001" -> "D000071597001"
    project_code_match = re.search(
        r"Código do Projeto:\s*(D\d{9,15}|\w{3}\d{3})", # Captura "D" + dígitos (pelo menos 9) OU "MIT" + 3 dígitos
        doc_content, re.IGNORECASE
    )
    if project_code_match:
        project_code = project_code_match.group(1).strip()
    
    # Fallback para o código do projeto (se não encontrado no conteúdo, talvez MIT041 no nome do arquivo)
    if project_code == "Unknown Project Code" or not project_code:
        mit_match = re.search(r"(MIT\d{3})", filename, re.IGNORECASE)
        if mit_match:
            project_code = mit_match.group(1).strip() # Ex: MIT041


    return {
        "client_name": client_name,
        "project_code": project_code,
        "original_filename": filename
    }

def load_documents_with_enriched_metadata(directory: pathlib.Path) -> List[Document]:
    """
    Carrega documentos de um diretório, extrai metadados do conteúdo
    e os adiciona ao objeto Document.
    """
    documents = []
    for filepath in directory.iterdir():
        if filepath.is_file() and filepath.suffix in [".pdf", ".docx", ".txt"]:
            print(f"Loading and extracting metadata from: {filepath.name}")
            try:
                # Carrega o documento, focando no conteúdo para extração de metadados
                loader = UnstructuredFileLoader(str(filepath), mode="elements")
                
                # O UnstructuredFileLoader pode retornar múltiplos elementos,
                # vamos tentar pegar o texto completo para extração de metadados.
                raw_docs_elements = loader.load()
                full_content = "\n".join([doc.page_content for doc in raw_docs_elements])

                # Extrai metadados personalizados
                custom_metadata = extract_metadata_from_document_content(full_content, filepath.name)

                # Criamos um único Documento para o arquivo completo, com os metadados.
                # O text_splitter irá quebrar este Documento em chunks,
                # e os metadados serão propagados.
                doc = Document(
                    page_content=full_content,
                    metadata={
                        "source": str(filepath),
                        **custom_metadata # Adiciona os metadados customizados aqui
                    }
                )
                documents.append(doc)
                print(f"  -> Extracted Client: '{custom_metadata['client_name']}', Project Code: '{custom_metadata['project_code']}'")

            except Exception as e:
                print(f"Error loading or processing {filepath.name}: {e}")
    return documents

# --- Pipeline de Construção da Base de Conhecimento ---

def build_knowledge_base():
    """
    Constrói (ou reconstrói) a base de conhecimento (ChromaDB)
    a partir dos documentos brutos.
    """
    print(f"Starting knowledge base construction...")

    # 1. Carregar documentos e extrair metadados enriquecidos
    print(f"Loading documents from {RAW_DOCS_DIR}...")
    documents = load_documents_with_enriched_metadata(RAW_DOCS_DIR)
    if not documents:
        print("No documents found or processed. Exiting.")
        return

    # 2. Dividir documentos em chunks
    print(f"Splitting {len(documents)} document(s) into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    # A atenção aqui: `create_documents` ou `split_documents` da LangChain
    # irá manter os metadados do documento original em cada chunk.
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Gerar embeddings
    print(f"Initializing OpenAI embeddings with model '{EMBEDDING_MODEL}'...")
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 4. Popular ChromaDB
    print(f"Populating ChromaDB at {VECTOR_DB_DIR}...")
    
    # Garante que o diretório do ChromaDB existe
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    # Remove o conteúdo existente para reconstruir (opcional, mas bom para testes iniciais)
    # Cuidado: shuta o balde!
    if VECTOR_DB_DIR.exists() and any(VECTOR_DB_DIR.iterdir()):
        print(f"  -> Existing ChromaDB detected. Deleting content for fresh build.")
        for item in VECTOR_DB_DIR.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=str(VECTOR_DB_DIR)
    )
    vectorstore.persist()
    print("ChromaDB population complete. Knowledge base is ready!")

if __name__ == "__main__":
    build_knowledge_base()