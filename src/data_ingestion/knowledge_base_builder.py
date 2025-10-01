# src/data_ingestion/knowledge_base_builder.py

import os
import shutil
import re
from typing import List, Dict, Any

# LangChain/ChromaDB imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Unstructured for document parsing
from unstructured.partition.auto import partition
from unstructured.documents.elements import Text

# HTTPX for custom client (proxy/SSL)
import httpx
import sys

# --- Configuration ---
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'chroma_db'))
RAW_DOCUMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw_documents'))
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_metadata_from_content(text_content: str) -> Dict[str, str]:
    """
    Extrai metadados específicos de um conteúdo textual, focando na seção 'AMBIENTAÇÃO'.
    Utiliza expressões regulares para buscar campos como 'Nome do cliente', 'Código do projeto', etc.
    Melhorado para lidar com formatação de linhas e capturar apenas o valor.
    """
    metadata = {}
    
    # Define padrões para os rótulos que queremos extrair e seus delimitadores
    # O (?:...) é um grupo de não-captura para os delimitadores
    patterns = {
        "client_name": r"Nome do cliente:\s*(.+?)(?:\n|Código de cliente:|Nome do projeto:|HISTÓRICO DE REVISÕES|SUMÁRIO|OBJETIVO|\Z)",
        "project_code_crm": r"Código do projeto:\s*(.+?)(?:\n|Segmento cliente:|Unidade TOTVS:|HISTÓRICO DE REVISÕES|SUMÁRIO|OBJETIVO|\Z)",
        "totvs_coordinator": r"Gerente\/Coordenador TOTVS:\s*(.+?)(?:\n|Gerente\/Coordenador cliente:|HISTÓRICO DE REVISÕES|SUMÁRIO|OBJETIVO|\Z)"
    }
    
    # Tenta extrair a seção AMBIENTAÇÃO para restringir a busca de metadados, se ela existir.
    # Usamos re.DOTALL para que '.' case com newlines também dentro da seção.
    amb_match = re.search(r"AMBIENTAÇÃO\s*(.*?)(?=(?:HISTÓRICO DE REVISÕES|SUMÁRIO|OBJETIVO|$))", text_content, re.DOTALL | re.IGNORECASE)
    
    section_to_parse = text_content # Por padrão, busca no texto completo
    if amb_match:
        section_to_parse = amb_match.group(1) # Se encontrou, restringe a busca a essa seção

    for key, pattern in patterns.items():
        # Usar re.DOTALL aqui para que (.*?) possa atravessar linhas se necessário até o delimitador
        match = re.search(pattern, section_to_parse, re.DOTALL | re.IGNORECASE)
        if match:
            # strip() para remover espaços em branco ou newlines extras
            metadata[key] = match.group(1).strip()
    
    return metadata

def parse_document(filepath: str) -> List[Document]:
    """
    Carrega e processa um documento, extraindo texto e metadados.
    Tenta extrair metadados do conteúdo do documento.
    """
    filename = os.path.basename(filepath)
    print(f"DEBUG: Parsing document: {filename}", flush=True)

    try:
        # Usar partition.auto para lidar com PDF, DOCX, TXT
        # strategy="fast" para um processamento mais rápido
        elements = partition(filename=filepath, strategy="fast", languages=["por"])
        
        # Concatena o texto dos elementos para a extração de metadados de conteúdo
        # Filtra elementos que não são de texto para evitar erros em str(el)
        full_text_content = "\n\n".join([str(el) for el in elements if isinstance(el, Text)])
        
        # Extrair metadados do conteúdo
        content_metadata = extract_metadata_from_content(full_text_content)
        
        # Metadados base (do arquivo)
        base_metadata = {
            "source": filename,
            "source_type": "document"
        }
        
        # Ajustado para extrair "MIT041" de "MIT041 V1.0" ou de forma mais robusta
        document_id = None
        # Tenta pegar "MIT041"
        mit_match = re.search(r"(MIT\d{3})", filename)
        if mit_match:
            document_id = mit_match.group(1)
        else: # Tenta pegar qualquer coisa que se pareça com um ID de documento
            # Pega o nome base do arquivo sem extensão e remove caracteres especiais
            clean_filename = os.path.splitext(filename)[0]
            # Remove "Scens", "Escopo Técnico", "TOTVS CRM", "Gestão de Clientes"
            clean_filename = re.sub(r"\[?Scens\]? -? Escopo Técnico TOTVS CRM Gestão de Clientes -?", "", clean_filename, flags=re.IGNORECASE).strip()
            # Pega a primeira sequência alfanumérica que não seja uma versão ou data
            generic_id_match = re.search(r"(\w+)(?: V\d+\.\d+|\s\d{2}-\d{2}-\d{4})?", clean_filename)
            if generic_id_match:
                document_id = generic_id_match.group(1)

        base_metadata["document_id"] = document_id if document_id else os.path.splitext(filename)[0].replace(' ', '_').replace('-', '_')

        # Combina metadados do conteúdo com metadados base. Metadados do conteúdo têm precedência.
        combined_metadata = {**base_metadata, **content_metadata}

        # Cria um Document para o texto completo (pode ser dividido em chunks depois)
        doc = Document(page_content=full_text_content, metadata=combined_metadata)
        print(f"DEBUG: Document parsed: {filename} with metadata: {combined_metadata}", flush=True)
        return [doc]
    except Exception as e:
        print(f"ERROR: Failed to parse {filename}: {e}", file=sys.stderr, flush=True)
        return []

def build_knowledge_base():
    print(f"--- Iniciando a construção da base de conhecimento ---", flush=True)
    print(f"Diretório de documentos raw: {RAW_DOCUMENTS_DIR}", flush=True)
    print(f"Diretório de destino do ChromaDB: {CHROMA_DB_DIRECTORY}", flush=True)

    # Remove o diretório ChromaDB existente para garantir uma reconstrução limpa
    if os.path.exists(CHROMA_DB_DIRECTORY):
        print(f"DEBUG: Removendo diretório ChromaDB existente: {CHROMA_DB_DIRECTORY}", flush=True)
        shutil.rmtree(CHROMA_DB_DIRECTORY)
        print("DEBUG: Diretório ChromaDB removido com sucesso.", flush=True)
    os.makedirs(CHROMA_DB_DIRECTORY, exist_ok=True) # Garante que o diretório base existe

    loaded_documents: List[Document] = []
    if not os.path.exists(RAW_DOCUMENTS_DIR):
        print(f"WARNING: Diretório de documentos raw não encontrado: {RAW_DOCUMENTS_DIR}", file=sys.stderr, flush=True)
    else:
        for filename in os.listdir(RAW_DOCUMENTS_DIR):
            filepath = os.path.join(RAW_DOCUMENTS_DIR, filename)
            if os.path.isfile(filepath):
                # Filtra arquivos temporários ou ocultos
                if filename.startswith('~$') or filename.startswith('.'):
                    print(f"DEBUG: Ignorando arquivo temporário/oculto: {filename}", flush=True)
                    continue

                print(f"DEBUG: Processando arquivo: {filepath}", flush=True)
                parsed_docs = parse_document(filepath)
                loaded_documents.extend(parsed_docs)

    print(f"Total de {len(loaded_documents)} documentos carregados para chunking.", flush=True)
    if not loaded_documents:
        print("WARNING: Nenhum documento válido encontrado ou processado. ChromaDB será vazio.", file=sys.stderr, flush=True)
        return # Sai da função se não houver documentos para processar

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(loaded_documents)
    print(f"Documentos divididos em {len(chunks)} chunks.", flush=True)

    # OpenAI API Key para embeddings
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_api_base = os.getenv("OPENAI_API_BASE")
    if not openai_api_key:
        print("ERROR: Variável de ambiente OPENAI_API_KEY não configurada. Não é possível gerar embeddings.", file=sys.stderr, flush=True)
        return

    embeddings_model = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        openai_api_key=openai_api_key,
        base_url=openai_api_base if openai_api_base else None,
        http_client=httpx.Client(verify=False)
    )
    print("DEBUG: Modelo de embeddings carregado.", flush=True)

    print(f"DEBUG: Populando ChromaDB em {CHROMA_DB_DIRECTORY} com {len(chunks)} chunks...", flush=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=CHROMA_DB_DIRECTORY # A persistência é tratada aqui.
    )
    # REMOVIDO: vector_store.persist() # Esta linha causava o AttributeError

    print("DEBUG: ChromaDB populado e persistido com sucesso.", flush=True)
    
    final_chunk_count = len(vector_store.get(include=['metadatas'])['metadatas'])
    print(f"RESULT: Número final de chunks persistidos no ChromaDB: {final_chunk_count}", flush=True)
    print("--- Construção da base de conhecimento finalizada ---", flush=True)

if __name__ == "__main__":
    build_knowledge_base()
