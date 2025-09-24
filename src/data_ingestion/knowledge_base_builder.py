# src/data_ingestion/knowledge_base_builder.py

import os
import shutil
import re
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_parser import extract_text_from_file_unstructured
import httpx # Importe httpx

# Configurações globais para o RAG
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/chroma_db'))
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def extract_doc_id_from_filename(filename: str) -> str | None:
    """
    Extrai o ID do documento (ex: MIT041) do nome de arquivo, baseado no padrão esperado.
    Ex: "[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx"
    """
    # Regex mais robusta para pegar o MITXXX, mesmo que tenha outros códigos perto
    match = re.search(r'\b([A-Z]{3}\d{3})\b', filename)
    if match:
        return match.group(1)
    return None

def initialize_embedding_model():
    """
    Inicializa o modelo de embeddings da OpenAI.
    """
    try:
        # Pega a base_url do ambiente, se existir. Para proxys como o da TOTVS.
        openai_api_base = os.getenv("OPENAI_API_BASE")
        
        # Cria um cliente HTTP personalizado para httpx (para permitir verify=False, se necessário)
        # Atenção: verify=False desabilita a verificação de certificados SSL e deve ser usado com cautela em produção.
        # É útil em ambientes de desenvolvimento ou com proxies que interceptam SSL.
        custom_http_client = httpx.Client(verify=False)

        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url=openai_api_base if openai_api_base else None, # Usa base_url se definida
            http_client=custom_http_client # Passa o cliente HTTP personalizado
        )
    except Exception as e:
        print(f"Erro ao inicializar o modelo de embeddings: {e}")
        print("Certifique-se de que OPENAI_API_KEY está configurada e o acesso à API está ok.")
        print("Em ambientes corporativos, pode ser necessário desabilitar a verificação SSL se houver um proxy interceptador.")
        return None

def create_vector_store(texts: List[str], metadatas: List[Dict[str, Any]], embeddings_model: OpenAIEmbeddings, collection_name: str = "tcrm_copilot_kb"):
    """
    Cria ou atualiza um banco de vetores ChromaDB com os chunks de texto e metadados.
    """
    if not embeddings_model:
        print("Modelo de embeddings não inicializado. Não é possível criar o vector store.")
        return None

    collection_path = os.path.join(CHROMA_DB_DIRECTORY, collection_name)
    if os.path.exists(collection_path):
        print(f"Removendo coleção ChromaDB existente em: {collection_path}")
        shutil.rmtree(collection_path)

    print(f"Criando/atualizando ChromaDB com {len(texts)} documentos...")
    vector_store = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings_model,
        persist_directory=CHROMA_DB_DIRECTORY,
        collection_name=collection_name
    )
    # Com Chroma 0.4.x+, .persist() não é mais necessário e gera warning.
    # Mas se você estiver em uma versão anterior ou quiser ser explícito, pode manter.
    # Por agora, vamos manter, mas esteja ciente do warning.
    # vector_store.persist() # Removido para evitar o warning de depreciação
    print("ChromaDB criado e persistido com sucesso!")
    return vector_store

def build_knowledge_base_from_documents(document_paths: List[str], collection_name: str = "tcrm_copilot_kb"):
    """
    Orquestra a extração de texto, chunking, embedding e armazenamento no ChromaDB
    a partir de uma lista de caminhos de documentos, enriquecendo metadados.
    """
    embeddings_model = initialize_embedding_model()
    if not embeddings_model:
        return None

    all_chunks_content = []
    all_metadatas = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    print(f"Processando {len(document_paths)} documentos...")
    for doc_path in document_paths:
        file_name = os.path.basename(doc_path)
        print(f"Extraindo texto e metadados de {file_name}...")
        
        # Recebendo duas saídas do parser
        full_text, content_specific_metadata = extract_text_from_file_unstructured(doc_path)
        
        # Extrai o document_id do nome do arquivo
        document_id = extract_doc_id_from_filename(file_name)
        if document_id:
            print(f" - ID do Documento (filename) detectado: {document_id}")
        else:
            print(f" - Nenhum ID de Documento (ex: MIT041) detectado no nome do arquivo '{file_name}'.")

        if full_text:
            # Metadados base para os chunks (vindos do nome do arquivo e tipo)
            base_metadata = {
                'source': file_name,
                'filename': file_name,
                'source_type': 'document'
            }
            if document_id:
                base_metadata['document_id'] = document_id

            # Mescla os metadados base com os metadados extraídos do conteúdo.
            # Metadados do conteúdo têm prioridade se houver conflito de chaves.
            combined_metadata = {**base_metadata, **content_specific_metadata}
            
            # Passamos os metadados combinados para cada Document gerado
            doc_chunk_objects = text_splitter.create_documents([full_text], metadatas=[combined_metadata])
            
            for chunk_obj in doc_chunk_objects:
                all_chunks_content.append(chunk_obj.page_content)
                all_metadatas.append(chunk_obj.metadata) # O chunk_obj.metadata já contém tudo mesclado
            print(f" - {len(doc_chunk_objects)} chunks gerados para {file_name} com metadados enriquecidos.")
        else:
            print(f" - Nenhum texto extraído de {file_name}. Pulando.")

    if not all_chunks_content:
        print("Nenhum chunk de texto disponível para criar o banco de conhecimento.")
        return None

    vector_store = create_vector_store(all_chunks_content, all_metadatas, embeddings_model, collection_name)
    return vector_store

# --- Bloco de Teste Rápido ---
if __name__ == "__main__":
    print("--- Iniciando construção da Base de Conhecimento ---")

    current_dir = os.path.dirname(__file__)
    raw_documents_path = os.path.abspath(os.path.join(current_dir, '../../data/raw_documents'))
    os.makedirs(raw_documents_path, exist_ok=True)

    doc_paths = []
    # Incluindo todos os documentos .pdf, .docx, .txt da pasta raw_documents
    for file_name in os.listdir(raw_documents_path):
        if file_name.endswith(('.pdf', '.docx', '.txt')):
            doc_paths.append(os.path.join(raw_documents_path, file_name))

    knowledge_base = build_knowledge_base_from_documents(doc_paths, collection_name="tcrm_copilot_poc_kb")

    if knowledge_base:
        print("\nBase de Conhecimento RAG construída com sucesso!")
        print(f"ChromaDB persistido em: {CHROMA_DB_DIRECTORY}")
        
        print("\n--- Realizando busca de teste com filtro de metadados ---")
        # Exemplo de query para verificar os novos metadados
        query_with_filter = "Qual o coordenador da TOTVS para o projeto SCENS?"
        # Filtro combinado CORRIGIDO para usar o operador $and do ChromaDB
        filter_metadata = {
            "$and": [
                {"document_id": "MIT041"},
                {"client_name": "SCENS INDUSTRIA E COMERCIO DE FRAGRANCIAS LTDA"}
            ]
        }
        print(f"Buscando: '{query_with_filter}' com filtro: {filter_metadata}")
        
        try:
            results_filtered = knowledge_base.similarity_search(query_with_filter, k=3, filter=filter_metadata)
            if results_filtered:
                print("\nResultados da busca filtrada:")
                for i, doc in enumerate(results_filtered):
                    print(f"--- Documento Filtrado {i+1} ---")
                    print(f"Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"Document ID: {doc.metadata.get('document_id', 'N/A')}")
                    print(f"Client Name: {doc.metadata.get('client_name', 'N/A')}")
                    print(f"TOTVS Coordinator: {doc.metadata.get('totvs_coordinator', 'N/A')}")
                    print(f"Content (parcial): {doc.page_content[:200]}...")
                    print("-" * 20)
            else:
                print("Nenhum resultado encontrado para a busca filtrada com esses critérios.")
        except Exception as e:
            print(f"Erro ao realizar busca filtrada: {e}")
            print("Verifique a documentação do ChromaDB para filtros ou se os metadados foram corretamente indexados.")

        print("\n--- Fim da Construção da Base de Conhecimento ---")
