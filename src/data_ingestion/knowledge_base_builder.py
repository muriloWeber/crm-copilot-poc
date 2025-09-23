# src/data_ingestion/knowledge_base_builder.py

import os
import shutil
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_parser import extract_text_from_file_unstructured # Importa o nosso parser aprimorado
import httpx # <<<--- NOVA IMPORTAÇÃO NECESSÁRIA

# Configurações globais para o RAG (podem ser movidas para um arquivo de config depois)
# Importante: O modelo de embedding está definido no PoC como text-embedding-ada-002
# Para gpt-4o-mini ou outros modelos, geralmente 'text-embedding-ada-002' ou 'text-embedding-3-small' são boas opções de embedding.
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
CHROMA_DB_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/chroma_db'))
CHUNK_SIZE = 1000 # Tamanho ideal de chunk depende do modelo e da natureza do texto
CHUNK_OVERLAP = 200 # Sobreposição entre chunks para não perder contexto entre eles

def initialize_embedding_model():
    """
    Inicializa o modelo de embeddings da OpenAI.
    Certifique-se de que a variável de ambiente OPENAI_API_KEY esteja configurada.
    Em ambientes corporativos com proxy interceptador, pode ser necessário desabilitar
    a verificação SSL com 'verify=False' no cliente HTTP.
    """
    try:
        openai_api_base = os.getenv("OPENAI_API_BASE")
        
        # --- CORREÇÃO APLICADA: Configurando httpx.Client diretamente ---
        # Cria um cliente httpx com a verificação SSL desabilitada.
        # Isso é o que o 'openai' (e por consequência o 'langchain_openai') usa internamente.
        custom_http_client = httpx.Client(verify=False)

        # Passa a base_url diretamente e o cliente httpx configurado para o OpenAIEmbeddings.
        # Assim, evitamos o 'client_kwargs' que estava causando o TypeError.
        return OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            base_url=openai_api_base if openai_api_base else None,
            http_client=custom_http_client
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

    # Verifica se já existe um diretório Chroma para a coleção e o remove para recriar (para PoC)
    # Em produção, você adicionaria novos documentos ou faria atualizações incrementais.
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
    vector_store.persist()
    print("ChromaDB criado e persistido com sucesso!")
    return vector_store

def build_knowledge_base_from_documents(document_paths: List[str], collection_name: str = "tcrm_copilot_kb"):
    """
    Orquestra a extração de texto, chunking, embedding e armazenamento no ChromaDB
    a partir de uma lista de caminhos de documentos.
    """
    embeddings_model = initialize_embedding_model()
    if not embeddings_model:
        return None

    all_chunks = []
    all_metadatas = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len, # Usa len() para contar caracteres, em vez de tokens
        add_start_index=True # Adiciona o índice de início do chunk no documento original
    )

    print(f"Processando {len(document_paths)} documentos...")
    for doc_path in document_paths:
        file_name = os.path.basename(doc_path)
        print(f"Extraindo texto de {file_name}...")
        full_text = extract_text_from_file_unstructured(doc_path)

        if full_text:
            # Divide o texto em chunks
            chunks = text_splitter.create_documents([full_text]) # LangChain Document objects

            for i, chunk in enumerate(chunks):
                # Adiciona metadados conforme especificado no PoC
                metadata = chunk.metadata
                metadata['source'] = file_name
                metadata['filename'] = file_name
                metadata['source_type'] = 'document'
                # O 'page' ou 'field' pode ser mais complexo com unstructured,
                # unstructured.partition pode retornar elementos com page_number,
                # mas para simplificar aqui, deixamos o filename como identificador principal.
                # Poderíamos refinar isso para incluir o page_number se unstructured fornecer.

                all_chunks.append(chunk.page_content)
                all_metadatas.append(metadata)
            print(f" - {len(chunks)} chunks gerados para {file_name}")
        else:
            print(f" - Nenhum texto extraído de {file_name}. Pulando.")

    if not all_chunks:
        print("Nenhum chunk de texto disponível para criar o banco de conhecimento.")
        return None

    vector_store = create_vector_store(all_chunks, all_metadatas, embeddings_model, collection_name)
    return vector_store

# --- Bloco de Teste Rápido ---
if __name__ == "__main__":
    print("--- Iniciando construção da Base de Conhecimento ---")

    # Garante que a pasta de documentos brutos exista
    current_dir = os.path.dirname(__file__)
    raw_documents_path = os.path.abspath(os.path.join(current_dir, '../../data/raw_documents'))
    os.makedirs(raw_documents_path, exist_ok=True)

    # Crie alguns arquivos de teste ou use os existentes que você salvou
    # Exemplo de documentos para a PoC (coloque seus arquivos reais aqui)
    # Por exemplo: seu DOCX da SCENS e um PDF de exemplo
    doc_paths = []
    for file_name in os.listdir(raw_documents_path):
        if file_name.endswith(('.pdf', '.docx', '.txt')):
            doc_paths.append(os.path.join(raw_documents_path, file_name))

    # O ChromaDB será criado em '../../data/chroma_db'
    knowledge_base = build_knowledge_base_from_documents(doc_paths, collection_name="tcrm_copilot_poc_kb")

    if knowledge_base:
        print("\nBase de Conhecimento RAG construída com sucesso!")
        print(f"ChromaDB persistido em: {CHROMA_DB_DIRECTORY}")

        # Exemplo de como fazer uma busca simples (para verificar se funciona)
        query = "Qual o nome do cliente do projeto Scens?"
        print(f"\nRealizando uma busca de teste: '{query}'")
        results = knowledge_base.similarity_search(query, k=2) # Busca os 2 chunks mais relevantes

        if results:
            print("\nResultados da busca:")
            for i, doc in enumerate(results):
                print(f"--- Documento {i+1} ---")
                print(f"Source: {doc.metadata.get('source', 'N/A')}")
                print(f"Content (parcial): {doc.page_content[:200]}...")
                print("-" * 20)
        else:
            print("Nenhum resultado encontrado para a busca de teste.")
    else:
        print("Falha na construção da Base de Conhecimento RAG.")

    print("\n--- Fim da Construção da Base de Conhecimento ---")
