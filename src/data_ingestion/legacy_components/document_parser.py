"""
Módulo LEGADO para extração de texto e metadados de documentos.

Este módulo foi desenvolvido na fase inicial do projeto TCRM Copilot
com o objetivo de extrair texto de arquivos diversos (PDF, DOCX, etc.)
e, principalmente, realizar uma extração heurística de metadados
estruturados a partir do conteúdo do documento, focando na seção "AMBIENTAÇÃO".

ATENÇÃO: Este arquivo é considerado LEGADO e não é mais utilizado na arquitetura principal
do sistema. Sua funcionalidade foi substituída e aprimorada pelo módulo
`src/data_ingestion/incremental_ingestor.py`, que utiliza loaders mais
integrados da LangChain e uma abordagem mais robusta para metadados de arquivo.

Ele é mantido para fins de referência histórica e para demonstrar a evolução
das estratégias de extração de dados no projeto.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element, Title, PageBreak # Importar PageBreak para melhor heurística

# --- Configuração de Logging ---
# Configura o logger para este módulo.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_file_unstructured(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extrai o texto completo de um arquivo e tenta identificar metadados estruturados
    a partir de seu conteúdo, especificamente da seção "AMBIENTAÇÃO", usando a
    biblioteca 'unstructured'.

    Esta função representa uma abordagem inicial para enriquecer os metadados dos documentos
    ao processá-los. Ela demonstra a capacidade de ir além dos metadados de arquivo
    e inferir informações importantes do próprio texto.

    Args:
        file_path (str): O caminho completo para o arquivo a ser processado.

    Returns:
        Tuple[str, Dict[str, Any]]: Uma tupla contendo:
                                    - O texto completo extraído do documento (str).
                                    - Um dicionário de metadados extraídos do *conteúdo*
                                      (ex: nome do cliente, código do projeto, etc.).
                                      Retorna um dicionário vazio se nenhum metadado for encontrado
                                      ou em caso de erro na extração.
    Raises:
        (Não levanta explicitamente, mas loga erros internos e retorna vazios)
    """
    if not os.path.exists(file_path):
        logger.error(f"Erro: Arquivo não encontrado - {file_path}")
        return "", {}

    content_specific_metadata = {} # Dicionário para armazenar metadados extraídos do conteúdo

    try:
        # Usa unstructured para particionar o documento em elementos.
        # 'languages=["por"]' é adicionado para indicar o idioma, melhorando a precisão
        # e suprimindo warnings de detecção de idioma.
        elements: List[Element] = partition(filename=file_path, languages=["por"])
        
        # --- LÓGICA DE EXTRAÇÃO HEURÍSTICA PARA METADADOS DA SEÇÃO "AMBIENTAÇÃO" ---
        # O objetivo é encontrar o bloco de texto sob o título "AMBIENTAÇÃO"
        # e extrair pares chave-valor específicos de dentro dele.
        ambientacao_elements_text = []
        found_ambientacao_title = False
        
        # 1. Coleta o texto pertencente à seção "AMBIENTAÇÃO".
        for el in elements:
            # Verifica se o elemento atual é o título "AMBIENTAÇÃO" (case-insensitive).
            if isinstance(el, Title) and el.text.strip().upper() == "AMBIENTAÇÃO":
                found_ambientacao_title = True
                logger.debug(f"Título 'AMBIENTAÇÃO' encontrado em {file_path}.")
                continue # Não inclui o próprio título no texto da seção.

            if found_ambientacao_title:
                # Define uma heurística para determinar o fim da seção "AMBIENTAÇÃO".
                # Coleta elementos até encontrar outro título principal ou uma quebra de página,
                # indicando o início de uma nova seção.
                ending_titles = [
                    "HISTÓRICO DE REVISÕES", "SUMÁRIO", "OBJETIVO",
                    "1. DEFINIÇÃO DE ESCOPO TÉCNICO", "ASSINATURA E ACEITE",
                    "2. PREMISSAS DO ESCOPO TÉCNICO E DE NEGÓCIO",
                    "HISTÓRICO DE VERSÕES" # Adicionado para cobrir variantes
                ]
                # Verifica se o elemento atual é um título que marca o fim da seção Ambientação.
                if isinstance(el, Title) and el.text.strip().upper() in [t.upper() for t in ending_titles]:
                    logger.debug(f"Fim da seção 'AMBIENTAÇÃO' detectado por título: '{el.text}'")
                    break
                # Verifica se é uma quebra de página, que também pode indicar o fim de uma seção lógica.
                if isinstance(el, PageBreak) or el.category == "PageBreak":
                    logger.debug(f"Fim da seção 'AMBIENTAÇÃO' detectado por quebra de página.")
                    break

                if el.text and el.text.strip():
                    ambientacao_elements_text.append(el.text.strip())
        
        # Combina os fragmentos de texto coletados da seção "AMBIENTAÇÃO" em um único bloco.
        ambientacao_full_block_text = " ".join(ambientacao_elements_text)
        logger.debug(f"Bloco de texto da Ambientação coletado (primeiros 200 chars): '{ambientacao_full_block_text[:200]}...'")

        # 2. Processa o bloco de texto da "AMBIENTAÇÃO" para extrair pares chave-valor.
        if ambientacao_full_block_text:
            # Mapeamento das chaves textuais encontradas no documento para nomes de metadados padronizados.
            known_keys_map = {
                "Nome do cliente": "client_name",
                "Código de cliente": "client_code",
                "Nome do projeto": "project_name_full",
                "Código do projeto": "project_code_crm", # Este foi um insight para diferenciar do MIT041
                "Segmento cliente": "client_segment",
                "Unidade TOTVS": "totvs_unit",
                "Data Projeto": "project_date",
                "Proposta comercial": "commercial_proposal",
                "Gerente/Coordenador TOTVS": "totvs_coordinator",
                "Gerente/Coordenador cliente": "client_coordinator"
            }
            
            # Ordena as chaves conhecidas por comprimento decrescente.
            # Isso é crucial para evitar correspondências parciais (ex: "Nome do cliente" ser
            # encontrado antes de "Nome do cliente completo"). Garante que a chave mais longa
            # e específica seja priorizada.
            sorted_keys = sorted(known_keys_map.keys(), key=len, reverse=True)
            
            # Constrói uma expressão regular para encontrar qualquer uma das chaves conhecidas,
            # seguida por um dois pontos e espaços opcionais.
            key_regex_pattern = r'(' + '|'.join(re.escape(k) for k in sorted_keys) + r'):\s*'

            # Usa `re.finditer` para encontrar todas as ocorrências das chaves e suas posições.
            matches = list(re.finditer(key_regex_pattern, ambientacao_full_block_text))
            
            for i, match in enumerate(matches):
                key_found_in_text = match.group(1).strip() # A chave literal encontrada (e.g., "Nome do cliente")
                start_of_value = match.end()               # Posição onde o valor da chave começa

                # Determina o fim do valor: ou o início da próxima chave encontrada,
                # ou o final do bloco de texto se for a última chave.
                end_of_value = len(ambientacao_full_block_text)
                if i + 1 < len(matches):
                    end_of_value = matches[i+1].start()
                
                value = ambientacao_full_block_text[start_of_value:end_of_value].strip()
                
                # Mapeia a chave encontrada para a chave de metadado padronizada.
                standardized_key = known_keys_map.get(key_found_in_text)
                if standardized_key and value: # Adiciona ao dicionário se a chave for mapeada e o valor não estiver vazio.
                    content_specific_metadata[standardized_key] = value
                    logger.debug(f"Metadado extraído: '{standardized_key}': '{value}'")

        # --- FIM DA LÓGICA DE EXTRAÇÃO DE METADADOS DO CONTEÚDO ---

        # Concatena o texto de todos os elementos extraídos para formar o texto completo do documento.
        full_text = "\n\n".join([str(el) for el in elements if el.text and el.text.strip()])
        logger.info(f"Texto e metadados extraídos com sucesso de '{os.path.basename(file_path)}'.")

        return full_text, content_specific_metadata

    except Exception as e:
        logger.error(f"Erro ao extrair texto e metadados do arquivo '{file_path}' usando unstructured: {e}", exc_info=True)
        return "", {}

# --- Bloco de Teste Rápido (Executado apenas quando o script é rodado diretamente) ---
if __name__ == "__main__":
    logger.setLevel(logging.DEBUG) # Aumenta o nível de log para debug em teste local
    logger.info("--- Iniciando teste do LEGADO document_parser.py com unstructured ---")

    current_dir = os.path.dirname(__file__)
    # Caminho para uma pasta de documentos brutos, um nível acima da raiz do projeto, depois em 'data/documents'
    # Ajustado para refletir a nova estrutura e a pasta real de documentos de teste.
    raw_documents_path = os.path.abspath(os.path.join(current_dir, '../../../data/documents'))
    os.makedirs(raw_documents_path, exist_ok=True) # Garante que o diretório de destino exista

    logger.info(f"Procurando documentos de teste na pasta: {raw_documents_path}")

    # Use um dos documentos DOCX de teste que você já possui.
    # Exemplo: "[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx"
    test_docx_file = os.path.join(raw_documents_path, '[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx')
    
    if os.path.exists(test_docx_file):
        logger.info(f"\nExtraindo de DOCX: {os.path.basename(test_docx_file)}")
        docx_content, docx_metadata = extract_text_from_file_unstructured(test_docx_file)
        logger.info(f"Conteúdo (primeiros 500 caracteres):\n---\n{docx_content[:500]}...\n---")
        logger.info(f"Metadados extraídos do conteúdo:\n---\n{docx_metadata}\n---")
        
        # Salvando a saída para inspeção manual
        output_filename = "legacy_document_parser_output.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(f"--- Output do LEGADO document_parser.py ---\n\n")
            f.write(f"Arquivo Processado: {os.path.basename(test_docx_file)}\n\n")
            f.write(f"METADADOS EXTRAÍDOS DO CONTEÚDO:\n{docx_metadata}\n\n")
            f.write(f"TEXTO COMPLETO (Primeiros 1000 caracteres):\n{docx_content[:1000]}...\n")
        logger.info(f"Conteúdo completo do DOCX e metadados salvos em '{output_filename}' para revisão.")
    else:
        logger.warning(f"ATENÇÃO: Arquivo de teste DOCX não encontrado em '{test_docx_file}'. "
                       f"Por favor, certifique-se de que o arquivo '[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx' "
                       f"esteja presente em '{raw_documents_path}' para testar a extração.")

    logger.info("--- Fim do Teste do LEGADO document_parser.py ---")
    logger.info("Verifique os logs e o arquivo de saída para entender a lógica de extração.")
