# src/data_ingestion/document_parser.py

import os
import re
from typing import List, Dict, Any, Tuple
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element, Title # Importar Title para identificar títulos

def extract_text_from_file_unstructured(file_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Extrai o texto de qualquer tipo de arquivo suportado usando a biblioteca unstructured
    e também tenta extrair metadados estruturados do CONTEÚDO do documento,
    focando na seção "AMBIENTAÇÃO".

    Retorna uma tupla: (texto_completo, dicionario_de_metadados_do_conteudo)
    """
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo não encontrado - {file_path}")
        return "", {}

    content_specific_metadata = {} # Dicionário para armazenar metadados do conteúdo

    try:
        # Adicionado 'languages=["por"]' para indicar o idioma, suprimindo o warning
        elements: List[Element] = partition(filename=file_path, languages=["por"])

        # --- LÓGICA PARA EXTRAIR METADADOS DO CONTEÚDO (Ex: seção AMBIENTAÇÃO) ---
        ambientacao_elements_text = []
        found_ambientacao_title = False
        
        # 1. Coletar todo o texto da seção AMBIENTAÇÃO
        for el in elements:
            # Verifica se encontramos o título "AMBIENTAÇÃO" (case-insensitive)
            if isinstance(el, Title) and el.text.strip().upper() == "AMBIENTAÇÃO":
                found_ambientacao_title = True
                continue # Pula o próprio título "AMBIENTAÇÃO"

            if found_ambientacao_title:
                # Heurística para parar de coletar quando a seção AMBIENTAÇÃO termina
                # Verifica se o elemento atual é outro Título principal conhecido
                # ou uma quebra de página (PageBreak).
                # Usamos uma lista mais completa de títulos que podem encerrar a seção
                ending_titles = ["HISTÓRICO DE REVISÕES", "SUMÁRIO", "OBJETIVO", "1. DEFINIÇÃO DE ESCOPO TÉCNICO", "ASSINATURA E ACEITE", "2. PREMISSAS DO ESCOPO TÉCNICO E DE NEGÓCIO"]
                if isinstance(el, Title) and el.text.strip().upper() in ending_titles:
                    break # Parar de coletar dados da Ambientação
                if el.category == "PageBreak":
                    break

                if el.text and el.text.strip():
                    ambientacao_elements_text.append(el.text.strip())
        
        # Combinar os textos coletados da Ambientação em um único bloco para parsing
        # Usamos espaço como separador, pois o unstructured pode ter concatenado tudo.
        ambientacao_full_block_text = " ".join(ambientacao_elements_text)

        # 2. Processar o bloco de texto para extrair pares chave-valor
        if ambientacao_full_block_text:
            # Mapeamento das chaves do documento para nomes de metadados padronizados
            known_keys_map = {
                "Nome do cliente": "client_name",
                "Código de cliente": "client_code",
                "Nome do projeto": "project_name_full",
                "Código do projeto": "project_code_crm",
                "Segmento cliente": "client_segment",
                "Unidade TOTVS": "totvs_unit",
                "Data Projeto": "project_date",
                "Proposta comercial": "commercial_proposal",
                "Gerente/Coordenador TOTVS": "totvs_coordinator",
                "Gerente/Coordenador cliente": "client_coordinator"
            }
            
            # Ordenar as chaves conhecidas por comprimento (maior primeiro)
            # Isso evita que "Nome do cliente" seja encontrado antes de "Nome do cliente longo e complexo"
            sorted_keys = sorted(known_keys_map.keys(), key=len, reverse=True)
            
            # Construir uma regex para encontrar qualquer uma das chaves conhecidas
            # seguida por um dois pontos e um espaço (ex: "Chave: ")
            key_regex_pattern = r'(' + '|'.join(re.escape(k) for k in sorted_keys) + r'):\s*'

            # Usar re.finditer para encontrar todas as ocorrências das chaves e suas posições no texto
            matches = list(re.finditer(key_regex_pattern, ambientacao_full_block_text))
            
            for i, match in enumerate(matches):
                key_found_in_text = match.group(1).strip() # A chave completa encontrada (ex: "Nome do cliente")
                start_of_value = match.end() # Posição no texto onde o valor começa (depois de "Chave: ")

                # Determinar o fim do valor:
                # É o início da próxima chave encontrada ou o final do bloco de texto se for a última chave.
                end_of_value = len(ambientacao_full_block_text)
                if i + 1 < len(matches):
                    end_of_value = matches[i+1].start()
                
                value = ambientacao_full_block_text[start_of_value:end_of_value].strip()
                
                # Mapear a chave encontrada para a chave de metadado padronizada
                standardized_key = known_keys_map.get(key_found_in_text)
                if standardized_key and value: # Se a chave for mapeada e o valor não estiver vazio
                    content_specific_metadata[standardized_key] = value

        # --- FIM DA LÓGICA DE EXTRAÇÃO DE METADADOS DO CONTEÚDO ---

        # Concatenar o texto de todos os elementos extraídos para o full_text principal.
        full_text = "\n\n".join([str(el) for el in elements if el.text.strip()])

        return full_text, content_specific_metadata

    except Exception as e:
        print(f"Erro ao extrair texto e metadados do arquivo '{file_path}' usando unstructured: {e}")
        return "", {}

# --- Bloco de Teste Rápido (Executado apenas quando o script é rodado diretamente) ---
if __name__ == "__main__":
    print("--- Testando document_parser.py com unstructured ---")

    current_dir = os.path.dirname(__file__)
    raw_documents_path = os.path.abspath(os.path.join(current_dir, '../../data/raw_documents'))
    os.makedirs(raw_documents_path, exist_ok=True)

    print(f"Verificando a pasta de documentos: {raw_documents_path}")

    # Crie um arquivo DOCX de teste na pasta 'data/raw_documents'
    # Use o documento "[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx"
    dummy_docx_file = os.path.join(raw_documents_path, '[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx')
    print(f"\nExtraindo de DOCX: {dummy_docx_file}")
    if os.path.exists(dummy_docx_file):
        docx_content, docx_metadata = extract_text_from_file_unstructured(dummy_docx_file)
        print(f"Conteúdo (primeiros 500 caracteres):\n---\n{docx_content[:500]}...\n---")
        print(f"Metadados extraídos do conteúdo:\n---\n{docx_metadata}\n---")
        
        with open("docx_extracted_text_and_metadata.txt", "w", encoding="utf-8") as f:
            f.write(f"FULL TEXT:\n{docx_content}\n\nMETADATA:\n{docx_metadata}")
        print("Conteúdo completo do DOCX e metadados salvos em 'docx_extracted_text_and_metadata.txt'")
    else:
        print(f"ATENÇÃO: Coloque o arquivo DOCX '{os.path.basename(dummy_docx_file)}' em '{raw_documents_path}' para testar a extração de DOCX e metadados.")

    print("\n--- Fim do Teste com unstructured ---")
    print("Verifique os outputs para ver como o unstructured está lidando com a estrutura dos documentos e a extração de metadados.")
