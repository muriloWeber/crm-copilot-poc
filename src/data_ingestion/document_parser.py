# src/data_ingestion/document_parser.py

import os
from typing import List
from unstructured.partition.auto import partition # Função universal do unstructured
from unstructured.documents.elements import Element

def extract_text_from_file_unstructured(file_path: str) -> str:
    """
    Extrai o texto de qualquer tipo de arquivo suportado usando a biblioteca unstructured.
    A função partition do unstructured tenta extrair elementos de texto, tabelas, etc.
    Para esta PoC, concatenamos o texto de todos os elementos.
    """
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo não encontrado - {file_path}")
        return ""

    try:
        # A função partition detecta automaticamente o tipo de arquivo e aplica o processamento adequado.
        # Estamos passando o file_path, e unstructured cuidará da leitura.
        elements: List[Element] = partition(filename=file_path)

        # Concatena o texto de todos os elementos extraídos.
        # Elementos de tabela podem ser processados de forma mais sofisticada depois,
        # mas para começar, ter o texto é o suficiente.
        full_text = "\n\n".join([str(el) for el in elements if el.text.strip()])

        return full_text

    except Exception as e:
        print(f"Erro ao extrair texto do arquivo '{file_path}' usando unstructured: {e}")
        # Retorna uma string vazia em caso de erro para não quebrar o pipeline.
        return ""

# --- Bloco de Teste Rápido (Executado apenas quando o script é rodado diretamente) ---
if __name__ == "__main__":
    print("--- Testando document_parser.py com unstructured ---")

    # Garante que a pasta de documentos brutos exista para nossos testes
    # O caminho deve ser relativo ao diretório raiz do projeto para o teste funcionar.
    current_dir = os.path.dirname(__file__)
    raw_documents_path = os.path.abspath(os.path.join(current_dir, '../../data/raw_documents'))
    os.makedirs(raw_documents_path, exist_ok=True)

    print(f"Verificando a pasta de documentos: {raw_documents_path}")

    # 1. Teste com arquivo TXT
    dummy_txt_file = os.path.join(raw_documents_path, 'teste_copilot.txt')
    with open(dummy_txt_file, 'w', encoding='utf-8') as f:
        f.write("Este é um documento de texto simples para teste.\n")
        f.write("Contém algumas linhas de exemplo para verificar a extração.\n")
        f.write("O TCRM Copilot promete automatizar tarefas e gerar inteligência.\n")
        f.write("--- Tabela Simples (para ver como unstructured lida) ---\n")
        f.write("Item | Quantidade | Preço\n")
        f.write("------------------------\n")
        f.write("Caneta | 10 | 2.50\n")
        f.write("Lápis | 5 | 1.00\n")

    print(f"\nExtraindo de TXT: {dummy_txt_file}")
    txt_content = extract_text_from_file_unstructured(dummy_txt_file)
    print(f"Conteúdo:\n---\n{txt_content}\n---")
    os.remove(dummy_txt_file) # Limpa o arquivo de teste

    # 2. Teste com arquivo PDF (requer um PDF de verdade na pasta)
    # Use o documento "[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx"
    # ou um PDF similar que contenha tabelas e cabeçalhos.
    # Salve-o como 'scens_escopo.pdf' (ou o nome original) dentro de 'data/raw_documents'
    dummy_pdf_file = os.path.join(raw_documents_path, 'scens_escopo.pdf')
    print(f"\nExtraindo de PDF: {dummy_pdf_file}")
    if os.path.exists(dummy_pdf_file):
        pdf_content = extract_text_from_file_unstructured(dummy_pdf_file)
        print(f"Conteúdo (primeiros 500 caracteres):\n---\n{pdf_content[:500]}...\n---")
        # Para inspecionar melhor a saída, você pode querer salvar para um arquivo:
        with open("pdf_extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(pdf_content)
        print("Conteúdo completo do PDF salvo em 'pdf_extracted_text.txt'")
    else:
        print(f"ATENÇÃO: Crie um arquivo PDF '{os.path.basename(dummy_pdf_file)}' em '{raw_documents_path}' para testar a extração de PDF.")

    # 3. Teste com arquivo DOCX (requer um DOCX de verdade na pasta)
    # Coloque o documento "[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx"
    # dentro de 'data/raw_documents'.
    dummy_docx_file = os.path.join(raw_documents_path, '[Scens] - Escopo Técnico TOTVS CRM Gestão de Clientes - MIT041 V1.0 13-05-2025.docx')
    print(f"\nExtraindo de DOCX: {dummy_docx_file}")
    if os.path.exists(dummy_docx_file):
        docx_content = extract_text_from_file_unstructured(dummy_docx_file)
        print(f"Conteúdo (primeiros 500 caracteres):\n---\n{docx_content[:500]}...\n---")
        # Para inspecionar melhor a saída:
        with open("docx_extracted_text.txt", "w", encoding="utf-8") as f:
            f.write(docx_content)
        print("Conteúdo completo do DOCX salvo em 'docx_extracted_text.txt'")
    else:
        print(f"ATENÇÃO: Coloque o arquivo DOCX '{os.path.basename(dummy_docx_file)}' em '{raw_documents_path}' para testar a extração de DOCX.")

    print("\n--- Fim do Teste com unstructured ---")
    print("Verifique os outputs para ver como o unstructured está lidando com a estrutura dos documentos.")
