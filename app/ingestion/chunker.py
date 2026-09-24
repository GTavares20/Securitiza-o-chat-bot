from pathlib import Path

from .document_loader import carregar_pdf
from .text_cleaner import limpar_texto


def dividir_texto(texto, tamanho_chunk=1000, sobreposicao=200):
    """
    Divide um texto em chunks com sobreposição.

    Parâmetros:
        texto: texto que será dividido
        tamanho_chunk: tamanho máximo de cada chunk
        sobreposicao: quantidade de caracteres repetidos
                      entre chunks consecutivos

    Retorna:
        Lista de textos.
    """

    if not texto:
        return []

    if sobreposicao >= tamanho_chunk:
        raise ValueError(
            "A sobreposição deve ser menor que o tamanho do chunk."
        )

    chunks = []

    inicio = 0
    tamanho_texto = len(texto)

    while inicio < tamanho_texto:

        fim = inicio + tamanho_chunk

        trecho = texto[inicio:fim].strip()

        if trecho:
            chunks.append(trecho)

        # Mantém 200 caracteres do chunk anterior
        inicio = fim - sobreposicao

    return chunks


def criar_chunks(paginas, nome_arquivo, tamanho_chunk=1000, sobreposicao=200):
    """
    Cria chunks a partir das páginas do PDF.

    Parâmetros:
        paginas: páginas extraídas do PDF
        nome_arquivo: nome do documento de origem
        tamanho_chunk: tamanho máximo de cada chunk
        sobreposicao: quantidade de caracteres repetidos
                      entre chunks consecutivos

    Retorna:
        Lista de chunks com seus respectivos metadados.
    """

    chunks = []
    chunk_id = 0

    for pagina in paginas:

        numero_pagina = pagina["pagina"]

        # Limpa o texto da página
        texto = limpar_texto(pagina["texto"])

        if not texto:
            continue

        # Divide o texto
        trechos = dividir_texto(
            texto,
            tamanho_chunk=tamanho_chunk,
            sobreposicao=sobreposicao
        )

        # Adiciona metadados
        for trecho in trechos:

            chunks.append({
                "chunk_id": chunk_id,
                "documento": nome_arquivo,
                "pagina": numero_pagina,
                "texto": trecho
            })

            chunk_id += 1

    return chunks


# ============================================================
# TESTE
# ============================================================

if __name__ == "__main__":

    caminho = r"documents\CRI-Be-Brave-Termo-de-Securitizacao-v.assinada.pdf"

    # Obtém apenas o nome do arquivo
    nome_arquivo = Path(caminho).name

    paginas = carregar_pdf(caminho)

    chunks = criar_chunks(
        paginas,
        nome_arquivo=nome_arquivo,
        tamanho_chunk=1000,
        sobreposicao=200
    )

    print("=" * 70)
    print("VALIDAÇÃO DOS CHUNKS")
    print("=" * 70)

    print(f"\nDocumento: {nome_arquivo}")
    print(f"Quantidade de páginas: {len(paginas)}")
    print(f"Quantidade de chunks: {len(chunks)}")

    tamanhos = [
        len(chunk["texto"])
        for chunk in chunks
    ]

    print(f"\nMenor chunk: {min(tamanhos)} caracteres")
    print(f"Maior chunk: {max(tamanhos)} caracteres")

    print(
        f"Tamanho médio: "
        f"{sum(tamanhos) / len(tamanhos):.0f} caracteres"
    )

    chunks_pequenos = [
        chunk
        for chunk in chunks
        if len(chunk["texto"]) < 300
    ]

    print(
        f"\nChunks menores que 300 caracteres: "
        f"{len(chunks_pequenos)}"
    )

    print("\n" + "=" * 70)
    print("PRIMEIROS 3 CHUNKS")
    print("=" * 70)

    for chunk in chunks[:3]:

        print(f"\nCHUNK ID: {chunk['chunk_id']}")
        print(f"DOCUMENTO: {chunk['documento']}")
        print(f"PÁGINA: {chunk['pagina']}")
        print(f"CARACTERES: {len(chunk['texto'])}")

        print("-" * 70)

        print(chunk["texto"][:500])

        print("-" * 70)