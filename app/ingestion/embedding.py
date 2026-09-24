import requests
import json
import os
from pathlib import Path

from ..core.config import OLLAMA_BASE_URL


EMBEDDING_MODEL = "nomic-embed-text"


def gerar_embedding(texto):
    """
    Gera o embedding de um texto usando Ollama.
    """

    url = f"{OLLAMA_BASE_URL}/api/embeddings"

    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": texto
    }

    resposta = requests.post(
        url,
        json=payload,
        timeout=120
    )

    resposta.raise_for_status()

    resultado = resposta.json()

    return resultado["embedding"]


def gerar_embeddings_chunks(chunks):
    """
    Gera embeddings para todos os chunks.

    Cada chunk recebe uma nova chave:
        "embedding"
    """

    chunks_com_embeddings = []

    total = len(chunks)

    for i, chunk in enumerate(chunks, start=1):

        embedding = gerar_embedding(
            chunk["texto"]
        )

        chunk_com_embedding = {
            **chunk,
            "embedding": embedding
        }

        chunks_com_embeddings.append(
            chunk_com_embedding
        )

        print(f"Embedding {i}/{total}")

    return chunks_com_embeddings


def salvar_embeddings(
    chunks,
    caminho="data/chunks_embeddings.json"
):
    """
    Salva os chunks e seus embeddings em um arquivo JSON.
    """

    os.makedirs(
        os.path.dirname(caminho),
        exist_ok=True
    )

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        json.dump(
            chunks,
            arquivo,
            ensure_ascii=False
        )

    print(
        f"\nEmbeddings salvos em: {caminho}"
    )


# ============================================================
# TESTE
# ============================================================

if __name__ == "__main__":

    from .document_loader import carregar_pdf
    from .chunker import criar_chunks

    caminho = (
        r"documents\CRI-Be-Brave-Termo-de-Securitizacao-v.assinada.pdf"
    )

    paginas = carregar_pdf(caminho)

    nome_arquivo = Path(caminho).name

    chunks = criar_chunks(
        paginas,
        nome_arquivo=nome_arquivo
    )

    print("=" * 70)
    print("GERAÇÃO DE EMBEDDINGS")
    print("=" * 70)

    print(
        f"\nQuantidade de chunks: {len(chunks)}"
    )

    chunks_com_embeddings = gerar_embeddings_chunks(
        chunks
    )

    salvar_embeddings(
        chunks_com_embeddings
    )

    primeiro = chunks_com_embeddings[0]

    print("\n" + "=" * 70)
    print("VALIDAÇÃO")
    print("=" * 70)

    print(
        f"\nChunk ID: {primeiro['chunk_id']}"
    )

    print(
        f"Documento: {primeiro['documento']}"
    )

    print(
        f"Página: {primeiro['pagina']}"
    )

    print(
        f"Caracteres: {len(primeiro['texto'])}"
    )

    print(
        f"Dimensões do embedding: "
        f"{len(primeiro['embedding'])}"
    )

    print("\nPrimeiros 10 valores do embedding:")

    print(
        primeiro["embedding"][:10]
    )