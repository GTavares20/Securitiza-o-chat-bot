import json
import re
import unicodedata
import numpy as np

from pathlib import Path


# ============================================================
# CONFIGURAÇÃO
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CAMINHO_EMBEDDINGS = (
    BASE_DIR / "data" / "evaluation" / "chunks_embeddings.json"
)


# ============================================================
# CARREGAR EMBEDDINGS
# ============================================================

def carregar_embeddings():
    """
    Carrega os chunks e seus embeddings do arquivo JSON.
    """

    with open(
        CAMINHO_EMBEDDINGS,
        "r",
        encoding="utf-8"
    ) as arquivo:

        return json.load(arquivo)


# ============================================================
# SIMILARIDADE DE COSSENO
# ============================================================

def similaridade_cosseno(vetor_a, vetor_b):
    """
    Calcula a similaridade de cosseno entre dois vetores.
    """

    vetor_a = np.array(vetor_a)
    vetor_b = np.array(vetor_b)

    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return np.dot(
        vetor_a,
        vetor_b
    ) / (
        norma_a * norma_b
    )


# ============================================================
# NORMALIZAÇÃO DE TEXTO
# ============================================================

def normalizar_texto(texto):
    """
    Normaliza o texto para busca lexical.

    Etapas:
    - transforma em minúsculas
    - remove acentos
    - remove caracteres especiais
    - normaliza espaços
    """

    texto = texto.lower()

    texto = unicodedata.normalize(
        "NFD",
        texto
    )

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    texto = re.sub(
        r"[^a-z0-9\s]",
        " ",
        texto
    )

    texto = re.sub(
        r"\s+",
        " ",
        texto
    ).strip()

    return texto


# Palavras que são muito frequentes em perguntas em português e não ajudam a
# distinguir uma cláusula do documento. Elas eram a principal causa de a busca
# lexical antiga favorecer trechos que só coincidiam com "qual", "é", "do" ou
# "a".
STOPWORDS_BUSCA = {
    "a", "as", "ao", "aos", "da", "das", "de", "do", "dos", "e", "em",
    "o", "os", "na", "nas", "no", "nos", "para", "por", "que", "qual",
    "quais", "um", "uma", "é", "sao", "ser", "sobre"
}


def tokenizar_para_busca(texto):
    """Retorna tokens completos e úteis para a busca lexical."""

    return [
        token
        for token in normalizar_texto(texto).split()
        if token not in STOPWORDS_BUSCA and len(token) > 1
    ]


# ============================================================
# BUSCA SEMÂNTICA
# ============================================================

def buscar_chunks_semantico(
    embedding_pergunta,
    top_k=5
):
    """
    Busca os chunks semanticamente mais semelhantes
    ao embedding da pergunta.
    """

    chunks = carregar_embeddings()

    resultados = []

    for chunk in chunks:

        similaridade = similaridade_cosseno(
            embedding_pergunta,
            chunk["embedding"]
        )

        resultado = {
            **chunk,
            "similaridade": float(similaridade)
        }

        resultados.append(resultado)

    resultados.sort(
        key=lambda x: x["similaridade"],
        reverse=True
    )

    return resultados[:top_k]


# ============================================================
# BUSCA LEXICAL
# ============================================================

def buscar_chunks_lexical(
    pergunta,
    top_k=20
):
    """Busca lexical BM25 usando tokens completos da pergunta.

    BM25 evita que termos genéricos e chunks muito longos dominem o ranking.
    Diferentemente da implementação anterior, ``cri`` não casa dentro de
    outra palavra e ``a``/``do`` não contam como evidência.
    """

    chunks = carregar_embeddings()

    termos = tokenizar_para_busca(pergunta)

    if not termos:
        return []

    documentos = []
    frequencia_documentos = {}

    for chunk in chunks:
        tokens = tokenizar_para_busca(chunk["texto"])
        frequencias = {}

        for token in tokens:
            frequencias[token] = frequencias.get(token, 0) + 1

        documentos.append((chunk, tokens, frequencias))

        for termo in set(termos):
            if termo in frequencias:
                frequencia_documentos[termo] = (
                    frequencia_documentos.get(termo, 0) + 1
                )

    tamanho_medio = sum(len(tokens) for _, tokens, _ in documentos) / len(documentos)
    total_documentos = len(documentos)
    k1 = 1.5
    b = 0.75

    resultados = []

    for chunk, tokens, frequencias in documentos:

        score_lexical = 0.0
        score_estrutura = 0.0
        termos_encontrados = []
        # Para reconhecer rótulos preservamos quebras de linha e ':'; a
        # normalização usada para tokens deliberadamente remove esses sinais.
        texto_para_rotulo = "".join(
            caractere
            for caractere in unicodedata.normalize("NFD", chunk["texto"].lower())
            if unicodedata.category(caractere) != "Mn"
        )

        for termo in set(termos):
            frequencia = frequencias.get(termo, 0)

            if not frequencia:
                continue

            termos_encontrados.append(termo)
            idf = np.log(
                1 + (total_documentos - frequencia_documentos[termo] + 0.5)
                / (frequencia_documentos[termo] + 0.5)
            )
            normalizador = frequencia + k1 * (
                1 - b + b * len(tokens) / tamanho_medio
            )
            score_lexical += idf * (frequencia * (k1 + 1)) / normalizador

            # Em contratos, um termo seguido de ':' em um item ou cláusula é
            # um rótulo (por exemplo, "(ix) Remuneração:"). Isso é evidência
            # mais forte de que o trecho define a informação pedida do que uma
            # menção incidental no corpo do texto.
            padrao_rotulo = (
                rf"(?m)(?:^|\n)\s*(?:\(?[ivxlcdm]+\)?|\d+(?:\.\d+)*|[a-z])?"
                rf"\s*{re.escape(termo)}\s*:"
            )
            if re.search(padrao_rotulo, texto_para_rotulo):
                score_estrutura += 2.0 * idf

        score_lexical += score_estrutura

        if score_lexical > 0:

            resultado = {
                **chunk,
                "score_lexical": float(score_lexical),
                "score_estrutura": float(score_estrutura),
                "termos_encontrados": termos_encontrados
            }

            resultados.append(resultado)

    resultados.sort(
        key=lambda x: x["score_lexical"],
        reverse=True
    )

    return resultados[:top_k]


# ============================================================
# BUSCA HÍBRIDA
# ============================================================

def buscar_chunks_hibrido(
    pergunta,
    top_k_semantico=20,
    top_k_lexical=20
):
    """
    Combina busca semântica e busca lexical.

    O resultado é um conjunto de candidatos que será
    posteriormente avaliado pelo reranker.
    """

    from ..ingestion.embedding import gerar_embedding

    # --------------------------------------------------------
    # Embedding da pergunta
    # --------------------------------------------------------

    embedding_pergunta = gerar_embedding(
        pergunta
    )

    # --------------------------------------------------------
    # Busca semântica
    # --------------------------------------------------------

    resultados_semanticos = buscar_chunks_semantico(
        embedding_pergunta,
        top_k=top_k_semantico
    )

    # --------------------------------------------------------
    # Busca lexical
    # --------------------------------------------------------

    resultados_lexicais = buscar_chunks_lexical(
        pergunta,
        top_k=top_k_lexical
    )

    # --------------------------------------------------------
    # União dos resultados
    # --------------------------------------------------------

    candidatos = {}

    for resultado in resultados_semanticos:

        chunk_id = resultado["chunk_id"]

        candidatos[chunk_id] = {
            **resultado,
            "score_lexical": resultado.get(
                "score_lexical",
                0
            )
        }

    for resultado in resultados_lexicais:

        chunk_id = resultado["chunk_id"]

        if chunk_id in candidatos:

            candidatos[chunk_id]["score_lexical"] = (
                resultado["score_lexical"]
            )

        else:

            candidatos[chunk_id] = {
                **resultado,
                "similaridade": resultado.get(
                    "similaridade",
                    None
                )
            }

    return list(
        candidatos.values()
    )


# ============================================================
# CONTEXTOS LOCAIS PARA O RERANKER
# ============================================================

def construir_contextos_locais(candidatos, janela=1):
    """Transforma cada candidato em uma passagem contextual para o BGE.

    A recuperação inicial continua no nível de chunks (sem inflar a lista de
    candidatos). Antes do reranking, porém, cada candidato é avaliado junto de
    seus vizinhos. Assim uma cláusula iniciada no chunk 82 e terminada no 83 é
    uma única evidência para o reranker, e não duas evidências incompletas.
    """

    if not candidatos:
        return []

    chunks = carregar_embeddings()
    indices_por_id = {
        chunk["chunk_id"]: indice
        for indice, chunk in enumerate(chunks)
    }
    contextos = {}

    for candidato in candidatos:
        indice = indices_por_id.get(candidato["chunk_id"])

        if indice is None:
            continue

        inicio = max(0, indice - janela)
        fim = min(len(chunks), indice + janela + 1)
        janela_chunks = chunks[inicio:fim]
        ids = tuple(chunk["chunk_id"] for chunk in janela_chunks)

        # Contextos idênticos podem surgir quando candidatos adjacentes são
        # recuperados. Mantemos apenas um e registramos todas as âncoras.
        if ids not in contextos:
            contexto = {
                **candidato,
                "chunk_id": candidato["chunk_id"],
                "chunk_ids_contexto": list(ids),
                "chunks_origem": [candidato["chunk_id"]],
                "pagina_inicio": janela_chunks[0]["pagina"],
                "pagina_fim": janela_chunks[-1]["pagina"],
                "texto": "\n\n".join(
                    f"[Chunk {chunk['chunk_id']} | Página {chunk['pagina']}]\n"
                    f"{chunk['texto']}"
                    for chunk in janela_chunks
                )
            }
            contextos[ids] = contexto
        else:
            contextos[ids]["chunks_origem"].append(candidato["chunk_id"])

    return list(contextos.values())


# ============================================================
# BUSCAR CHUNKS VIZINHOS
# ============================================================

def buscar_chunks_vizinhos(
    chunk_id,
    janela=1
):
    """
    Recupera os chunks anteriores e posteriores
    ao chunk informado.

    Exemplo com janela=1:

        Chunk 81
        Chunk 82  <- origem
        Chunk 83
    """

    chunks = carregar_embeddings()

    indice_chunk = None

    for i, chunk in enumerate(chunks):

        if chunk["chunk_id"] == chunk_id:

            indice_chunk = i
            break

    if indice_chunk is None:
        return []

    inicio = max(
        0,
        indice_chunk - janela
    )

    fim = min(
        len(chunks),
        indice_chunk + janela + 1
    )

    return chunks[inicio:fim]


# ============================================================
# EXPANDIR CANDIDATOS ANTES DO RERANKING
# ============================================================

def expandir_candidatos(
    candidatos,
    janela=1
):
    """
    Expande os candidatos encontrados pela busca híbrida
    adicionando seus chunks vizinhos.

    Essa etapa acontece antes do reranking.
    """

    if not candidatos:
        return []

    candidatos_expandidos = {}

    for candidato in candidatos:

        chunk_id = candidato["chunk_id"]

        vizinhos = buscar_chunks_vizinhos(
            chunk_id,
            janela=janela
        )

        for chunk in vizinhos:

            id_chunk = chunk["chunk_id"]

            if id_chunk not in candidatos_expandidos:

                candidatos_expandidos[id_chunk] = {
                    **chunk,
                    "chunk_origem": chunk_id,
                    "similaridade": candidato.get(
                        "similaridade"
                    ),
                    "score_lexical": candidato.get(
                        "score_lexical",
                        0
                    )
                }

    resultados = list(
        candidatos_expandidos.values()
    )

    resultados.sort(
        key=lambda x: x["chunk_id"]
    )

    return resultados


# ============================================================
# EXPANDIR CONTEXTO APÓS RERANKING
# ============================================================

def expandir_contexto(
    resultados,
    janela=1
):
    """
    Expande os resultados selecionados pelo reranker
    adicionando seus chunks vizinhos.
    """

    if not resultados:
        return []

    chunks_expandidos = {}

    for resultado in resultados:

        chunk_id = resultado["chunk_id"]

        vizinhos = buscar_chunks_vizinhos(
            chunk_id,
            janela=janela
        )

        for chunk in vizinhos:

            id_vizinho = chunk["chunk_id"]

            if id_vizinho not in chunks_expandidos:

                chunks_expandidos[id_vizinho] = {
                    **chunk,
                    "chunk_origem": chunk_id,
                    "score_reranker": resultado.get(
                        "score_reranker"
                    )
                }

    resultados_expandidos = list(
        chunks_expandidos.values()
    )

    resultados_expandidos.sort(
        key=lambda x: x["chunk_id"]
    )

    return resultados_expandidos


# ============================================================
# TESTE V3 — DIAGNÓSTICO DO CHUNK 82
# ============================================================

if __name__ == "__main__":

    from ..ingestion.embedding import gerar_embedding

    pergunta = "Qual é a remuneração do CRI Meraki?"

    print("\n" + "=" * 70)
    print("TESTE V3 — DIAGNÓSTICO DO CHUNK 82")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Carregar chunks
    # --------------------------------------------------------

    chunks = carregar_embeddings()

    print(
        f"\nTotal de chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # 2. Gerar embedding da pergunta
    # --------------------------------------------------------

    embedding_pergunta = gerar_embedding(
        pergunta
    )

    # --------------------------------------------------------
    # 3. Ranking semântico completo
    # --------------------------------------------------------

    resultados_semanticos = []

    for chunk in chunks:

        score = similaridade_cosseno(
            embedding_pergunta,
            chunk["embedding"]
        )

        resultados_semanticos.append({
            "chunk_id": chunk["chunk_id"],
            "pagina": chunk["pagina"],
            "similaridade": float(score)
        })

    resultados_semanticos.sort(
        key=lambda x: x["similaridade"],
        reverse=True
    )

    # --------------------------------------------------------
    # 4. Localizar chunk 82 no ranking semântico
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CHUNK 82 — BUSCA SEMÂNTICA")
    print("-" * 70)

    for posicao, resultado in enumerate(
        resultados_semanticos,
        start=1
    ):

        if resultado["chunk_id"] == 82:

            print(
                f"Posição: {posicao}"
            )

            print(
                f"Similaridade: "
                f"{resultado['similaridade']:.4f}"
            )

            print(
                f"Página: "
                f"{resultado['pagina']}"
            )

            break

    # --------------------------------------------------------
    # 5. Ranking lexical completo
    # --------------------------------------------------------

    pergunta_normalizada = normalizar_texto(
        pergunta
    )

    termos = pergunta_normalizada.split()

    resultados_lexicais = []

    for chunk in chunks:

        texto_normalizado = normalizar_texto(
            chunk["texto"]
        )

        score_lexical = sum(
            1
            for termo in termos
            if termo in texto_normalizado
        )

        resultados_lexicais.append({
            "chunk_id": chunk["chunk_id"],
            "pagina": chunk["pagina"],
            "score_lexical": score_lexical
        })

    resultados_lexicais.sort(
        key=lambda x: x["score_lexical"],
        reverse=True
    )

    # --------------------------------------------------------
    # 6. Localizar chunk 82 no ranking lexical
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("CHUNK 82 — BUSCA LEXICAL")
    print("-" * 70)

    for posicao, resultado in enumerate(
        resultados_lexicais,
        start=1
    ):

        if resultado["chunk_id"] == 82:

            print(
                f"Posição: {posicao}"
            )

            print(
                f"Score lexical: "
                f"{resultado['score_lexical']}"
            )

            print(
                f"Página: "
                f"{resultado['pagina']}"
            )

            break

    # --------------------------------------------------------
    # 7. Top 10 semântico
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP 10 — BUSCA SEMÂNTICA")
    print("-" * 70)

    for i, resultado in enumerate(
        resultados_semanticos[:10],
        start=1
    ):

        print(
            f"{i:2d}. "
            f"Chunk {resultado['chunk_id']} | "
            f"Página {resultado['pagina']} | "
            f"Score {resultado['similaridade']:.4f}"
        )

    # --------------------------------------------------------
    # 8. Top 10 lexical
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TOP 10 — BUSCA LEXICAL")
    print("-" * 70)

    for i, resultado in enumerate(
        resultados_lexicais[:10],
        start=1
    ):

        print(
            f"{i:2d}. "
            f"Chunk {resultado['chunk_id']} | "
            f"Página {resultado['pagina']} | "
            f"Score {resultado['score_lexical']}"
        )

    print("\n" + "=" * 70)
