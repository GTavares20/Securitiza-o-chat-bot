from FlagEmbedding import FlagReranker

from .retriever import buscar_chunks_hibrido


# ============================================================
# 1. MODELO RERANKER
# ============================================================

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=False
)


# ============================================================
# 2. FUNÇÃO DE RERANKING
# ============================================================

def reranquear(pergunta, candidatos, top_k=5):

    if not candidatos:
        return []

    # --------------------------------------------------------
    # Criar pares para o BGE
    # --------------------------------------------------------

    pares = [
        [
            f"PERGUNTA:\n{pergunta}",
            f"TRECHO DO DOCUMENTO:\n{candidato['texto']}"
        ]
        for candidato in candidatos
    ]

    # --------------------------------------------------------
    # Calcular scores do BGE
    # --------------------------------------------------------

    scores = reranker.compute_score(
        pares,
        normalize=True
    )

    # --------------------------------------------------------
    # Associar score aos candidatos
    # --------------------------------------------------------

    resultados = []

    for candidato, score in zip(candidatos, scores):

        resultado = {
            **candidato,
            "score_reranker": float(score)
        }

        resultados.append(resultado)

    # --------------------------------------------------------
    # Ordenar pelo score do reranker
    # --------------------------------------------------------

    resultados.sort(
        key=lambda x: x["score_reranker"],
        reverse=True
    )

    return resultados[:top_k]


# ============================================================
# 3. TESTE DE DIAGNÓSTICO
# ============================================================

if __name__ == "__main__":

    pergunta = (
        "Qual é a remuneração do CRI Meraki?"
    )

    print("=" * 80)
    print("DIAGNÓSTICO DO RETRIEVAL + RERANKER")
    print("=" * 80)

    print(
        f"\nPergunta: {pergunta}"
    )

    # ========================================================
    # 4. BUSCA HÍBRIDA
    # ========================================================

    candidatos = buscar_chunks_hibrido(
        pergunta,
        top_k_semantico=20,
        top_k_lexical=20
    )

    print(
        f"\nCandidatos recuperados pelo Hybrid Retrieval: "
        f"{len(candidatos)}"
    )

    # ========================================================
    # 5. RERANKING
    # ========================================================

    resultados = reranquear(
        pergunta,
        candidatos,
        top_k=10
    )

    # ========================================================
    # 6. RESULTADO DO BGE
    # ========================================================

    print("\n" + "=" * 80)
    print("TOP 10 APÓS BGE RERANKER")
    print("=" * 80)

    for i, resultado in enumerate(
        resultados,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"Posição: {i}"
        )

        print(
            f"Chunk: "
            f"{resultado['chunk_id']}"
        )

        print(
            f"Página: "
            f"{resultado['pagina']}"
        )

        print(
            f"Score BGE: "
            f"{resultado['score_reranker']:.4f}"
        )

        print(
            f"Score semântico: "
            f"{resultado.get('similaridade')}"
        )

        print(
            f"Score lexical: "
            f"{resultado.get('score_lexical')}"
        )

        print(
            f"Termos encontrados: "
            f"{resultado.get('termos_encontrados')}"
        )

        print("\nTexto:")

        print(
            resultado["texto"][:700]
        )
