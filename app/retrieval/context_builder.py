def construir_contexto(resultados):
    """
    Constrói o contexto que será enviado ao LLM
    a partir dos chunks selecionados pelo reranker.
    """

    partes = []

    for i, resultado in enumerate(resultados, start=1):

        documento = resultado.get(
            "documento",
            "Documento não informado"
        )

        pagina = resultado.get(
            "pagina",
            "Página não informada"
        )

        chunk_id = resultado.get(
            "chunk_id",
            "Chunk não informado"
        )

        texto = resultado.get(
            "texto",
            ""
        )

        bloco = f"""
--- TRECHO {i} ---
Documento: {documento}
Página: {pagina}
Chunk: {chunk_id}

{texto}
"""

        partes.append(bloco)

    contexto = "\n".join(partes)

    return contexto


# ============================================================
# TESTE
# ============================================================

if __name__ == "__main__":

    from .reranker import reranquear
    from .retriever import buscar_chunks_hibrido

    pergunta = "Qual é o prazo de vencimento dos CRI?"

    print("=" * 70)
    print("TESTE DO CONTEXT BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. BUSCA HÍBRIDA
    # --------------------------------------------------------

    candidatos = buscar_chunks_hibrido(
        pergunta,
        top_k_semantico=20,
        top_k_lexical=20
    )

    print(
        f"\nCandidatos recuperados: "
        f"{len(candidatos)}"
    )

    # --------------------------------------------------------
    # 2. RERANKING
    # --------------------------------------------------------

    resultados = reranquear(
        pergunta,
        candidatos,
        top_k=5
    )

    # --------------------------------------------------------
    # 3. CONSTRUIR CONTEXTO
    # --------------------------------------------------------

    contexto = construir_contexto(resultados)

    # --------------------------------------------------------
    # 4. MOSTRAR CONTEXTO
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONTEXTO GERADO")
    print("=" * 70)

    print(contexto)