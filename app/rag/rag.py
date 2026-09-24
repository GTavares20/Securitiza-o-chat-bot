from ..retrieval.retriever import (
    buscar_chunks_hibrido,
    construir_contextos_locais,
    expandir_contexto
)
from ..retrieval.reranker import reranquear
from ..retrieval.context_builder import construir_contexto
from ..llm.llm import perguntar_llm


# ============================================================
# RAG
# ============================================================

def responder_com_rag(pergunta, modelo):
    """
    Executa o pipeline completo de RAG:

    1. Busca híbrida
    2. Construção de contextos locais para os candidatos
    3. Reranking contextual
    4. Expansão final de contexto
    4. Construção do contexto
    5. Geração da resposta pelo LLM
    """

    # --------------------------------------------------------
    # 1. BUSCA HÍBRIDA
    # --------------------------------------------------------

    candidatos = buscar_chunks_hibrido(
        pergunta,
        top_k_semantico=20,
        top_k_lexical=20
    )

    # --------------------------------------------------------
    # 2. CONTEXTO LOCAL ANTES DO RERANKING
    #
    # Não expandimos a lista de candidatos: cada candidato vira uma passagem
    # local (âncora + vizinhos), que é o objeto avaliado pelo BGE.
    contextos_candidatos = construir_contextos_locais(
        candidatos,
        janela=1
    )

    # --------------------------------------------------------
    # 3. RERANKING CONTEXTUAL
    # --------------------------------------------------------

    resultados = reranquear(
        pergunta,
        contextos_candidatos,
        top_k=3
    )

    # --------------------------------------------------------
    # 4. EXPANSÃO FINAL DO CONTEXTO
    # --------------------------------------------------------

    resultados_expandidos = expandir_contexto(
        resultados,
        janela=1
    )

    # --------------------------------------------------------
    # 5. CONSTRUÇÃO DO CONTEXTO
    # --------------------------------------------------------

    contexto = construir_contexto(
        resultados_expandidos
    )

    # --------------------------------------------------------
    # 6. CONSTRUÇÃO DO PROMPT
    # --------------------------------------------------------

    prompt = f"""
Você é um assistente especializado em documentos de securitização.

Responda à pergunta utilizando EXCLUSIVAMENTE as informações
presentes no contexto fornecido.

REGRAS:

1. Não invente informações.
2. Não utilize informações externas ao contexto.
3. Se o contexto não possuir informação suficiente para responder,
   diga claramente que não foi possível encontrar a informação
   nos documentos fornecidos.
4. Seja objetivo e direto.
5. Sempre que possível, informe a página utilizada como fonte.
6. Diferencie informações específicas da operação de informações
   gerais sobre securitização.

CONTEXTO:

{contexto}

PERGUNTA:

{pergunta}
"""

    # --------------------------------------------------------
    # 7. CHAMADA AO LLM
    # --------------------------------------------------------

    resposta = perguntar_llm(
        prompt,
        historico=[],
        modelo=modelo
    )

    # --------------------------------------------------------
    # 8. RETORNO
    # --------------------------------------------------------

    return {
        "resposta": resposta,
        "fontes": resultados,
        "contexto": contexto,
        "candidatos": candidatos,
        "contextos_avaliados": contextos_candidatos
    }


# ============================================================
# TESTE
# ============================================================

if __name__ == "__main__":

    pergunta = (
        "Qual é a remuneração do CRI Meraki?"
    )

    modelo = "qwen3:4b"

    print("=" * 70)
    print("TESTE DO RAG — V3")
    print("=" * 70)

    print(
        f"\nPergunta:\n{pergunta}"
    )

    print(
        f"\nModelo:\n{modelo}"
    )

    # --------------------------------------------------------
    # EXECUÇÃO DO RAG
    # --------------------------------------------------------

    resultado = responder_com_rag(
        pergunta,
        modelo
    )

    # --------------------------------------------------------
    # DIAGNÓSTICO DA RECUPERAÇÃO CONTEXTUAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATOS ANTES DO RERANKING")
    print("=" * 70)

    for candidato in resultado["candidatos"]:
        print(
            f"Chunk {candidato['chunk_id']} | Página {candidato['pagina']} | "
            f"Semântico: {candidato.get('similaridade')} | "
            f"Lexical: {candidato.get('score_lexical')}"
        )

    print("\nChunks 82 e 83 recuperados diretamente:", [
        candidato["chunk_id"]
        for candidato in resultado["candidatos"]
        if candidato["chunk_id"] in (82, 83)
    ])

    print("\n" + "=" * 70)
    print("CONTEXTOS AVALIADOS PELO BGE")
    print("=" * 70)

    for contexto_candidato in resultado["contextos_avaliados"]:
        print(
            f"Âncora {contexto_candidato['chunk_id']} | "
            f"Chunks {contexto_candidato['chunk_ids_contexto']} | "
            f"Páginas {contexto_candidato['pagina_inicio']}-"
            f"{contexto_candidato['pagina_fim']}"
        )

    # --------------------------------------------------------
    # RESPOSTA
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESPOSTA DO LLM")
    print("=" * 70)

    print(
        f"\n{resultado['resposta']}"
    )

    # --------------------------------------------------------
    # FONTES ORIGINAIS DO RERANKER
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHUNKS ORIGINAIS DO RERANKER")
    print("=" * 70)

    for i, fonte in enumerate(
        resultado["fontes"],
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"RANK: {i}"
        )

        print(
            f"Chunk: {fonte['chunk_id']}"
        )

        print(
            f"Documento: {fonte['documento']}"
        )

        print(
            f"Página: {fonte['pagina']}"
        )

        print(
            f"Score Reranker: "
            f"{fonte['score_reranker']:.4f}"
        )

        print(
            f"Score Semântico: "
            f"{fonte['similaridade']}"
        )

        print(
            f"Score Lexical: "
            f"{fonte['score_lexical']}"
        )

        print("\nTEXTO DO CHUNK:")

        print(
            fonte["texto"]
        )

    print("\n" + "=" * 70)
    print("TOP 3 CONTEXTUAL DO BGE")
    print("=" * 70)

    for posicao, fonte in enumerate(resultado["fontes"], start=1):
        print(
            f"{posicao}. Âncora {fonte['chunk_id']} | "
            f"Contexto {fonte['chunk_ids_contexto']} | "
            f"Score {fonte['score_reranker']:.4f}"
        )

    # --------------------------------------------------------
    # CONTEXTO EXPANDIDO
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONTEXTO EXPANDIDO — V3")
    print("=" * 70)

    print(
        resultado["contexto"]
    )
