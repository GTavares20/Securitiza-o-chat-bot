import json

from ..retrieval.retriever import buscar_chunks_hibrido
from ..retrieval.reranker import reranquear
from ..rag.rag import responder_com_rag


# ============================================================
# CONFIGURAÇÃO
# ============================================================

CAMINHO_TESTES = "data/evaluation/rag_tests.json"

MODELO = "qwen3:4b"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def encontrar_primeiro_rank(top_chunks, chunks_esperados):
    """
    Retorna a posição do primeiro chunk esperado encontrado.

    Exemplo:

    Top 5:
    [10, 20, 30, 40, 50]

    Chunks esperados:
    [30]

    Retorno:
    3
    """

    for rank, chunk_id in enumerate(
        top_chunks,
        start=1
    ):

        if chunk_id in chunks_esperados:
            return rank

    return None


def calcular_metricas(ranks):
    """
    Calcula:

    Recall@1
    Recall@3
    Recall@5
    MRR
    """

    total = len(ranks)

    if total == 0:

        return {
            "recall@1": 0,
            "recall@3": 0,
            "recall@5": 0,
            "mrr": 0
        }

    # --------------------------------------------------------
    # RECALL@K
    # --------------------------------------------------------

    recall_1 = sum(
        rank <= 1
        for rank in ranks
    ) / total

    recall_3 = sum(
        rank <= 3
        for rank in ranks
    ) / total

    recall_5 = sum(
        rank <= 5
        for rank in ranks
    ) / total

    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    mrr = sum(
        1 / rank
        for rank in ranks
    ) / total

    return {
        "recall@1": recall_1,
        "recall@3": recall_3,
        "recall@5": recall_5,
        "mrr": mrr
    }


def avaliar_resposta_inexistente(resposta):
    """
    Verifica se o modelo reconheceu que não encontrou
    informação suficiente nos documentos.

    A função procura expressões que indicam uma
    recusa adequada da resposta.
    """

    resposta_normalizada = resposta.lower()

    indicadores = [

        "não foi possível encontrar",

        "não foi possível encontrar a informação",

        "não encontrei",

        "não consta",

        "não está presente",

        "não há informação",

        "informação insuficiente",

        "não possui informações suficientes"

    ]

    return any(
        indicador in resposta_normalizada
        for indicador in indicadores
    )


# ============================================================
# CARREGAR TESTES
# ============================================================

with open(
    CAMINHO_TESTES,
    "r",
    encoding="utf-8"
) as arquivo:

    testes = json.load(arquivo)


# ============================================================
# CONTADORES
# ============================================================

ranks_obtidos = []

passaram = 0
falharam = 0

passaram_inexistentes = 0
falharam_inexistentes = 0


# ============================================================
# CABEÇALHO
# ============================================================

print("\n" + "=" * 70)
print("AVALIAÇÃO DO RETRIEVAL")
print("=" * 70)


# ============================================================
# EXECUÇÃO DOS TESTES
# ============================================================

for teste in testes:

    pergunta = teste["pergunta"]

    tipo = teste["tipo"]

    chunks_esperados = teste["chunks_esperados"]


    print("\n" + "-" * 70)

    print(
        f"Teste {teste['id']}"
    )

    print(
        f"Pergunta: {pergunta}"
    )


    # ========================================================
    # TESTE DE RESPOSTA INEXISTENTE
    # ========================================================

    if tipo == "resposta_inexistente":

        print(
            "Tipo: resposta inexistente"
        )


        # ----------------------------------------------------
        # EXECUTAR RAG COMPLETO
        # ----------------------------------------------------

        resultado_rag = responder_com_rag(
            pergunta,
            MODELO
        )


        resposta = resultado_rag["resposta"]


        # ----------------------------------------------------
        # EXIBIR RESPOSTA
        # ----------------------------------------------------

        print(
            f"\nResposta do modelo:\n{resposta}"
        )


        # ----------------------------------------------------
        # AVALIAR RECUSA
        # ----------------------------------------------------

        passou_inexistente = avaliar_resposta_inexistente(
            resposta
        )


        if passou_inexistente:

            print(
                "\nResultado: PASSOU"
            )

            passaram_inexistentes += 1

        else:

            print(
                "\nResultado: FALHOU"
            )

            falharam_inexistentes += 1


        continue


    # ========================================================
    # TESTE DE RESPOSTA EXISTENTE
    # ========================================================

    candidatos = buscar_chunks_hibrido(
        pergunta,
        top_k_semantico=20,
        top_k_lexical=20
    )


    # --------------------------------------------------------
    # RERANKING
    # --------------------------------------------------------

    resultados = reranquear(
        pergunta,
        candidatos,
        top_k=5
    )


    # --------------------------------------------------------
    # CHUNKS RECUPERADOS
    # --------------------------------------------------------

    top_chunks = [

        resultado["chunk_id"]

        for resultado in resultados

    ]


    # --------------------------------------------------------
    # ENCONTRAR RANK
    # --------------------------------------------------------

    rank = encontrar_primeiro_rank(
        top_chunks,
        chunks_esperados
    )


    print(
        f"Chunks esperados: {chunks_esperados}"
    )

    print(
        f"Top 5 recuperados: {top_chunks}"
    )


    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    if rank is not None:

        print(
            f"Rank do chunk esperado: {rank}"
        )

        print(
            "Resultado: PASSOU"
        )

        passaram += 1

        ranks_obtidos.append(rank)

    else:

        print(
            "Rank do chunk esperado: NÃO ENCONTRADO"
        )

        print(
            "Resultado: FALHOU"
        )

        falharam += 1


# ============================================================
# MÉTRICAS DE RETRIEVAL
# ============================================================

metricas = calcular_metricas(
    ranks_obtidos
)


# ============================================================
# RESUMO DO RETRIEVAL
# ============================================================

print("\n" + "=" * 70)
print("RESUMO DA AVALIAÇÃO")
print("=" * 70)


print(
    f"\nTestes de resposta existente: "
    f"{len(ranks_obtidos)}"
)

print(
    f"Passaram: {passaram}"
)

print(
    f"Falharam: {falharam}"
)


# ============================================================
# MÉTRICAS
# ============================================================

print("\n" + "-" * 70)
print("MÉTRICAS DE RETRIEVAL")
print("-" * 70)


print(
    f"Recall@1: {metricas['recall@1']:.2%}"
)

print(
    f"Recall@3: {metricas['recall@3']:.2%}"
)

print(
    f"Recall@5: {metricas['recall@5']:.2%}"
)

print(
    f"MRR:      {metricas['mrr']:.4f}"
)


# ============================================================
# AVALIAÇÃO DE RESPOSTAS INEXISTENTES
# ============================================================

print("\n" + "-" * 70)
print("AVALIAÇÃO DE RESPOSTAS INEXISTENTES")
print("-" * 70)


total_inexistentes = (
    passaram_inexistentes
    + falharam_inexistentes
)


print(
    f"Testes de resposta inexistente: "
    f"{total_inexistentes}"
)

print(
    f"Passaram: {passaram_inexistentes}"
)

print(
    f"Falharam: {falharam_inexistentes}"
)


# ------------------------------------------------------------
# TAXA DE RECUSA CORRETA
# ------------------------------------------------------------

if total_inexistentes > 0:

    taxa_recusa = (
        passaram_inexistentes
        / total_inexistentes
    )

    print(
        f"Taxa de recusa correta: "
        f"{taxa_recusa:.2%}"
    )


# ============================================================
# FIM
# ============================================================

print("\n" + "=" * 70)