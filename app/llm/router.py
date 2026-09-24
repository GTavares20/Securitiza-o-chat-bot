from ..core.config import OLLAMA_MODEL, OLLAMA_MODEL_2
from .llm import perguntar_llm


def classificar_pergunta(pergunta):

    pergunta = pergunta.lower().strip()

    palavras_documentais = [
        "contrato",
        "cláusula",
        "cascata",
        "debênture",
        "cri",
        "cra",
        "emissão",
        "investidor",
        "recebíveis",
        "termo de securitização",
        "cessão",
        "cedente",
        "devedor",
        "lastro"
    ]

    palavras_gerais = [
        "o que é",
        "como funciona",
        "qual a diferença",
        "explique",
        "conceito",
        "significa",
        "definição"
    ]

    for palavra in palavras_documentais:
        if palavra in pergunta:
            return "documental"

    for palavra in palavras_gerais:
        if palavra in pergunta:
            return "geral"

    return "ambigua"


def classificar_com_llm(pergunta):

    prompt = f"""
Classifique a pergunta abaixo em apenas uma das categorias:

- documental
- geral
- ambigua

DOCUMENTAL:
Pergunta relacionada a uma operação, contrato, documento,
cláusula, estrutura financeira ou informação específica.

GERAL:
Pergunta conceitual ou explicativa sobre um assunto.

AMBIGUA:
Não é possível determinar claramente a intenção.

Responda SOMENTE com uma das três palavras:
documental
geral
ambigua

Pergunta:
{pergunta}
"""

    resposta = perguntar_llm(
        prompt,
        historico=[],
        modelo=OLLAMA_MODEL_2
    )

    resposta = resposta.strip().lower()

    if "documental" in resposta:
        return "documental"

    if "geral" in resposta:
        return "geral"

    return "ambigua"


def escolher_modelo(tipo_pergunta):

    if tipo_pergunta == "documental":
        return OLLAMA_MODEL

    return OLLAMA_MODEL_2