import re


def limpar_texto(texto):
    """
    Limpa o texto extraído do PDF.

    Objetivos:
    - remover espaços desnecessários;
    - normalizar quebras de linha;
    - remover o rodapé repetitivo de assinatura digital;
    - preservar o conteúdo jurídico do documento.
    """

    # --------------------------------------------------------
    # 1. REMOVER RODAPÉ DE ASSINATURA DIGITAL
    # --------------------------------------------------------

    texto = re.sub(
        r"Document signed at Assinador Registro de Imóveis\."
        r".*?"
        r"U62P9-FE7KM-9A7MM-CQTBM\.",
        "",
        texto,
        flags=re.DOTALL
    )

    # --------------------------------------------------------
    # 2. NORMALIZAR ESPAÇOS
    # --------------------------------------------------------

    texto = re.sub(r"[ \t]+", " ", texto)

    # --------------------------------------------------------
    # 3. REMOVER ESPAÇOS NO INÍCIO/FIM DAS LINHAS
    # --------------------------------------------------------

    texto = re.sub(r" +\n", "\n", texto)
    texto = re.sub(r"\n +", "\n", texto)

    # --------------------------------------------------------
    # 4. NORMALIZAR QUEBRAS DE LINHA
    # --------------------------------------------------------

    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # --------------------------------------------------------
    # 5. REMOVER ESPAÇOS EXCESSIVOS NO TEXTO
    # --------------------------------------------------------

    texto = texto.strip()

    return texto


# ============================================================
# TESTE
# ============================================================

if __name__ == "__main__":

    from .document_loader import carregar_pdf

    caminho = (
        r"documents\CRI-Be-Brave-Termo-de-Securitizacao-v.assinada.pdf"
    )

    paginas = carregar_pdf(caminho)

    texto_bruto = paginas[0]["texto"]

    texto_limpo = limpar_texto(texto_bruto)

    print("=" * 70)
    print("TESTE DO TEXT CLEANER")
    print("=" * 70)

    print("\n--- TEXTO LIMPO ---\n")

    print(texto_limpo[:3000])