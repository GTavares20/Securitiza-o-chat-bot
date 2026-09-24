from pypdf import PdfReader


def carregar_pdf(caminho):

    leitor = PdfReader(caminho)

    paginas = []

    for numero, pagina in enumerate(leitor.pages, start=1):

        texto = pagina.extract_text() or ""

        paginas.append({
            "pagina": numero,
            "texto": texto
        })

    return paginas


if __name__ == "__main__":

    caminho = r"C:\Users\Gustavo\Desktop\AULAS\Curso_dados\Kaggle_project\Chat_bot\documents\CRI-Be-Brave-Termo-de-Securitizacao-v.assinada.pdf"

    paginas = carregar_pdf(caminho)

    print(f"Quantidade de páginas: {len(paginas)}")

    print("\n--- PÁGINA 1 ---\n")
    print(paginas[0]["texto"][:2000])