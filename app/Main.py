from .llm.chatbot import Chatbot
from .llm.router import classificar_com_llm, escolher_modelo


# ============================================================
# INICIALIZAÇÃO
# ============================================================

print("Chatbot de Securitização")
print("Digite '/limpar' para iniciar uma nova conversa.")
print("Digite 'sair' para encerrar.\n")


chatbot = Chatbot(None)

modelo_atual = None


# ============================================================
# LOOP PRINCIPAL
# ============================================================

while True:

    pergunta = input("Você: ")


    # ========================================================
    # ENCERRAR
    # ========================================================

    if pergunta.lower() == "sair":

        print("Chat encerrado.")

        break


    # ========================================================
    # LIMPAR HISTÓRICO
    # ========================================================

    if pergunta.lower() == "/limpar":

        chatbot = Chatbot(modelo_atual)

        print(
            "\nHistórico apagado. "
            "Nova conversa iniciada.\n"
        )

        continue


    # ========================================================
    # ROUTER
    # ========================================================

    tipo_pergunta = classificar_com_llm(
        pergunta
    )

    modelo = escolher_modelo(
        tipo_pergunta
    )


    # ========================================================
    # TROCA DE MODELO
    # ========================================================

    if modelo != modelo_atual:

        modelo_atual = modelo

        chatbot.trocar_modelo(
            modelo
        )

        print(
            f"\n[Router → {modelo}]"
        )

        print(
            f"[Tipo de pergunta → "
            f"{tipo_pergunta}]\n"
        )


    # ========================================================
    # CHATBOT
    # ========================================================

    resultado = chatbot.perguntar(
        pergunta,
        tipo_pergunta
    )


    # ========================================================
    # RESPOSTA
    # ========================================================

    print(
        f"\nAssistente: "
        f"{resultado['resposta']}\n"
    )


    # ========================================================
    # FONTES
    # ========================================================

    if resultado["fontes"]:

        print(
            "Fontes:"
        )

        fontes_exibidas = set()


        for fonte in resultado["fontes"]:

            chave = (
                fonte["documento"],
                fonte["pagina"],
                fonte["chunk_id"]
            )


            # Evita repetir a mesma fonte

            if chave in fontes_exibidas:
                continue


            fontes_exibidas.add(
                chave
            )


            print(
                f"- "
                f"{fonte['documento']} | "
                f"Página {fonte['pagina']} | "
                f"Chunk {fonte['chunk_id']}"
            )


        print()