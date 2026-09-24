from .llm import perguntar_llm
from ..rag.rag import responder_com_rag


class Chatbot:

    def __init__(self, modelo):
        self.modelo = modelo
        self.historico = []

    def trocar_modelo(self, modelo):
        self.modelo = modelo

    def perguntar(self, pergunta, tipo="geral"):

        # ====================================================
        # PERGUNTA DOCUMENTAL → RAG
        # ====================================================

        if tipo == "documental":

            resultado = responder_com_rag(
                pergunta,
                self.modelo
            )

            resposta = resultado["resposta"]

            fontes = resultado["fontes"]

        # ====================================================
        # PERGUNTA GERAL → LLM
        # ====================================================

        else:

            resposta = perguntar_llm(
                pergunta,
                self.historico,
                self.modelo
            )

            fontes = []

        # ====================================================
        # HISTÓRICO
        # ====================================================

        self.historico.append({
            "role": "user",
            "content": pergunta
        })

        self.historico.append({
            "role": "assistant",
            "content": resposta
        })

        # ====================================================
        # RETORNO
        # ====================================================

        return {
            "resposta": resposta,
            "fontes": fontes
        }