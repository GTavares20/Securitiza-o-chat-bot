
import requests
from ..core.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    SYSTEM_PROMPT,
    OLLAMA_TEMPERATURE)

def perguntar_llm(pergunta, historico, modelo):

    url = f"{OLLAMA_BASE_URL}/api/chat"

    mensagens = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    mensagens.extend(historico)

    mensagens.append(
        {
            "role": "user",
            "content": pergunta
        }
    )

    data = {
        "model": modelo,
        "messages": mensagens,
        "stream": False, 
        "options": {
        "temperature": OLLAMA_TEMPERATURE
    }
    }

    try:
        response = requests.post(
            url,
            json=data,
            timeout=240
        )

        response.raise_for_status()

        resultado = response.json()

        return resultado["message"]["content"]

    except requests.exceptions.Timeout:
        return "O modelo demorou muito para responder."

    except requests.exceptions.ConnectionError:
        return "Não foi possível conectar ao Ollama."

    except requests.exceptions.HTTPError as erro:
        return f"Erro HTTP ao consultar o modelo: {erro}"

    except Exception as erro:
        return f"Ocorreu um erro inesperado: {erro}"