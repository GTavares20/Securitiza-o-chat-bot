import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE"))
OLLAMA_MODEL_2 = os.getenv("OLLAMA_MODEL_2")

SYSTEM_PROMPT = """
Regras:
1. Responda sempre em português do Brasil.
2. Não invente informações.
3. Não apresente como fato uma informação que você não tenha segurança de que seja verdadeira.
4. Quando não tiver informações suficientes para responder, diga claramente que não possui informações suficientes.
5. Diferencie informações gerais sobre securitização de informações específicas de uma operação.
"""