import requests
import json
import time
from typing import Optional, Dict, Any


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen:7b"


class LLMError(Exception):
    """Raised when LLM execution fails."""
    pass


def run_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: int = 120,
    retries: int = 2,
) -> str:
    """
    Execute a prompt against Ollama and return the response text.
    """

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if system_prompt:
        payload["system"] = system_prompt

    last_error = None

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=timeout,
            )

            if response.status_code != 200:
                raise LLMError(
                    f"Ollama returned {response.status_code}: {response.text}"
                )

            data = response.json()

            if "response" not in data:
                raise LLMError("Malformed response from Ollama")

            return data["response"].strip()

        except Exception as e:
            last_error = e
            if attempt < retries:
                time.sleep(2)
            else:
                raise LLMError(f"LLM failed after retries: {e}") from e
