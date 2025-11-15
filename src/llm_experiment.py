from __future__ import annotations
from typing import Tuple
import os

# LLM client setup copied from notebook

try:
    from ollama import Client
except Exception:  # if ollama isn't installed in some env, don't crash imports
    Client = None  # type: ignore

MODEL_NAME = os.getenv("MODEL_NAME", "Llama3.1:8b")

OLLAMA_URL = os.getenv("OLLAMA_HOST", "localhost:11434")


class _DummyClient:
    """Fallback so that importing this module doesn't explode in CI."""
    def chat(self, *args, **kwargs):
        class Message:
            def __init__(self, content):
                self.content = content

        class Response:
            def __init__(self, content):
                self.message = Message(content)

        return Response("dummy response")


if Client:
    client = Client(host=OLLAMA_URL)
else:
    client = _DummyClient()


def get_translation(post: str) -> str:
    """
    Translate a non-English post into English using the local Ollama model.
    (Copied from notebook.)
    """
    context = (
        "You are a translation assistant. Translate the following text into natural English. "
        "Only output the translated text, with no explanations or commentary."
    )

    try:
        prompt = f"{context}\n\nText to translate:\n{post}"

        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.message.content.strip()

    except Exception as e:
        return f"[Error: {type(e).__name__} - {e}]"


def get_language(post: str) -> str:
    """
    Detects the language of a given post using the LLM.
    Returns the name of the language in English (e.g. 'German', 'Spanish', 'Chinese', etc.)
    (Copied from notebook, with a small robustness tweak.)
    """
    prompt = (
        "Identify the language of the following text. "
        "Respond with only the language name in English (for example, 'German', 'Spanish', 'Chinese'). "
        "Do not answer in the language itself.\n\n"
        f"Text:\n{post}"
    )

    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )

        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            content = getattr(getattr(response, "message", None), "content", "")

        return str(content).strip()
    except Exception as e:
        return f"[Error: {type(e).__name__} - {e}]"


def query_llm_robust(post: str) -> tuple[bool, str]:
    """
    A robust version of query_llm that safely handles unexpected model responses or errors.
    Ensures output is always in the correct format (bool, str).
    (Exactly your notebook code.)
    """
    try:
        # Try language detection
        lang = get_language(post)

        # Validate language detection output
        if not isinstance(lang, str) or len(lang.strip()) == 0:
            # Model gave empty or invalid response
            return False, "[Invalid language detection output]"

        # Determine if English
        is_english = lang.strip().lower() in ["english", "en"]

        # If English, return original post
        if is_english:
            return True, post.strip()

        # If not English, attempt translation
        translation = get_translation(post)

        # Validate translation output
        if not isinstance(translation, str) or len(translation.strip()) == 0:
            return False, "[Invalid translation output]"

        return False, translation.strip()

    except Exception as e:
        return False, f"[Error: {type(e).__name__} - {e}]"



