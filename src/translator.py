from __future__ import annotations
from typing import Tuple
import os
import json
import re  # kept in case you later extend with regex parsing

# Try to import the real Ollama client; fall back to a dummy in environments
# where ollama isn't installed (e.g., CI, some test rigs).
try:
    from ollama import Client
except Exception:
    Client = None  # type: ignore

# Model + host configuration
MODEL_NAME = os.getenv("MODEL_NAME", "Llama3.1:8b")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class _DummyClient:
    """Fallback so that importing this module doesn't explode in CI."""
    def chat(self, *args, **kwargs):
        class Message:
            def __init__(self, content: str):
                self.content = content

        class Response:
            def __init__(self, content: str):
                self.message = Message(content)

        return Response("dummy response")


if Client:
    client = Client(host=OLLAMA_URL)
else:
    client = _DummyClient()


def get_translation(post: str) -> str:
    """
    Low-level helper: translates a post into English via the LLM.

    Returns a string (which may be an error marker if something goes wrong).
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
    Returns the name of the language in English (e.g. 'German', 'Spanish', 'Chinese').

    Robust to different shapes of the client response.
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

        # Some clients may return a dict-like structure; others an object with .message.content
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
        else:
            message = getattr(response, "message", None)
            content = getattr(message, "content", "") if message is not None else ""

        return str(content).strip()
    except Exception as e:
        return f"[Error: {type(e).__name__} - {e}]"


def translate_content(post: str) -> Tuple[bool, str]:
    """
    High-level, robust translator that combines language detection and translation.

    Returns:
        (is_english, output_text)

    Behavior:
      - Empty / whitespace-only input:
            → (True, "")
      - If language detection says English (or 'en'):
            → (True, original_text)
      - If language detection says non-English:
            → (False, translated_text)
      - If detection/translation output is invalid:
            → (False, '[Invalid ...]' or an error marker)
      - On any exception:
            → (False, '[Error: <ExceptionName> - <message>]')
    """
    text = (post or "").strip()
    if not text:
        return True, ""

    try:
        # 1) Detect language using the LLM
        lang = get_language(text)

        # Validate language detection output
        if not isinstance(lang, str) or len(lang.strip()) == 0:
            return False, "[Invalid language detection output]"

        lang_clean = lang.strip().lower()
        is_english = lang_clean in ["english", "en"]

        # 2) If English, just return original text
        if is_english:
            return True, text

        # 3) If not English, attempt translation
        translation = get_translation(text)

        # Validate translation output
        if not isinstance(translation, str) or len(translation.strip()) == 0:
            return False, "[Invalid translation output]"

        return False, translation.strip()

    except Exception as e:
        return False, f"[Error: {type(e).__name__} - {e}]"


def query_llm_robust(post: str) -> Tuple[bool, str]:
    """
    Backwards-compatible wrapper used by other tests/notebooks.

    All real logic lives in translate_content so there is a single source of truth.
    """
    return translate_content(post)
