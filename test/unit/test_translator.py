from src.translator import translate_content
from unittest.mock import patch


@patch("src.translator.get_translation")
@patch("src.translator.get_language")
def test_chinese(mock_get_language, mock_get_translation):
    """
    Non-English input.

    We mock get_language and get_translation so this test does NOT depend
    on the real LLM behavior or exact wording of its translations.
    """
    # Simulate language detection saying "Chinese"
    mock_get_language.return_value = "Chinese"
    # Simulate translation result
    mock_get_translation.return_value = "This is a Chinese message"

    is_english, translated_content = translate_content("这是一条中文消息")

    assert is_english is False
    assert translated_content == "This is a Chinese message"


@patch("src.translator.get_translation")
@patch("src.translator.get_language")
def test_llm_normal_response(mock_get_language, mock_get_translation):
    """
    LLM behaves normally: detects a non-English language and returns a translation.
    We verify that translate_content wires them together correctly.
    """
    mock_get_language.return_value = "Spanish"
    mock_get_translation.return_value = "Hello"

    is_english, translated_content = translate_content("Hola")

    assert is_english is False
    assert translated_content == "Hello"


@patch("src.translator.get_translation")
@patch("src.translator.get_language")
def test_llm_gibberish_response(mock_get_language, mock_get_translation):
    """
    LLM returns gibberish / invalid outputs.

    We simulate:
      - language detection returns empty/invalid string
      - translation returns empty string

    translate_content should handle this gracefully and return
    an "invalid language detection" marker instead of crashing.
    """
    # Invalid language detection output
    mock_get_language.return_value = ""
    # Translation won't even be used, but set it to empty for clarity
    mock_get_translation.return_value = ""

    is_english, translated_content = translate_content("some text")

    assert is_english is False
    assert translated_content == "[Invalid language detection output]"
