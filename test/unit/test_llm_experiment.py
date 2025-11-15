from unittest.mock import patch
from src.llm_experiment import query_llm_robust, client


@patch.object(client, "chat")
def test_unexpected_language(mock_chat):
    # we mock the model's response to return a random message
    mock_chat.return_value.message.content = "I don't understand your request"

    # just check it returns a valid tuple and doesn't crash
    result = query_llm_robust("Hier ist dein erstes Beispiel.")
    assert isinstance(result, tuple)
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


@patch.object(client, "chat")
def test_empty_response(mock_chat):
    mock_chat.return_value.message.content = ""
    result = query_llm_robust("Bonjour le monde")
    assert isinstance(result, tuple)
    assert result[0] is False


@patch.object(client, "chat")
def test_nonstring_response(mock_chat):
    mock_chat.return_value.message.content = {"text": "Hello"}
    result = query_llm_robust("Hola amigo")
    assert result[0] is False


@patch.object(client, "chat", side_effect=Exception("Network error"))
def test_model_exception(mock_chat):
    result = query_llm_robust("Ciao amico")
    assert result[0] is False


@patch.object(client, "chat")
def test_none_response(mock_chat):
    mock_chat.return_value.message.content = None
    result = query_llm_robust("こんにちは")
    assert result[0] is False
    assert "Invalid" in result[1] or "Error" in result[1]


@patch.object(client, "chat")
def test_very_long_response(mock_chat):
    mock_chat.return_value.message.content = "Hello" * 10000
    result = query_llm_robust("Привет")
    assert isinstance(result, tuple)
    assert len(result[1]) < 60000


@patch.object(client, "chat")
def test_gibberish_response(mock_chat):
    mock_chat.return_value.message.content = "�#@!∂ƒ©˙∆˚¬…æ≈ç√"
    result = query_llm_robust("안녕하세요")
    assert result[0] is False


