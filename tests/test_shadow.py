import os
import shadow


def test_chat_with_shadow(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_SHADOW", "1")
    result = shadow.chat_with_shadow([{"role": "user", "content": "hello"}])
    assert isinstance(result, dict)
    assert "type" in result
