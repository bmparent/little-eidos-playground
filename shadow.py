import os
import json
from typing import List, Dict


def _local_reply(prompt: str) -> str:
    """Generate a dummy JSON reply using a local transformers model if available."""
    try:
        from transformers import pipeline

        model_name = os.getenv("LOCAL_SHADOW_MODEL", "sshleifer/tiny-gpt2")
        generator = pipeline("text-generation", model=model_name)
        out = generator(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
        return out
    except Exception:
        # Fallback to a static response
        return '{"type": "card", "card": "kind: note\nname: local"}'


def chat_with_shadow(messages: List[Dict]) -> Dict:
    """Send ``messages`` to the shadow model and return a parsed JSON dict."""
    use_local = os.getenv("USE_LOCAL_SHADOW")
    if use_local:
        prompt = messages[-1]["content"] if messages else ""
        text = _local_reply(prompt)
    else:
        import openai

        client = getattr(chat_with_shadow, "_client", None)
        if client is None:
            client = openai.OpenAI()
            setattr(chat_with_shadow, "_client", client)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        text = resp.choices[0].message.content
    try:
        data = json.loads(text)
    except Exception:
        data = {"type": "card", "card": "kind: note\nname: parse_error"}
    return data


if __name__ == "__main__":
    import sys

    result = chat_with_shadow([{"role": "user", "content": "".join(sys.argv[1:])}])
    print(json.dumps(result, indent=2))
