import openai
import numpy as np

EMBED_MODEL = "text-embedding-3-small"


def embed(text: str) -> np.ndarray:
    """Return OpenAI embedding as ``np.float32`` array."""
    response = openai.Embedding.create(model=EMBED_MODEL, input=text)
    return np.asarray(response["data"][0]["embedding"], dtype=np.float32)
