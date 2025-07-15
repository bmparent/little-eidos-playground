from typing import Optional
from .crawler import Artifact


def allow(artifact: Artifact) -> bool:
    """Basic allow filter that limits size and blocks simple adult terms."""
    if len(artifact.content) > 5 * 1024 * 1024:
        return False
    block_terms = [b"sex", b"porn"]
    lower = artifact.content.lower()
    if any(term in lower for term in block_terms):
        return False
    return True
