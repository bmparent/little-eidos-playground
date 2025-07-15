import hashlib
from dataclasses import dataclass
from typing import Optional
import requests


@dataclass
class Artifact:
    url: str
    content: bytes
    sha256: str


def fetch(url: str, timeout: int = 10) -> Optional[Artifact]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException:
        return None
    content = resp.content
    digest = hashlib.sha256(content).hexdigest()
    return Artifact(url=url, content=content, sha256=digest)
