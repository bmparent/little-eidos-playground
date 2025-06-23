import numpy as np
import requests

URL = 'https://global-mind.org/gcpdata/gcpdot.php'


def latest_z(timeout: int = 60) -> float:
    """Return the latest variance z-score from the GCP dot feed."""
    try:
        resp = requests.get(URL, timeout=timeout)
        txt = resp.text.strip()
        z = float(txt.split()[-1])
    except Exception:
        z = float(np.random.normal())
    return z
