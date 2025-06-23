from pathlib import Path
import numpy as np

from engine import QuantumToy


BURST_GLYPH = 'H'


def shannon_entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def main():
    q = QuantumToy()
    p = np.abs(q.state) ** 2
    H = shannon_entropy(p)
    if H < 0.2 * q.n:
        path = Path('generated.eidos')
        with path.open('a', encoding='utf8') as f:
            f.write(f'{BURST_GLYPH}\n')
        return True
    return False


if __name__ == '__main__':
    main()
