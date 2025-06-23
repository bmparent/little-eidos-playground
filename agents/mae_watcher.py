import sys
from pathlib import Path

import numpy as np

from forecasting import forecast_price
from memory import load_memory, save_memory

MEM_PATH = Path('.eidos_memory.json')


def main():
    mem = load_memory() or {}
    prices = np.array(mem.get('prices', []), dtype=float)
    if prices.size < 2:
        # nothing to forecast, just persist
        mem.setdefault('mae', 0.0)
        save_memory(mem)
        return

    preds = []
    for i in range(1, len(prices)):
        pred, _ = forecast_price(prices[:i])
        preds.append(pred)
    actual = prices[1:]
    mae = float(np.mean(np.abs(np.array(preds) - actual)))
    prev = mem.get('mae')
    mem['mae'] = mae
    save_memory(mem)
    if prev is None or mae <= prev:
        sys.exit(0)
    sys.exit(1)


def cli(argv=None):
    main()


if __name__ == '__main__':
    cli()
