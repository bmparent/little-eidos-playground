import sys
from pathlib import Path

import numpy as np

from forecasting import walk_forward_mae
from state import load_state, save_state


def main():
    mem = load_state()
    prices = np.array(mem.get("prices", []), dtype=float)
    if prices.size < 2:
        mem.setdefault("mae", 0.0)
        save_state(mem)
        return

    mae = walk_forward_mae(prices, **mem.get("tuned_params", {}))
    prev = mem.get("mae")
    mem["mae"] = mae
    save_state(mem)
    if prev is None or mae <= prev:
        sys.exit(0)
    sys.exit(1)


def cli(argv=None):
    main()


if __name__ == "__main__":
    cli()
