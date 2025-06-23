from __future__ import annotations

import pandas as pd
from pathlib import Path

from state import load_state, save_state


FEATURES_PATH = Path("features/latest.parquet")
SCRIPT_PATH = Path("generated.eidos")


def main() -> int:
    if not FEATURES_PATH.exists():
        return 0
    df = pd.read_parquet(FEATURES_PATH)
    if df.empty:
        return 0
    row = df.iloc[-1]
    n = len(df)
    runes = []
    if row.get("mae_z", 0) > 2.5:
        runes.append("ᚦ")
    if row.get("entropy_bits", n) < 0.3 * n:
        runes.append("ᚱ")
    if not runes:
        return 0
    with SCRIPT_PATH.open("a", encoding="utf8") as f:
        for r in runes:
            f.write(f"{r}\n")
    mem = load_state()
    mem["last_rune"] = runes[-1]
    save_state(mem)
    return 0


def cli(argv=None) -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
