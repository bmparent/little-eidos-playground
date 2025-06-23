import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_radar(df: pd.DataFrame, path: str) -> None:
    """Render a simple storm radar plot and save as PNG."""
    n = len(df) if len(df) else 1
    sizes = 50 + 200 * (1 - df["entropy"] / n)
    fig, ax = plt.subplots()
    sc = ax.scatter(
        df.index,
        df["mae_z"],
        c=df["mae_z"],
        cmap="RdYlGn_r",
        s=sizes,
        alpha=df["sent_dz"],
    )
    ax.set_xlabel("index")
    ax.set_ylabel("mae_z")
    fig.colorbar(sc, label="mae_z")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Render radar plot")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--data", default="features/latest.parquet", help="parquet file with features"
    )
    args = p.parse_args(argv)
    df = pd.read_parquet(args.data)
    render_radar(df, args.out)


if __name__ == "__main__":
    main()
