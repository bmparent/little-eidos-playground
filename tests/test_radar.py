import pandas as pd
import sys
from pathlib import Path as P

sys.path.insert(0, str(P(__file__).resolve().parents[1]))

from visual import radar


def test_render_radar(tmp_path):
    df = pd.DataFrame(
        {
            "mae_z": [0.0, 1.0],
            "entropy": [0.5, 0.2],
            "sent_dz": [0.5, 1.0],
        }
    )
    out = tmp_path / "radar.png"
    radar.render_radar(df, str(out))
    assert out.exists()
