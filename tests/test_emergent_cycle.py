from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import emergent_intelligence as ei
import state


@pytest.mark.integration
def test_emergent_cycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(state, "STATE_PATH", tmp_path / "state.json")

    monkeypatch.setattr(ei, "fetch_bitcoin", lambda: ("₿", 1.0))
    monkeypatch.setattr(ei, "fetch_weather", lambda lat="0", lon="0": ("☀️", 1.0))
    monkeypatch.setattr(ei, "fetch_trending", lambda: ("📈", 1))
    monkeypatch.setattr(ei.ai_buddy, "alignment_score", lambda: 0.5)
    monkeypatch.setattr(ei, "auto_commit", lambda: None)
    import sensors.gcp_rng as rng

    monkeypatch.setattr(rng, "latest_z", lambda timeout=60: 0.0)

    ei.main(dry_run=True)

    assert (tmp_path / "generated.eidos").exists()
    st = state.load_state()
    assert "qubit_state" in st
