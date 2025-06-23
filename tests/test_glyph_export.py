import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repl import GLYPH_TO_GATE, glyph_map


def test_export_mapping():
    expected_keys = {"₿", "☀️", "📈", "⚡", "🤝"}
    mapping = glyph_map()
    assert isinstance(mapping, dict)
    assert expected_keys.issubset(mapping.keys())
    assert mapping is GLYPH_TO_GATE
