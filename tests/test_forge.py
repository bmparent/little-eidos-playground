import forge


def test_scaffold(tmp_path):
    dest = forge.scaffold('demo', tmp_path)
    assert dest.exists()
    assert (dest / '__init__.py').exists()
