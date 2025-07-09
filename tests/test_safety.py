import safety


def test_within_budget():
    assert safety.within_budget("openai", 5, limit=10)
    assert not safety.within_budget("openai", 15, limit=10)


def test_moderate_text():
    assert safety.moderate_text('ok')
    assert not safety.moderate_text('x' * 5000)
