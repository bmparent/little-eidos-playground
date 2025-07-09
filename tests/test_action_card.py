from executor import ActionCard


def test_action_card(tmp_path):
    card_file = tmp_path / "card.yaml"
    card_file.write_text("""kind: creator\nname: test\nrationale: x\nsuccess: y\n""")
    card = ActionCard.from_path(card_file)
    assert card.kind == "creator"
    assert card.name == "test"
