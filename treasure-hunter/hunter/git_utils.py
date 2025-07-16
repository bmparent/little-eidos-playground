import subprocess


def current_branch() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()


def assert_on_treasure_branch(expected: str = "treasure-hunter") -> None:
    if current_branch() != expected:
        raise RuntimeError(f"Treasure Hunter must run on '{expected}' branch.")
