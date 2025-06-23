from importlib import import_module

NAMES = [
    "mae_watcher",
    "entropy_watcher",
    "tuner",
    "branch_pruner",
    "visionary",
    "rune_caster",
]


def get(name):
    if name not in NAMES:
        raise ValueError(f"Unknown agent {name}")
    return import_module(f"agents.{name}")
