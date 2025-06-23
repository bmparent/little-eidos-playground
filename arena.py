import subprocess

AGENTS = [
    "mae_watcher",
    "entropy_watcher",
    "tuner",
    "branch_pruner",
    "visionary",
    "rune_caster",
]

for name in AGENTS:
    subprocess.run(["python", "-m", f"agents.{name}"], check=False)
