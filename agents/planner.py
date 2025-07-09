"""LLM-powered planner generating Action Cards."""

from pathlib import Path
from datetime import datetime
import os
import uuid
import yaml
from safety import within_budget, moderate_text

TODO_DIR = Path("todo")


def plan(prompt: str = "Plan next action") -> Path:
    """Write an Action Card YAML file under :mod:`todo`."""
    TODO_DIR.mkdir(exist_ok=True)
    name = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6] + ".yaml"
    path = TODO_DIR / name

    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai

            if not moderate_text(prompt):
                raise RuntimeError("Prompt failed moderation")
            cost_estimate = len(prompt)
            if not within_budget("openai", cost_estimate):
                raise RuntimeError("API budget exceeded")
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            text = resp.choices[0].message.content
            data = yaml.safe_load(text)
        except Exception:
            data = {"kind": "note", "name": "llm_error", "rationale": prompt, "success": ""}
    else:
        data = {"kind": "creator", "name": "demo", "rationale": "no api", "success": ""}

    ordered = {
        "kind": data.get("kind"),
        "name": data.get("name"),
        "rationale": data.get("rationale", ""),
        "success": data.get("success", ""),
    }
    with path.open("w") as f:
        yaml.safe_dump(ordered, f, sort_keys=False)
    return path


def main() -> None:
    plan()


if __name__ == "__main__":
    main()
