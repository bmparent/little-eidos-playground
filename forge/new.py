from __future__ import annotations

import argparse
from pathlib import Path
from cookiecutter.main import cookiecutter

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "helper"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    helper_p = sub.add_parser("helper")
    helper_p.add_argument("--name", required=True)
    helper_p.add_argument("--template", default="metrics")
    args = parser.parse_args(argv)

    if args.command == "helper":
        template = TEMPLATES / f"helper_{args.template}"
        dest_dir = Path("agents/helpers")
        dest_dir.mkdir(parents=True, exist_ok=True)
        cookiecutter(str(template), no_input=True, output_dir=str(dest_dir),
                    extra_context={"helper_name": args.name})
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
