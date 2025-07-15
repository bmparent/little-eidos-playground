from datetime import datetime
import os
from pathlib import Path

from . import sources, crawler, scorer, filters, archiver


def run(dry_run: bool = True) -> None:
    hour = datetime.utcnow().hour
    feeds = sources.candidate_feeds(hour)
    engine = scorer.RuleEngine()
    for feed in feeds:
        art = crawler.fetch(feed)
        if not art:
            continue
        if not filters.allow(art):
            continue
        score = engine.score(art.content.decode('utf-8', 'ignore'))
        if score < 0.75:
            continue
        if dry_run:
            print(f"DRY RUN: would archive {feed} score {score:.2f}")
        else:
            dest = archiver.stash(art, score)
            print(f"Archived to {dest}")


if __name__ == "__main__":
    dry = os.getenv("DRY_RUN", "1") == "1"
    run(dry_run=dry)
