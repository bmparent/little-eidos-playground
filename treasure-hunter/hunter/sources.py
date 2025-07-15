from typing import List


def candidate_feeds(hour: int) -> List[str]:
    """Return a list of candidate feed URLs."""
    return [
        "https://news.ycombinator.com/rss",
        "https://export.arxiv.org/rss/cs.LG",
        "https://commons.wikimedia.org/w/index.php?title=Special:NewFiles&feed=rss",
    ]
