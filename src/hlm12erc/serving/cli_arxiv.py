import requests
import feedparser
import time
from urllib.parse import quote

from tqdm import trange


class ArxivCommands:
    def __init__(self, api_url: str = "http://export.arxiv.org/api/query?"):
        self.api_url = api_url

    def count(self, term: str, year: int, out: str) -> int:
        total_count = 0
        # ArXiv only allows you to retrieve 2000 entries at a time. If you expect more than 2000 results, you'll need to paginate.
        pgsz = 1_000
        with trange(0, 10_000, pgsz, desc="pages", position=0) as pbar:
            for i in pbar:
                query = f"search_query=cat:cs.AI+OR+cat:cs.LG+AND+all:{quote('natural language')}+AND+all:{quote(term)}&start={i}&max_results={pgsz}"
                response = requests.get(self.api_url + query)
                feed = feedparser.parse(response.content)
                if feed and feed.entries and len(feed.entries) == 0:
                    break
                else:
                    for entry in feed.entries:
                        if entry.published_parsed.tm_year == year:
                            total_count += 1
                # To respect ArXiv's terms of use, sleep for 3 seconds before making another request
                time.sleep(3)
        if out:
            with open(out, "a+", encoding="utf-8") as f:
                f.write(f"{term},{year},{total_count}\n")
        return total_count
