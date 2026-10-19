"""Build a sourced details catalog for every TravelWise destination."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import html
import json
import re
import threading
import time
import unicodedata
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
DESTINATIONS_FILE = BASE_DIR / "destination_ids.txt"
DEFAULT_OUTPUT = BASE_DIR / "destination_details.csv"
USER_AGENT = (
    "TravelWiseStudentProject/1.0 "
    "(https://github.com/akritis26/Travel-Wise-)"
)
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
API_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
ACTION_API_URL = "https://en.wikipedia.org/w/api.php"
SEARCH_ALIASES = {
    "Århus, Denmark": "Aarhus, Denmark",
    "Bankok, Thailand": "Bangkok, Thailand",
    "Chefchaouene, Morocco": "Chefchaouen, Morocco",
    "Gaillimh, Ireland": "Galway, Ireland",
    "Ouarzazat, Morocco": "Ouarzazate, Morocco",
    "Vangviang, Laos": "Vang Vieng, Laos",
}
MAX_WORKERS = 3
REQUEST_INTERVAL_SECONDS = 0.5
_request_lock = threading.Lock()
_last_request_time = 0.0


def wait_for_request_slot():
    """Pace all worker requests together to stay below the API limit."""
    global _last_request_time
    with _request_lock:
        elapsed = time.monotonic() - _last_request_time
        if elapsed < REQUEST_INTERVAL_SECONDS:
            time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
        _last_request_time = time.monotonic()


class TextOnlyParser(HTMLParser):
    """Remove HTML markup from the search excerpt returned by Wikipedia."""

    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


INTEREST_KEYWORDS = {
    "beach": ["beach", "coast", "coastal", "seaside", "shore"],
    "adventure": [
        "hiking", "hike", "trekking", "climbing", "skiing", "diving",
        "surfing", "rafting", "kayaking", "trail",
    ],
    "nature": [
        "national park", "forest", "lake", "mountain", "wildlife",
        "waterfall", "valley", "island", "nature reserve",
    ],
    "culture": [
        "museum", "architecture", "art", "festival", "cultural",
        "temple", "church", "mosque",
    ],
    "nightlife": ["nightlife", "night club", "nightclub", "bars"],
    "history": [
        "historic", "historical", "ancient", "archaeological", "ruins",
        "unesco", "heritage", "castle", "monument",
    ],
    "shopping": ["shopping", "market", "bazaar", "shops"],
    "cuisine": ["cuisine", "culinary", "food", "gastronomy", "restaurant"],
}


def clean_excerpt(value):
    parser = TextOnlyParser()
    parser.feed(value or "")
    text = html.unescape(" ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()


def normalized_words(value):
    plain_text = unicodedata.normalize("NFKD", value)
    plain_text = "".join(
        character for character in plain_text
        if not unicodedata.combining(character)
    )
    return re.findall(r"[a-z0-9]+", plain_text.lower())


def choose_result(pages, destination_name):
    """Prefer a result whose title and description fit the requested place."""
    parts = [part.strip() for part in destination_name.split(",") if part.strip()]
    place = parts[0]
    context_parts = parts[1:]
    place_words = set(normalized_words(place))
    normalized_place = " ".join(normalized_words(place))

    best_page = None
    best_score = -1
    best_context_match = False
    best_title_match = False
    best_title_ratio = 0

    for page in pages:
        title = page.get("title", "")
        description = page.get("description") or ""
        excerpt = clean_excerpt(page.get("excerpt", ""))
        title_words = set(normalized_words(title))
        body_words = set(normalized_words(description + " " + excerpt))

        title_overlap = len(place_words & title_words)
        title_ratio = title_overlap / max(1, len(place_words))
        normalized_body = " ".join(normalized_words(description + " " + excerpt))
        normalized_title = " ".join(normalized_words(title))
        spelling_similarity = SequenceMatcher(
            None, normalized_place, normalized_title
        ).ratio()
        title_ratio = max(title_ratio, spelling_similarity)
        exact_title = normalized_title == normalized_place
        context_match = any(
            " ".join(normalized_words(context)) in normalized_body
            for context in context_parts
            if normalized_words(context)
        )

        score = title_ratio * 50
        if exact_title:
            score += 40
        if context_match:
            score += 25
        if place_words and place_words <= body_words:
            score += 10

        if score > best_score:
            best_page = page
            best_score = score
            best_context_match = context_match
            best_title_match = exact_title or title_ratio >= 0.8
            best_title_ratio = title_ratio

    if best_page is None:
        return None, "no_match"

    if best_title_ratio < 0.75:
        return None, "no_match"
    if best_title_match and (best_context_match or len(place_words) == 1):
        return best_page, "matched"
    if best_title_ratio >= 0.75 and best_context_match:
        return best_page, "review_match"
    return None, "no_match"


def tags_from_text(text):
    lowered = text.lower()
    return [
        tag
        for tag, keywords in INTEREST_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    ]


def search_destination(destination_name, query_text=None):
    search_text = query_text or SEARCH_ALIASES.get(destination_name, destination_name)
    query = urlencode({"q": search_text, "limit": 5})
    request = Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )

    for attempt in range(4):
        try:
            wait_for_request_slot()
            with urlopen(request, timeout=30) as response:
                return json.load(response).get("pages", [])
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            wait_seconds = int(error.headers.get("Retry-After", str(2 ** attempt)))
            time.sleep(max(5, wait_seconds))
        except (TimeoutError, URLError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

    return []


def fetch_page_summaries(page_ids):
    """Fetch article introductions in batches, keeping API traffic low."""
    summaries = {}
    for start in range(0, len(page_ids), 50):
        batch = page_ids[start:start + 50]
        query = urlencode({
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "pageids": "|".join(batch),
            "format": "json",
        })
        request = Request(
            f"{ACTION_API_URL}?{query}",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        for attempt in range(4):
            try:
                wait_for_request_slot()
                with urlopen(request, timeout=30) as response:
                    pages = json.load(response).get("query", {}).get("pages", {})
                for page_id, page in pages.items():
                    summaries[page_id] = re.sub(
                        r"\s+", " ", page.get("extract", "")
                    ).strip()
                break
            except HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    break
                wait_seconds = int(error.headers.get("Retry-After", str(2 ** attempt)))
                time.sleep(max(5, wait_seconds))
            except (TimeoutError, URLError):
                if attempt == 3:
                    break
                time.sleep(2 ** attempt)
    return summaries


FIELDNAMES = [
    "destination_id",
    "destination_name",
    "overview",
    "page_description",
    "interest_tags",
    "wikipedia_title",
    "wikipedia_page_id",
    "source_url",
    "source_name",
    "content_license",
    "license_url",
    "match_status",
]


def save_rows(output_path, rows):
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(output_path)


def enrich_destination(destination_id, destination_name):
    try:
        match_name = SEARCH_ALIASES.get(destination_name, destination_name)
        pages = search_destination(destination_name)
        page, match_status = choose_result(pages, match_name)
        if match_status == "no_match":
            place_only = destination_name.split(",", 1)[0].strip()
            fallback_query = SEARCH_ALIASES.get(destination_name, place_only)
            if fallback_query != destination_name:
                fallback_pages = search_destination(
                    destination_name,
                    query_text=fallback_query,
                )
                fallback_page, fallback_status = choose_result(
                    fallback_pages,
                    match_name,
                )
                if fallback_status != "no_match":
                    page, match_status = fallback_page, fallback_status
    except Exception as error:
        page = None
        if isinstance(error, HTTPError):
            match_status = f"lookup_error: HTTP {error.code}"
        else:
            match_status = f"lookup_error: {type(error).__name__}"

    if page:
        title = page.get("title", "")
        key = page.get("key", title.replace(" ", "_"))
        overview = clean_excerpt(page.get("excerpt", ""))
        description = page.get("description") or ""
        page_id = str(page.get("id", ""))
        page_url = "https://en.wikipedia.org/wiki/" + key.replace(" ", "_")
        tags = tags_from_text(overview + " " + description)
    else:
        title = ""
        overview = ""
        description = ""
        page_id = ""
        page_url = (
            "https://en.wikipedia.org/w/index.php?"
            + urlencode({"search": destination_name})
        )
        tags = []

    return {
        "destination_id": destination_id,
        "destination_name": destination_name,
        "overview": overview,
        "page_description": description,
        "interest_tags": json.dumps(tags, ensure_ascii=False),
        "wikipedia_title": title,
        "wikipedia_page_id": page_id,
        "source_url": page_url,
        "source_name": "Wikipedia contributors" if page else "Wikipedia search",
        "content_license": "CC BY-SA 4.0; excerpt may be shortened" if page else "",
        "license_url": LICENSE_URL if page else "",
        "match_status": match_status,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create sourced overview records for TravelWise destinations."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the details CSV.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N destinations (for a small initial check).",
    )
    parser.add_argument(
        "--retry-review",
        action="store_true",
        help="Retry entries that were not confidently matched in an earlier run.",
    )
    args = parser.parse_args()

    destination_names = DESTINATIONS_FILE.read_text(
        encoding="utf-8-sig"
    ).splitlines()
    if args.limit is not None:
        destination_names = destination_names[:args.limit]

    output_path = args.output.resolve()
    existing_rows = []
    completed_ids = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8-sig", newline="") as file:
            existing_rows = list(csv.DictReader(file))
        completed_ids = {
            int(row["destination_id"])
            for row in existing_rows
            if row.get("destination_id", "").isdigit()
        }

    rows_by_id = {
        int(row["destination_id"]): row
        for row in existing_rows
        if row.get("destination_id", "").isdigit()
    }
    confident_ids = {
        destination_id
        for destination_id, row in rows_by_id.items()
        if row.get("match_status") == "matched"
    }

    pending = [
        (destination_id, destination_name)
        for destination_id, destination_name in enumerate(destination_names)
        if destination_id not in completed_ids
        or (args.retry_review and destination_id not in confident_ids)
    ]
    completed_this_run = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(enrich_destination, destination_id, destination_name):
                destination_id
            for destination_id, destination_name in pending
        }
        for future in as_completed(futures):
            destination_id = futures[future]
            rows_by_id[destination_id] = future.result()
            completed_this_run += 1

            if completed_this_run % 25 == 0:
                save_rows(output_path, [rows_by_id[key] for key in sorted(rows_by_id)])
                print(
                    f"Processed {len(rows_by_id)}/{len(destination_names)} destinations...",
                    flush=True,
                )

    # Replace search snippets with fuller article introductions in batches.
    page_ids = sorted({
        row["wikipedia_page_id"]
        for row in rows_by_id.values()
        if row.get("wikipedia_page_id", "").isdigit()
    })
    summaries = fetch_page_summaries(page_ids)
    for row in rows_by_id.values():
        summary = summaries.get(row.get("wikipedia_page_id", ""), "")
        if summary:
            overview = summary
            if len(overview) > 1200:
                overview = overview[:1200].rsplit(" ", 1)[0].rstrip(" ,;:") + "..."
            row["overview"] = overview
            row["interest_tags"] = json.dumps(
                tags_from_text(summary + " " + row["page_description"]),
                ensure_ascii=False,
            )

    save_rows(output_path, [rows_by_id[key] for key in sorted(rows_by_id)])
    match_counts = {}
    for row in rows_by_id.values():
        status = row["match_status"].split(":", 1)[0]
        match_counts[status] = match_counts.get(status, 0) + 1

    print(f"Saved {len(rows_by_id)} destination records to {output_path}")
    print(f"Match status counts: {match_counts}")


if __name__ == "__main__":
    main()
