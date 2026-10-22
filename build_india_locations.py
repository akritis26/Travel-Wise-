"""Create a compact India cities file from the GeoNames India extract."""

import csv
from pathlib import Path
import re
import unicodedata
import zipfile

BASE_DIR = Path(__file__).resolve().parent
SOURCE_ZIP = BASE_DIR / "IN.zip"
ADMIN_CODES = BASE_DIR / "admin1CodesASCII.txt"
OUTPUT = BASE_DIR / "india_locations.csv"


def normalize(value):
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_value.lower())

if not SOURCE_ZIP.exists() or not ADMIN_CODES.exists():
    raise SystemExit("Download IN.zip and admin1CodesASCII.txt from GeoNames first.")

state_names = {}
with ADMIN_CODES.open(encoding="utf-8") as source:
    for line in source:
        fields = line.rstrip("\n").split("\t")
        if len(fields) >= 2 and fields[0].startswith("IN."):
            state_names[fields[0][3:]] = fields[1]

# Keep every India destination already used by the recommender even if it is
# a small place below the population cutoff.
india_destinations = {
    normalize(line.split(",")[0])
    for line in (BASE_DIR / "destination_ids.txt").read_text(encoding="utf-8-sig").splitlines()
    if line.endswith(", India")
}

with zipfile.ZipFile(SOURCE_ZIP) as archive:
    member = next(name for name in archive.namelist() if name.startswith("IN.") or name == "IN.txt")
    with archive.open(member) as source, OUTPUT.open("w", newline="", encoding="utf-8") as target:
        rows = csv.reader((line.decode("utf-8") for line in source), delimiter="\t")
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "geoname_id", "name", "ascii_name", "state_code", "state",
                "latitude", "longitude", "population", "feature_code",
            ],
        )
        writer.writeheader()
        for row in rows:
            if len(row) < 19 or row[6] != "P":
                continue
            population = int(row[14] or 0)
            feature_code = row[7]
            # Keep larger populated places and administrative capitals, avoiding
            # hundreds of thousands of tiny villages in the website selectors.
            if (
                population < 5000
                and not feature_code.startswith("PPLA")
                and feature_code != "PPLC"
                and normalize(row[2]) not in india_destinations
            ):
                continue
            writer.writerow({
                "geoname_id": row[0],
                "name": row[1],
                "ascii_name": row[2],
                "state_code": row[10],
                "state": state_names.get(row[10], ""),
                "latitude": row[4],
                "longitude": row[5],
                "population": population,
                "feature_code": feature_code,
            })

print(f"Created {OUTPUT.name}")
