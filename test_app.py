from pathlib import Path
import json
import re
import traceback
import unicodedata

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from travelwise_features import SINGLE_CHOICE_COLUMNS, make_profile_features


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

SURVEY_COLUMNS = [
    "form_a", "form_b", "form_c", "form_f", "form_g",
    "form_h", "form_i", "form_j", "form_r", "form_rr",
]

# These names match the JSON keys sent by templates/index.html.
KEY_MAP = {column: column for column in SURVEY_COLUMNS}

print("Loading real data...")
DATA = pd.read_csv(BASE_DIR / "training_data.csv", dtype=str)
DEST_NAMES = (BASE_DIR / "destination_ids.txt").read_text(
    encoding="utf-8-sig"
).splitlines()

try:
    DETAILS = pd.read_csv(
        BASE_DIR / "destination_details.csv",
        dtype={"destination_id": str},
    ).fillna("")
    DETAILS_BY_ID = DETAILS.set_index("destination_id").to_dict(orient="index")
except FileNotFoundError:
    DETAILS_BY_ID = {}

USER_PROFILES = (
    DATA[["user_id"] + SURVEY_COLUMNS]
    .drop_duplicates("user_id")
    .reset_index(drop=True)
)
PROFILE_FEATURES = make_profile_features(USER_PROFILES)

# Include every single-choice answer, including age (form_a).
SINGLE_COLS = SINGLE_CHOICE_COLUMNS
MULTI_COLS = [column for column in PROFILE_FEATURES.columns if column not in SINGLE_COLS]

ENCODER = OneHotEncoder(handle_unknown="ignore")
USER_SINGLE = ENCODER.fit_transform(PROFILE_FEATURES[SINGLE_COLS])
USER_MULTI = csr_matrix(PROFILE_FEATURES[MULTI_COLS].astype(float).to_numpy())
USER_VECTORS = hstack([USER_SINGLE, USER_MULTI], format="csr")

USER_TO_ROW = {uid: i for i, uid in enumerate(USER_PROFILES["user_id"])}
INTERACTION_ROWS = DATA["user_id"].map(USER_TO_ROW).to_numpy(dtype=int)
DEST_COLS = DATA["destination_id"].astype(int).to_numpy()
LIKED_VALS = DATA["liked"].astype(float).to_numpy()

N_DEST = len(DEST_NAMES)
YES_MATRIX = csr_matrix(
    (LIKED_VALS, (INTERACTION_ROWS, DEST_COLS)),
    shape=(len(USER_PROFILES), N_DEST),
)
SEEN_MATRIX = csr_matrix(
    (np.ones(len(DATA)), (INTERACTION_ROWS, DEST_COLS)),
    shape=(len(USER_PROFILES), N_DEST),
)

ALL_YES = np.asarray(YES_MATRIX.sum(axis=0)).ravel()
ALL_SEEN = np.asarray(SEEN_MATRIX.sum(axis=0)).ravel()
AVG_RATE = ALL_YES.sum() / max(ALL_SEEN.sum(), 1)
PRIOR = 20
OVERALL_RATES = (ALL_YES + PRIOR * AVG_RATE) / (ALL_SEEN + PRIOR)

INDIA_LOCATIONS = pd.read_csv(
    BASE_DIR / "india_locations.csv",
    dtype={"geoname_id": str, "state_code": str},
).fillna("")


def normalize_location(value):
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_value.lower())


CITY_ALIASES = {"mysore": "mysuru", "panaji": "panjim"}
INDIA_LOCATIONS["normalized_name"] = INDIA_LOCATIONS["ascii_name"].map(normalize_location)
INDIA_BY_NAME = {
    name: rows.sort_values("population", ascending=False).iloc[0].to_dict()
    for name, rows in INDIA_LOCATIONS.groupby("normalized_name")
}
COUNTRY_BY_DESTINATION = {
    index: name.rsplit(",", 1)[-1].strip() if "," in name else ""
    for index, name in enumerate(DEST_NAMES)
}

LANDMARKS_BY_CITY = {
    "Jaipur, India": [
        {
            "name": "Amer Fort",
            "overview": "A hilltop fort known for its Rajput architecture, courtyards, and views over Maota Lake.",
            "interest_tags": ["culture", "history", "adventure", "mountain"],
            "wikipedia_title": "Amer Fort",
        },
        {
            "name": "Jal Mahal",
            "overview": "A palace set in the middle of Man Sagar Lake, best admired from the lakeside promenade.",
            "interest_tags": ["culture", "history", "lake", "nature"],
            "wikipedia_title": "Jal Mahal",
        },
        {
            "name": "Hawa Mahal",
            "overview": "Jaipur's five-storey Palace of Winds, famous for its honeycomb facade and old-city setting.",
            "interest_tags": ["culture", "history", "urban"],
            "wikipedia_title": "Hawa_Mahal",
        },
        {
            "name": "City Palace",
            "overview": "A royal complex of courtyards, museums, and palaces in the historic heart of Jaipur.",
            "interest_tags": ["culture", "history", "urban"],
            "wikipedia_title": "City_Palace,_Jaipur",
        },
        {
            "name": "Jantar Mantar",
            "overview": "An 18th-century astronomical observatory and UNESCO World Heritage Site beside the City Palace.",
            "interest_tags": ["culture", "history", "urban"],
            "wikipedia_title": "Jantar_Mantar,_Jaipur",
        },
        {
            "name": "Nahargarh Fort",
            "overview": "A historic fort on the Aravalli hills with sweeping views across Jaipur.",
            "interest_tags": ["culture", "history", "mountain", "nature"],
            "wikipedia_title": "Nahargarh_Fort",
        },
    ],
}

ACTIVITY_TAGS = {
    "0": "beach", "1": "adventure", "2": "nature", "3": "culture",
    "4": "nightlife", "5": "history", "6": "shopping", "7": "cuisine",
}
SCENERY_TAGS = {
    "0": "urban", "1": "rural", "2": "sea", "3": "mountain",
    "4": "lake", "5": "desert", "6": "plains", "7": "jungle",
}


def india_location_for_destination(destination_id):
    if COUNTRY_BY_DESTINATION.get(destination_id) != "India":
        return None
    city_name = DEST_NAMES[destination_id].split(",", 1)[0].strip()
    normalized = normalize_location(city_name)
    normalized = CITY_ALIASES.get(normalized, normalized)
    return INDIA_BY_NAME.get(normalized)


def city_landmark_recommendations(city_name, city_score, body):
    landmarks = LANDMARKS_BY_CITY.get(city_name)
    if not landmarks:
        return None

    selected_tags = {
        tag
        for key, tag_map in (("form_f", ACTIVITY_TAGS), ("form_g", SCENERY_TAGS))
        for choice in body.get(key, [])
        if (tag := tag_map.get(str(choice)))
    }

    ranked = []
    for landmark in landmarks:
        matching_tags = selected_tags.intersection(landmark["interest_tags"])
        preference_score = (
            len(matching_tags) / len(selected_tags) if selected_tags else 0
        )
        score = 0.7 * city_score + 0.3 * preference_score
        ranked.append((score, landmark))

    ranked.sort(key=lambda item: (-item[0], item[1]["name"]))
    return [
        {
            "destination_name": f"{landmark['name']}, Jaipur, India",
            "score": round(float(score), 4),
            "overview": landmark["overview"],
            "interest_tags": landmark["interest_tags"],
            "source_url": (
                "https://en.wikipedia.org/wiki/" + landmark["wikipedia_title"]
            ),
        }
        for score, landmark in ranked[:5]
    ]


def destination_options():
    options = []
    for destination_id, destination_name in enumerate(DEST_NAMES):
        parts = [part.strip() for part in destination_name.rsplit(",", 1)]
        country = parts[-1] if parts else ""
        city = destination_name.split(",", 1)[0].strip()
        location = india_location_for_destination(destination_id)
        options.append({
            "destination_id": destination_id,
            "name": destination_name,
            "city": city,
            "country": country,
            "state": location["state"] if location else "",
        })
    return options


DESTINATION_OPTIONS = destination_options()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/location-options")
def location_options():
    india_states = sorted(
        state for state in INDIA_LOCATIONS["state"].dropna().unique() if state
    )
    india_destination_states = sorted({
        item["state"] for item in DESTINATION_OPTIONS if item["country"] == "India" and item["state"]
    })
    return jsonify({
        "countries": sorted({item["country"] for item in DESTINATION_OPTIONS if item["country"]}),
        "india_states": india_states,
        "india_destination_states": india_destination_states,
        "destinations": DESTINATION_OPTIONS,
    })


@app.route("/api/india-cities")
def india_cities():
    state = request.args.get("state", "")
    rows = INDIA_LOCATIONS
    if state:
        rows = rows[rows["state"] == state]
    rows = rows.sort_values(["population", "name"], ascending=[False, True])
    return jsonify({
        "cities": [
            {"id": row.geoname_id, "name": row.name, "state": row.state}
            for row in rows.itertuples()
        ]
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    try:
        body = request.get_json(force=True)

        # Preserve multi-select answers as lists; the feature helper parses them.
        profile = {}
        for frontend_key, form_key in KEY_MAP.items():
            value = body.get(frontend_key, [])
            if value is None or value == "":
                profile[form_key] = []
            elif isinstance(value, list):
                profile[form_key] = [str(item) for item in value]
            else:
                profile[form_key] = [str(value)]

        your_profile = pd.DataFrame([profile], columns=SURVEY_COLUMNS)
        your_features = make_profile_features(your_profile)
        your_single = ENCODER.transform(your_features[SINGLE_COLS])
        your_multi = csr_matrix(your_features[MULTI_COLS].astype(float).to_numpy())
        your_vector = hstack([your_single, your_multi], format="csr")

        k = min(50, len(USER_PROFILES))
        if k == 0:
            return jsonify({"error": "No traveler profiles are available."}), 503

        # Rank profiles by their preference similarity to this request.
        similarities = cosine_similarity(your_vector, USER_VECTORS).ravel()
        neighbor_rows = np.argsort(similarities)[::-1][:k]
        weights = np.clip(similarities[neighbor_rows], 0, None)
        if weights.sum() == 0:
            weights = np.ones(len(neighbor_rows), dtype=float)
        weights /= weights.sum()

        weight_row = csr_matrix(
            (weights, (np.zeros(len(neighbor_rows), dtype=int), neighbor_rows)),
            shape=(1, len(USER_PROFILES)),
        )
        neighbor_yes = (weight_row @ YES_MATRIX).toarray().ravel()
        neighbor_seen = (weight_row @ SEEN_MATRIX).toarray().ravel()
        scores = (neighbor_yes + 5 * OVERALL_RATES) / (neighbor_seen + 5)

        location_mode = body.get("location_mode", "anywhere")
        allowed = np.ones(N_DEST, dtype=bool)
        distance_by_destination = {}

        if location_mode == "specific":
            country = str(body.get("destination_country", "")).strip()
            if country not in COUNTRY_BY_DESTINATION.values():
                return jsonify({"error": "Choose a destination country."}), 400
            allowed &= np.array([
                COUNTRY_BY_DESTINATION.get(i) == country for i in range(N_DEST)
            ])

            state = str(body.get("destination_state", "")).strip()
            if state:
                allowed &= np.array([
                    (india_location_for_destination(i) or {}).get("state", "") == state
                    for i in range(N_DEST)
                ])

            city = str(body.get("destination_city", "")).strip()
            if city:
                allowed &= np.array([
                    DEST_NAMES[i] == city for i in range(N_DEST)
                ])

        elif location_mode == "nearby":
            # Nearby mode is supported only for the Indian candidate locations
            # covered by the bundled coordinate lookup.
            allowed[:] = False
            origin_id = str(body.get("origin_geoname_id", "")).strip()
            origin_matches = INDIA_LOCATIONS[INDIA_LOCATIONS["geoname_id"] == origin_id]
            if origin_matches.empty:
                return jsonify({"error": "Choose a starting city in India."}), 400
            origin = origin_matches.iloc[0]
            try:
                radius_km = float(body.get("distance_km", 300))
            except (TypeError, ValueError):
                return jsonify({"error": "Choose a valid distance range."}), 400
            if radius_km not in {100, 300, 500, 1000}:
                return jsonify({"error": "Choose one of the listed distance ranges."}), 400

            origin_lat = np.radians(float(origin["latitude"]))
            origin_lon = np.radians(float(origin["longitude"]))
            for destination_id in range(N_DEST):
                location = india_location_for_destination(destination_id)
                if not location:
                    continue
                destination_lat = np.radians(float(location["latitude"]))
                destination_lon = np.radians(float(location["longitude"]))
                delta_lat = destination_lat - origin_lat
                delta_lon = destination_lon - origin_lon
                haversine = (
                    np.sin(delta_lat / 2) ** 2
                    + np.cos(origin_lat) * np.cos(destination_lat)
                    * np.sin(delta_lon / 2) ** 2
                )
                distance = 6371 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
                distance_by_destination[destination_id] = float(distance)
                allowed[destination_id] = distance <= radius_km

        ranking_order = np.argsort(scores)[::-1]
        ranked = [int(i) for i in ranking_order if allowed[i]][:5]

        selected_city = str(body.get("destination_city", "")).strip()
        if location_mode == "specific" and selected_city:
            city_destination_id = next(
                (
                    destination_id
                    for destination_id, name in enumerate(DEST_NAMES)
                    if name == selected_city
                ),
                None,
            )
            if city_destination_id is not None:
                landmarks = city_landmark_recommendations(
                    selected_city, scores[city_destination_id], body
                )
                if landmarks is not None:
                    return jsonify({
                        "similar_travelers": k,
                        "recommendations": landmarks,
                        "message": "",
                    })

        recommendations = []
        for destination_id in ranked:
            detail = DETAILS_BY_ID.get(str(destination_id), {})
            raw_tags = detail.get("interest_tags", "")
            try:
                tags = json.loads(raw_tags) if raw_tags else []
            except (ValueError, TypeError):
                tags = []

            recommendation = {
                "destination_name": DEST_NAMES[destination_id],
                "score": round(float(scores[destination_id]), 4),
                "overview": detail.get("overview", ""),
                "interest_tags": tags,
                "source_url": detail.get("source_url", ""),
            }
            if destination_id in distance_by_destination:
                recommendation["distance_km"] = round(distance_by_destination[destination_id])
            recommendations.append(recommendation)

        return jsonify({
            "similar_travelers": k,
            "recommendations": recommendations,
            "message": "No destinations matched that location and distance. Try a broader area or radius."
            if not recommendations and location_mode != "anywhere" else "",
        })
    except Exception as error:
        print("Error while generating recommendations:")
        traceback.print_exc()
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
