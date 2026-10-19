"""Local, no-API-key HTTP backend for TravelWise recommendations."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from travelwise_features import make_profile_features


BASE_DIR = Path(__file__).resolve().parent
TRAINING_FILE = BASE_DIR / "training_data.csv"
DESTINATION_IDS_FILE = BASE_DIR / "destination_ids.txt"
DETAILS_FILE = BASE_DIR / "destination_details.csv"
PORT = 8000
NEIGHBOR_COUNT = 50

SURVEY_COLUMNS = [
    "form_a", "form_b", "form_c", "form_f", "form_g",
    "form_h", "form_i", "form_j", "form_r", "form_rr",
]
SINGLE_CHOICE_COLUMNS = ["form_b", "form_c", "form_h", "form_i", "form_j", "form_r"]
MULTI_CHOICE_COLUMNS = [
    "form_a_0", "form_a_1", "form_a_2", "form_a_3",
    "form_f_0", "form_f_1", "form_f_2", "form_f_3",
    "form_f_4", "form_f_5", "form_f_6", "form_f_7",
    "form_g_0", "form_g_1", "form_g_2", "form_g_3",
    "form_g_4", "form_g_5", "form_g_6", "form_g_7",
    "form_rr_e", "form_rr_n", "form_rr_c", "form_rr_a",
    "form_rr_s", "form_rr_m", "form_rr_f", "form_rr_o",
]
ALLOWED = {
    "age": {"0", "1", "2", "3"},
    "budget": {"0", "1", "2", "3"},
    "season": {"0", "1", "2", "3"},
    "experiences": {str(number) for number in range(8)},
    "scenery": {str(number) for number in range(8)},
    "activity": {"0", "1", "2"},
    "safety": {"0", "1", "2"},
    "popularity": {"0", "1", "2"},
    "where": {"0", "1"},
    "regions": set("encasmfo"),
}


def load_recommender_data():
    """Load the full dataset once, then keep its searchable matrices in memory."""
    data = pd.read_csv(TRAINING_FILE, dtype=str)
    profiles = data[["user_id"] + SURVEY_COLUMNS].drop_duplicates("user_id").reset_index(drop=True)
    profile_features = make_profile_features(profiles)

    encoder = OneHotEncoder(handle_unknown="ignore")
    single_vectors = encoder.fit_transform(profile_features[SINGLE_CHOICE_COLUMNS])
    multi_vectors = csr_matrix(profile_features[MULTI_CHOICE_COLUMNS].astype(float).to_numpy())
    user_vectors = hstack([single_vectors, multi_vectors], format="csr")

    user_to_row = {user_id: row for row, user_id in enumerate(profiles["user_id"])}
    interaction_rows = data["user_id"].map(user_to_row).to_numpy(dtype=int)
    destination_columns = data["destination_id"].astype(int).to_numpy()
    liked_values = data["liked"].astype(float).to_numpy()
    destination_names = DESTINATION_IDS_FILE.read_text(encoding="utf-8-sig").splitlines()
    matrix_shape = (len(profiles), len(destination_names))

    yes_matrix = csr_matrix((liked_values, (interaction_rows, destination_columns)), shape=matrix_shape)
    seen_matrix = csr_matrix((np.ones(len(data)), (interaction_rows, destination_columns)), shape=matrix_shape)
    all_yes = np.asarray(yes_matrix.sum(axis=0)).ravel()
    all_seen = np.asarray(seen_matrix.sum(axis=0)).ravel()
    average_yes_rate = all_yes.sum() / all_seen.sum()
    overall_rates = (all_yes + 20 * average_yes_rate) / (all_seen + 20)

    neighbor_finder = NearestNeighbors(
        n_neighbors=min(NEIGHBOR_COUNT, len(profiles)), metric="cosine", algorithm="brute"
    )
    neighbor_finder.fit(user_vectors)

    details = pd.read_csv(DETAILS_FILE, dtype={"destination_id": str}).fillna("")
    details_by_id = details.set_index("destination_id")
    return {
        "profiles": profiles,
        "encoder": encoder,
        "neighbor_finder": neighbor_finder,
        "yes_matrix": yes_matrix,
        "seen_matrix": seen_matrix,
        "overall_rates": overall_rates,
        "all_seen": all_seen,
        "destination_names": destination_names,
        "details_by_id": details_by_id,
    }


def normalize_request(body):
    """Check form values and translate web-friendly names to dataset columns."""
    if not isinstance(body, dict):
        raise ValueError("Send a JSON object containing the traveler preferences.")

    def one(name):
        value = str(body.get(name, "")).strip()
        if value not in ALLOWED[name]:
            raise ValueError(f"{name} must be one of: {', '.join(sorted(ALLOWED[name]))}.")
        return value

    def many(name):
        values = body.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a non-empty list of choices.")
        choices = list(dict.fromkeys(str(value).strip() for value in values))
        invalid = [value for value in choices if value not in ALLOWED[name]]
        if invalid:
            raise ValueError(f"Invalid {name} choice(s): {', '.join(invalid)}.")
        return choices

    where = one("where")
    regions = body.get("regions", [])
    if where == "1":
        if not isinstance(regions, list) or not regions:
            raise ValueError("Choose at least one region when 'where' is 1.")
        regions = list(dict.fromkeys(str(value).strip() for value in regions))
        invalid = [value for value in regions if value not in ALLOWED["regions"]]
        if invalid:
            raise ValueError("Regions must use: e, n, c, a, s, m, f, o.")
    else:
        regions = []

    profile = {
        "form_a": str([one("age")]),
        "form_b": str([one("budget")]),
        "form_c": str([one("season")]),
        "form_f": str(many("experiences")),
        "form_g": str(many("scenery")),
        "form_h": str([one("activity")]),
        "form_i": str([one("safety")]),
        "form_j": str([one("popularity")]),
        "form_r": str([where]),
        "form_rr": str(regions) if regions else "",
    }
    return pd.DataFrame([profile], columns=SURVEY_COLUMNS)


def recommend(body, data):
    """Return the top five destinations using the existing similar-traveler method."""
    profile = normalize_request(body)
    features = make_profile_features(profile)
    user_vectors = hstack([
        data["encoder"].transform(features[SINGLE_CHOICE_COLUMNS]),
        csr_matrix(features[MULTI_CHOICE_COLUMNS].astype(float).to_numpy()),
    ], format="csr")

    distances, rows = data["neighbor_finder"].kneighbors(user_vectors)
    neighbor_rows = rows[0]
    weights = np.clip(1 - distances[0], 0, 1)
    weight_matrix = csr_matrix(
        (weights, (np.zeros(len(neighbor_rows), dtype=int), neighbor_rows)),
        shape=(1, len(data["profiles"])),
    )
    neighbor_yes = (weight_matrix @ data["yes_matrix"]).toarray().ravel()
    neighbor_seen = (weight_matrix @ data["seen_matrix"]).toarray().ravel()
    scores = (neighbor_yes + 5 * data["overall_rates"]) / (neighbor_seen + 5)
    eligible = (data["all_seen"] >= 10) & (neighbor_seen > 0)
    scores[~eligible] = -1
    ranked = sorted(zip(scores, range(len(data["destination_names"]))), reverse=True)[:5]

    results = []
    for score, destination_id in ranked:
        if score < 0:
            continue
        name = data["destination_names"][destination_id]
        detail = data["details_by_id"].loc[str(destination_id)]
        place_name = name.split(",", 1)[0].strip().casefold()
        article_title = detail["wikipedia_title"].strip().casefold()
        title_matches = article_title == place_name or article_title.startswith(f"{place_name} (")
        safe = detail["match_status"] == "matched" and title_matches
        try:
            tags = json.loads(detail["interest_tags"]) if safe and detail["interest_tags"] else []
        except json.JSONDecodeError:
            tags = []
        results.append({
            "destination_id": str(destination_id),
            "destination_name": name,
            "score": round(float(score), 3),
            "overview": detail["overview"] if safe else "",
            "interest_tags": tags,
            "source_url": detail["source_url"],
        })
    return {"recommendations": results, "similar_travelers": len(neighbor_rows)}


class TravelWiseHandler(BaseHTTPRequestHandler):
    recommender_data = None

    def send_json(self, status, payload):
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self.send_json(200, {"ok": True})

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "message": "TravelWise backend is running."})
        else:
            self.send_json(404, {"error": "Not found. Try GET /health or POST /recommend."})

    def do_POST(self):
        if self.path != "/recommend":
            self.send_json(404, {"error": "Not found. Use POST /recommend."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json(200, recommend(body, self.recommender_data))
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})
        except Exception as error:
            self.send_json(500, {"error": f"Recommendation failed: {error}"})

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    print("Loading TravelWise data; the first startup can take a little while...")
    TravelWiseHandler.recommender_data = load_recommender_data()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), TravelWiseHandler)
    print(f"TravelWise backend is ready at http://127.0.0.1:{PORT}")
    print("Press Control+C to stop it.")
    server.serve_forever()


if __name__ == "__main__":
    main()
