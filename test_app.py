from pathlib import Path
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from travelwise_features import make_profile_features

app = Flask(__name__)

SURVEY_COLUMNS = [
    "form_a", "form_b", "form_c", "form_f", "form_g",
    "form_h", "form_i", "form_j", "form_r", "form_rr"
]

KEY_MAP = {
    "age": "form_a",
    "budget": "form_b",
    "season": "form_c",
    "experiences": "form_f",
    "scenery": "form_g",
    "activity": "form_h",
    "safety": "form_i",
    "popularity": "form_j",
    "where": "form_r",
    "regions": "form_rr"
}

print("Loading real data...")
DATA = pd.read_csv("training_data.csv", dtype=str)
DEST_NAMES = Path("destination_ids.txt").read_text(encoding="utf-8-sig").splitlines()

try:
    DETAILS = pd.read_csv("destination_details.csv", dtype={"destination_id": str}).fillna("")
    DETAILS_BY_ID = DETAILS.set_index("destination_id").to_dict(orient="index")
except FileNotFoundError:
    DETAILS_BY_ID = {}

USER_PROFILES = DATA[["user_id"] + SURVEY_COLUMNS].drop_duplicates("user_id").reset_index(drop=True)
PROFILE_FEATURES = make_profile_features(USER_PROFILES)

SINGLE_COLS = ["form_b", "form_c", "form_h", "form_i", "form_j", "form_r"]
MULTI_COLS = [c for c in PROFILE_FEATURES.columns if c not in SINGLE_COLS]

ENCODER = OneHotEncoder(handle_unknown="ignore")
USER_SINGLE = ENCODER.fit_transform(PROFILE_FEATURES[SINGLE_COLS])
USER_MULTI = csr_matrix(PROFILE_FEATURES[MULTI_COLS].astype(float).to_numpy())
USER_VECTORS = hstack([USER_SINGLE, USER_MULTI], format="csr")

USER_TO_ROW = {uid: i for i, uid in enumerate(USER_PROFILES["user_id"])}
INTERACTION_ROWS = DATA["user_id"].map(USER_TO_ROW).to_numpy(dtype=int)
DEST_COLS = DATA["destination_id"].astype(int).to_numpy()
LIKED_VALS = DATA["liked"].astype(float).to_numpy()

N_DEST = len(DEST_NAMES)
YES_MATRIX = csr_matrix((LIKED_VALS, (INTERACTION_ROWS, DEST_COLS)), shape=(len(USER_PROFILES), N_DEST))
SEEN_MATRIX = csr_matrix((np.ones(len(DATA)), (INTERACTION_ROWS, DEST_COLS)), shape=(len(USER_PROFILES), N_DEST))

ALL_YES = np.asarray(YES_MATRIX.sum(axis=0)).ravel()
ALL_SEEN = np.asarray(SEEN_MATRIX.sum(axis=0)).ravel()
AVG_RATE = ALL_YES.sum() / max(ALL_SEEN.sum(), 1)
PRIOR = 20
OVERALL_RATES = (ALL_YES + PRIOR * AVG_RATE) / (ALL_SEEN + PRIOR)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.get_json(force=True)
    
    profile = {}
    for frontend_key, form_key in KEY_MAP.items():
        val = body.get(frontend_key, "")
        if isinstance(val, list):
            profile[form_key] = str(val)
        elif val:
            profile[form_key] = str([val])
        else:
            profile[form_key] = ""

    your_profile = pd.DataFrame([profile], columns=SURVEY_COLUMNS)
    your_features = make_profile_features(your_profile)

    your_single = ENCODER.transform(your_features[SINGLE_COLS])
    your_multi = csr_matrix(your_features[MULTI_COLS].astype(float).to_numpy())
    your_vector = hstack([your_single, your_multi], format="csr")

    k = min(50, len(USER_PROFILES))
    if k == 0:
        k = 1
        
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    knn.fit(USER_VECTORS)
    dist, idx = knn.kneighbors(your_vector)
    idx = idx[0]
    weights = np.clip(1 - dist[0], 0, 1)

    weight_row = csr_matrix((weights, (np.zeros(len(idx), dtype=int), idx)), shape=(1, len(USER_PROFILES)))
    neighbor_yes = (weight_row @ YES_MATRIX).toarray().ravel()
    neighbor_seen = (weight_row @ SEEN_MATRIX).toarray().ravel()

    scores = (neighbor_yes + 5 * OVERALL_RATES) / (neighbor_seen + 5)
    
    # ---------------------------------------------------------
    # THE FIX: "Chill Mode" - Hum kisi bhi destination ko skip nahi karenge
    # ---------------------------------------------------------
    eligible = np.ones(N_DEST, dtype=bool) 
    
    scores[~eligible] = -1

    ranked = sorted(zip(scores, range(N_DEST)), reverse=True)[:5]

    results = []
    for score, dest_id in ranked:
        if score < 0: continue
        detail = DETAILS_BY_ID.get(str(dest_id), {})
        tags = []
        if detail.get("interest_tags"):
            try: tags = json.loads(detail["interest_tags"])
            except: pass
            
        results.append({
            "destination_name": DEST_NAMES[dest_id],
            "score": round(float(score), 4),
            "overview": detail.get("overview", ""),
            "interest_tags": tags,
            "source_url": detail.get("source_url", "")
        })

    return jsonify({
        "similar_travelers": k,
        "recommendations": results
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)