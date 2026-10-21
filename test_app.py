from pathlib import Path
import json
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from travelwise_features import make_profile_features, SINGLE_CHOICE_COLUMNS

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

SINGLE_COLS = SINGLE_CHOICE_COLUMNS
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
    try:
        body = request.get_json(force=True)
        print("\n--- NEW REQUEST RECEIVED ---")

        # Frontend se aane wale data ko safely parse karna
        profile = {}
        for frontend_key, form_key in KEY_MAP.items():
            val = body.get(frontend_key, [])
            if not val:
                profile[form_key] = []
            elif isinstance(val, list):
                profile[form_key] = [str(v) for v in val]
            else:
                profile[form_key] = [str(val)]

        your_profile = pd.DataFrame([profile], columns=SURVEY_COLUMNS)
        your_features = make_profile_features(your_profile)

        your_single = ENCODER.transform(your_features[SINGLE_COLS])
        your_multi = csr_matrix(your_features[MULTI_COLS].astype(float).to_numpy())
        your_vector = hstack([your_single, your_multi], format="csr")

        k = min(50, len(USER_PROFILES))
        if k == 0:
            k = 1

        # Similarity calculate karna (Fix for weights = 0 bug)
        sims = cosine_similarity(your_vector, USER_VECTORS).ravel()
        top_idx = np.argsort(sims)[::-1][:k]

        weights = sims[top_idx].astype(float)
        weights[weights <= 0] = 1e-6
        weights = weights / weights.sum()

        weight_row = csr_matrix(
            (weights, (np.zeros(len(top_idx), dtype=int), top_idx)),
            shape=(1, len(USER_PROFILES)),
        )
        neighbor_yes = (weight_row @ YES_MATRIX).toarray().ravel()
        neighbor_seen = (weight_row @ SEEN_MATRIX).toarray().ravel()

        scores = (neighbor_yes + 5 * OVERALL_RATES) / (neighbor_seen + 5)

        # Top 5 rank karna (Fix for same results bug)
        eligible = (ALL_SEEN >= 10) & (neighbor_seen > 0)
        order = np.argsort(scores)[::-1]
        
        ranked = [int(i) for i in order if eligible[i]][:5]
        if len(ranked) < 5:
            for i in order:
                if len(ranked) >= 5:
                    break
                if int(i) not in ranked:
                    ranked.append(int(i))
        ranked = ranked[:5]

        results = []
        for dest_id in ranked:
            detail = DETAILS_BY_ID.get(str(dest_id), {})
            tags = []
            if detail.get("interest_tags"):
                try:
                    tags = json.loads(detail["interest_tags"])
                except (ValueError, TypeError):
                    pass
            results.append({
                "destination_name": DEST_NAMES[dest_id],
                "score": round(float(scores[dest_id]), 4),
                "overview": detail.get("overview", ""),
                "interest_tags": tags,
                "source_url": detail.get("source_url", ""),
            })

        return jsonify({
            "similar_travelers": k,
            "recommendations": results,
        })

    except Exception as e:
        # Agar koi error aaya toh terminal mein print hoga, app crash nahi hogi
        print("\n!!! ERROR IN BACKEND !!!")
        print(type(e).__name__, ":", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
