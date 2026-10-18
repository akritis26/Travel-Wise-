from pathlib import Path

import joblib
import pandas as pd

survey_columns = [
    "form_a", "form_b", "form_c", "form_f", "form_g",
    "form_h", "form_i", "form_j", "form_r", "form_rr"
]

# Load the trained model and the prepared data.
model = joblib.load("travelwise_model.joblib")
data = pd.read_csv("training_data.csv", dtype=str)
destination_names = Path("destination_ids.txt").read_text(
    encoding="utf-8-sig"
).splitlines()

# Use the first traveler's survey answers as our demo profile.
first_user_id = data.iloc[0]["user_id"]
user_rows = data[data["user_id"] == first_user_id]
profile = user_rows.iloc[0][survey_columns].fillna("").astype(str).to_dict()

# Don't recommend places this traveler already swiped on.
already_seen = set(user_rows["destination_id"].astype(int))
candidate_ids = [
    destination_id
    for destination_id in range(len(destination_names))
    if destination_id not in already_seen
]

# Ask the model to score every destination the traveler hasn't seen.
candidates = pd.DataFrame([
    {**profile, "destination_id": str(destination_id)}
    for destination_id in candidate_ids
])
scores = model.predict_proba(candidates)[:, 1]

# Show the five highest-scoring destinations.
ranked = sorted(zip(scores, candidate_ids), reverse=True)[:5]

print("Demo traveler:", first_user_id)
print("\nTop 5 destination suggestions:")
for score, destination_id in ranked:
    print(f"- {destination_names[destination_id]} (model score: {score:.3f})")