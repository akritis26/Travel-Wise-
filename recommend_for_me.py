from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from travelwise_features import make_profile_features


survey_columns = [
    "form_a", "form_b", "form_c", "form_f", "form_g",
    "form_h", "form_i", "form_j", "form_r", "form_rr"
]


def ask_codes(question, allowed_codes, multiple=False, optional=False):
    while True:
        print(question)
        answer = input("> ").strip()

        if not answer and optional:
            return ""

        choices = list(dict.fromkeys(
            choice.strip()
            for choice in answer.split(",")
            if choice.strip()
        ))

        if not choices:
            print("Please enter at least one choice.")
            continue

        invalid = [choice for choice in choices if choice not in allowed_codes]
        if invalid:
            print("Please use only the codes shown in the question.")
            continue

        if not multiple and len(choices) != 1:
            print("Please enter exactly one choice.")
            continue

        return str(choices)


# Ask the traveler for their preferences.
profile = {
    "form_a": ask_codes(
        "Age: 0=0-19, 1=20-39, 2=40-59, 3=60+:",
        ["0", "1", "2", "3"],
        multiple=False
    ),
    "form_b": ask_codes(
        "Budget per person/night (approx. INR): "
        "0=under ₹4,800, 1=₹4,800-₹9,400, "
        "2=₹9,500-₹23,700, 3=₹28,500+:",
        ["0", "1", "2", "3"]
    ),
    "form_c": ask_codes(
        "Season: 0=winter, 1=spring, 2=summer, 3=fall:",
        ["0", "1", "2", "3"]
    ),
    "form_f": ask_codes(
        "Experiences: 0=beach, 1=adventure, 2=nature, 3=culture, "
        "4=nightlife, 5=history, 6=shopping, 7=cuisine "
        "(multiple allowed):",
        ["0", "1", "2", "3", "4", "5", "6", "7"],
        multiple=True
    ),
    "form_g": ask_codes(
        "Scenery: 0=urban, 1=rural, 2=sea, 3=mountain, 4=lake, "
        "5=desert, 6=plains, 7=jungle (multiple allowed):",
        ["0", "1", "2", "3", "4", "5", "6", "7"],
        multiple=True
    ),
    "form_h": ask_codes(
        "Activity: 0=chill, 1=balanced, 2=active:",
        ["0", "1", "2"]
    ),
    "form_i": ask_codes(
        "Safety: 0=very safety conscious, 1=balanced, "
        "2=ready for anything:",
        ["0", "1", "2"]
    ),
    "form_j": ask_codes(
        "Popularity: 0=off the beaten path, 1=classic, "
        "2=mainstream/trendy:",
        ["0", "1", "2"]
    ),
    "form_r": ask_codes(
        "Where: 0=anywhere, 1=specific regions:",
        ["0", "1"]
    )
}


# Ask about regions only if the traveler chose specific regions.
if profile["form_r"] == "['0']":
    print("You chose anywhere, so region choices aren't needed.")
    profile["form_rr"] = ""
else:
    profile["form_rr"] = ask_codes(
        "Regions: e=Europe, n=North America, c=Caribbean, a=Asia, "
        "s=South America, m=Middle East, f=Africa, o=Oceania "
        "(multiple allowed):",
        ["e", "n", "c", "a", "s", "m", "f", "o"],
        multiple=True,
        optional=True
    )


# Load the prepared traveler reactions and destination names.
print("\nLoading the traveler data...")
data = pd.read_csv("training_data.csv", dtype=str)
destination_names = Path("destination_ids.txt").read_text(
    encoding="utf-8-sig"
).splitlines()


# Keep one preference profile for each traveler.
user_profiles = (
    data[["user_id"] + survey_columns]
    .drop_duplicates("user_id")
    .reset_index(drop=True)
)


# Turn survey answers into features that can be compared.
profile_features = make_profile_features(user_profiles)

single_choice_columns = [
    "form_b", "form_c", "form_h", "form_i", "form_j", "form_r"
]

multi_choice_columns = [
    column for column in profile_features.columns
    if column not in single_choice_columns
]

encoder = OneHotEncoder(handle_unknown="ignore")

user_single_features = encoder.fit_transform(
    profile_features[single_choice_columns]
)

user_multi_features = csr_matrix(
    profile_features[multi_choice_columns].astype(float).to_numpy()
)

user_vectors = hstack(
    [user_single_features, user_multi_features],
    format="csr"
)


# Convert the answers entered above into the same feature format.
your_profile = pd.DataFrame([profile], columns=survey_columns)
your_features = make_profile_features(your_profile)

your_single_features = encoder.transform(
    your_features[single_choice_columns]
)

your_multi_features = csr_matrix(
    your_features[multi_choice_columns].astype(float).to_numpy()
)

your_vector = hstack(
    [your_single_features, your_multi_features],
    format="csr"
)


# Find up to 50 travelers whose answers are closest to yours.
print("Finding travelers with similar answers...")
neighbor_count = min(50, len(user_profiles))

neighbor_finder = NearestNeighbors(
    n_neighbors=neighbor_count,
    metric="cosine",
    algorithm="brute"
)

neighbor_finder.fit(user_vectors)
distances, neighbor_rows = neighbor_finder.kneighbors(your_vector)

neighbor_rows = neighbor_rows[0]
weights = np.clip(1 - distances[0], 0, 1)


# Create tables showing which travelers liked or reacted to each destination.
user_to_row = {
    user_id: row_number
    for row_number, user_id in enumerate(user_profiles["user_id"])
}

interaction_rows = data["user_id"].map(user_to_row).to_numpy(dtype=int)
destination_columns = data["destination_id"].astype(int).to_numpy()
liked_values = data["liked"].astype(float).to_numpy()

matrix_shape = (len(user_profiles), len(destination_names))

yes_matrix = csr_matrix(
    (liked_values, (interaction_rows, destination_columns)),
    shape=matrix_shape
)

seen_matrix = csr_matrix(
    (np.ones(len(data)), (interaction_rows, destination_columns)),
    shape=matrix_shape
)


# Give more influence to travelers whose answers are closer to yours.
weight_matrix = csr_matrix(
    (weights, (np.zeros(len(neighbor_rows), dtype=int), neighbor_rows)),
    shape=(1, len(user_profiles))
)

neighbor_yes = (weight_matrix @ yes_matrix).toarray().ravel()
neighbor_seen = (weight_matrix @ seen_matrix).toarray().ravel()


# Count each destination's positive reactions and total reactions.
all_yes = np.asarray(yes_matrix.sum(axis=0)).ravel()
all_seen = np.asarray(seen_matrix.sum(axis=0)).ravel()

# Smooth destination rates toward the overall average. This prevents a
# destination with only a few positive ratings from looking guaranteed.
average_yes_rate = all_yes.sum() / all_seen.sum()
prior_strength = 20

overall_rates = (
    all_yes + prior_strength * average_yes_rate
) / (
    all_seen + prior_strength
)

# Combine similar travelers' feedback with the overall destination rate.
scores = (neighbor_yes + 5 * overall_rates) / (neighbor_seen + 5)

# Only recommend destinations with enough total feedback and some feedback
# from travelers similar to you.
eligible = (all_seen >= 10) & (neighbor_seen > 0)
scores[~eligible] = -1


# Print the five eligible destinations with the highest scores.
ranked = sorted(
    zip(scores, range(len(destination_names))),
    reverse=True
)[:5]

print(f"\nYour top 5 suggestions, based on {neighbor_count} similar travelers:")

for number, (score, destination_id) in enumerate(ranked, start=1):
    print(
        f"{number}. {destination_names[destination_id]} "
        f"(model score: {score:.3f})"
    )


# Load the destination details so a traveler can choose one recommendation.
details = pd.read_csv(
    "destination_details.csv",
    dtype={"destination_id": str}
).fillna("")
details_by_id = details.set_index("destination_id")


# Ask which recommendation the traveler wants to learn more about.
while True:
    choice = input(
        f"\nType a recommendation number (1-{len(ranked)}) for details, "
        "or press Enter to finish: "
    ).strip()

    if not choice:
        break

    if not choice.isdigit() or not 1 <= int(choice) <= len(ranked):
        print(f"Please enter a number from 1 to {len(ranked)}.")
        continue

    _, destination_id = ranked[int(choice) - 1]
    detail = details_by_id.loc[str(destination_id)]

    print(f"\nMore about {detail['destination_name']}:")

    # Only show article text when its title clearly matches the place name.
    place_name = detail["destination_name"].split(",", 1)[0].strip().casefold()
    article_title = detail["wikipedia_title"].strip().casefold()
    title_matches_place = (
        article_title == place_name
        or article_title.startswith(f"{place_name} (")
    )
    safe_to_show_details = (
        detail["match_status"] == "matched" and title_matches_place
    )

    if safe_to_show_details and detail["overview"]:
        print("Overview:", detail["overview"])
        if detail["interest_tags"]:
            tags = ", ".join(json.loads(detail["interest_tags"]))
            print("Interest tags:", tags)
    else:
        print("No confirmed destination overview is available for this match.")
        if detail["wikipedia_title"]:
            print("Possible article title:", detail["wikipedia_title"])

    if detail["source_url"]:
        print("Source:", detail["source_url"])

    if not safe_to_show_details:
        print("Note: check the source before using it; this match is uncertain.")

    break
