from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from travelwise_features import make_profile_features, SINGLE_CHOICE_COLUMNS


survey_columns = [
    "form_a", "form_b", "form_c", "form_f", "form_g",
    "form_h", "form_i", "form_j", "form_r", "form_rr"
]


def ask_codes(question, allowed_codes, multiple=False, optional=False):
    while True:
        print(question)
        answer = input("> ").strip()
        if not answer and optional:
            return []
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
        return choices


profile = {
    "form_a": ask_codes(
        "Age: 0=0-19, 1=20-39, 2=40-59, 3=60+:",
        ["0", "1", "2", "3"]
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
        "(multiple allowed, comma-separated):",
        ["0", "1", "2", "3", "4", "5", "6", "7"],
        multiple=True
    ),
    "form_g": ask_codes(
        "Scenery: 0=urban, 1=rural, 2=sea, 3=mountain, 4=lake, "
        "5=desert, 6=plains, 7=jungle (multiple allowed, comma-separated):",
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


if profile["form_r"] == ["0"]:
    print("You chose anywhere, so region choices aren't needed.")
    profile["form_rr"] = []
else:
    profile["form_rr"] = ask_codes(
        "Regions: e=Europe, n=North America, c=Caribbean, a=Asia, "
        "s=South America, m=Middle East, f=Africa, o=Oceania "
        "(multiple allowed, comma-separated; press Enter to skip):",
        ["e", "n", "c", "a", "s", "m", "f", "o"],
        multiple=True,
        optional=True
    )


print("\nLoading the traveler data...")
data = pd.read_csv("training_data.csv", dtype=str)
destination_names = Path("destination_ids.txt").read_text(
    encoding="utf-8-sig"
).splitlines()


user_profiles = (
    data[["user_id"] + survey_columns]
    .drop_duplicates("user_id")
    .reset_index(drop=True)
)


profile_features = make_profile_features(user_profiles)

single_choice_columns = SINGLE_CHOICE_COLUMNS
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


print("Finding travelers with similar answers...")
neighbor_count = min(50, len(user_profiles))

similarities = cosine_similarity(your_vector, user_vectors).ravel()
top_rows = np.argsort(similarities)[::-1][:neighbor_count]

weights = similarities[top_rows].astype(float)
weights[weights <= 0] = 1e-6
weights = weights / weights.sum()


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


weight_matrix = csr_matrix(
    (weights, (np.zeros(len(top_rows), dtype=int), top_rows)),
    shape=(1, len(user_profiles))
)

neighbor_yes = (weight_matrix @ yes_matrix).toarray().ravel()
neighbor_seen = (weight_matrix @ seen_matrix).toarray().ravel()


all_yes = np.asarray(yes_matrix.sum(axis=0)).ravel()
all_seen = np.asarray(seen_matrix.sum(axis=0)).ravel()

average_yes_rate = all_yes.sum() / max(all_seen.sum(), 1)
prior_strength = 20

overall_rates = (
    all_yes + prior_strength * average_yes_rate
) / (
    all_seen + prior_strength
)

scores = (neighbor_yes + 5 * overall_rates) / (neighbor_seen + 5)

eligible = (all_seen >= 10) & (neighbor_seen > 0)

order = np.argsort(scores)[::-1]
ranked = [int(i) for i in order if eligible[i]][:5]
if len(ranked) < 5:
    for i in order:
        if len(ranked) >= 5:
            break
        if int(i) not in ranked:
            ranked.append(int(i))
ranked = ranked[:5]


print(f"\nYour top 5 suggestions, based on {neighbor_count} similar travelers:")
for number, destination_id in enumerate(ranked, start=1):
    print(
        f"{number}. {destination_names[destination_id]} "
        f"(model score: {scores[destination_id]:.3f})"
    )


details = pd.read_csv(
    "destination_details.csv",
    dtype={"destination_id": str}
).fillna("")
details_by_id = details.set_index("destination_id")


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

    destination_id = ranked[int(choice) - 1]
    detail = details_by_id.loc[str(destination_id)]

    print(f"\nMore about {detail['destination_name']}:")

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
