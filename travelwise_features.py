import ast

import pandas as pd


SINGLE_CHOICE_COLUMNS = [
    "form_b", "form_c", "form_h", "form_i", "form_j", "form_r"
]

MULTI_CHOICE_OPTIONS = {
    "form_a": ["0", "1", "2", "3"],
    "form_f": ["0", "1", "2", "3", "4", "5", "6", "7"],
    "form_g": ["0", "1", "2", "3", "4", "5", "6", "7"],
    "form_rr": ["e", "n", "c", "a", "s", "m", "f", "o"],
}


def read_choices(value):
    if pd.isna(value) or not str(value).strip():
        return []

    try:
        choices = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []

    if not isinstance(choices, list):
        choices = [choices]

    return [str(choice) for choice in choices]


def make_profile_features(rows):
    features = pd.DataFrame(index=rows.index)

    # Keep each single-choice answer as its own category.
    for column in SINGLE_CHOICE_COLUMNS:
        choices = rows[column].apply(read_choices)
        features[column] = choices.apply(
            lambda values: values[0] if values else ""
        )

    # Give each multi-choice option its own yes/no feature.
    for column, options in MULTI_CHOICE_OPTIONS.items():
        choices = rows[column].apply(read_choices)

        for option in options:
            features[f"{column}_{option}"] = choices.apply(
                lambda values: "1" if option in values else "0"
            )

    return features


def make_features(rows):
    # Keep this for the original classifier script.
    features = make_profile_features(rows)
    features["destination_id"] = rows["destination_id"].fillna("").astype(str)
    return features