import ast

import pandas as pd


SINGLE_CHOICE_COLUMNS = [
    "form_a", "form_b", "form_c", "form_h", "form_i", "form_j", "form_r"
]

MULTI_CHOICE_OPTIONS = {
    "form_f": ["0", "1", "2", "3", "4", "5", "6", "7"],
    "form_g": ["0", "1", "2", "3", "4", "5", "6", "7"],
    "form_rr": ["e", "n", "c", "a", "s", "m", "f", "o"],
}


def read_choices(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return []
    try:
        choices = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(choices, (list, tuple)):
        choices = [choices]
    return [str(choice) for choice in choices]


def make_profile_features(rows):
    features = pd.DataFrame(index=rows.index)
    for column in SINGLE_CHOICE_COLUMNS:
        choices = rows[column].apply(read_choices)
        features[column] = choices.apply(
            lambda values: values[0] if values else ""
        )
    for column, options in MULTI_CHOICE_OPTIONS.items():
        choices = rows[column].apply(read_choices)
        for option in options:
            features[f"{column}_{option}"] = choices.apply(
                lambda values: "1" if option in values else "0"
            )
    return features


def make_features(rows):
    features = make_profile_features(rows)
    features["destination_id"] = rows["destination_id"].fillna("").astype(str)
    return features
