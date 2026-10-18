import ast
import csv
from pathlib import Path

# Find the CSV beside this script.
file_path = Path("Stravl_Travel_Preference_Data.csv")

# Turn text like "[837, 779]" from the CSV into a real Python list.
def read_id_list(value):
    if not value:
        return []

    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

user_count = 0
swipe_counts = {
    "yes_swipes": 0,
    "no_swipes": 0,
    "maybe_swipes": 0,
}
sample_rows = []

with file_path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    # Save the column names so we can display them.
    columns = reader.fieldnames

    # Read each traveler and count their swipe reactions.
    for row in reader:
        user_count += 1

        # Save a small preview of the first 3 travelers.
        if len(sample_rows) < 3:
            sample_rows.append({
                "id": row["id"],
                "yes_swipes": row["yes_swipes"],
                "no_swipes": row["no_swipes"],
                "maybe_swipes": row["maybe_swipes"],
            })

        # Add this traveler's reactions to the running totals.
        for column in swipe_counts:
            ids = read_id_list(row[column])
            swipe_counts[column] += len(ids)

# Show the results.
print("Columns:")
print(columns)

print("\nFirst 3 travelers (showing their swipe data):")
for traveler in sample_rows:
    print(traveler)

print("\nTotal travelers:", user_count)
print("Total swipe reactions:", swipe_counts)
# Open the file that connects destination IDs to names.
mapping_file = Path("destination_ids.txt")

with mapping_file.open(encoding="utf-8-sig") as file:
    mapping_lines = file.readlines()

print("\nDestination ID file:")
print("Number of lines:", len(mapping_lines))
print("First 10 lines:")
for line in mapping_lines[:10]:
    print(line.strip())
    # Remove line breaks so each list position is a clean destination name.
destination_names = [line.strip() for line in mapping_lines]

# Reopen the CSV and get its first traveler.
with file_path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    first_traveler = next(reader)

# Show that traveler's yes and no swipes as destination names.
for column in ("yes_swipes", "no_swipes"):
    print(f"\n{column}:")
    ids = read_id_list(first_traveler[column])

    for destination_id in ids:
        print(f"- ID {destination_id}: {destination_names[destination_id]}")
        
        # Create a new file for the model's yes/no examples.
interactions_file = Path("interactions.csv")
interaction_count = 0
sample_interactions = []

with file_path.open(encoding="utf-8-sig", newline="") as input_file:
    reader = csv.DictReader(input_file)

    with interactions_file.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["user_id", "destination_id", "liked"])

        for traveler in reader:
            # A yes reaction becomes 1, meaning liked.
            # A no reaction becomes 0, meaning not liked.
            for column, liked_value in [("yes_swipes", 1), ("no_swipes", 0)]:
                ids = read_id_list(traveler[column])

                for destination_id in ids:
                    example = [traveler["id"], destination_id, liked_value]
                    writer.writerow(example)
                    interaction_count += 1

                    # Keep just 5 examples to show in the terminal.
                    if len(sample_interactions) < 5:
                        sample_interactions.append(example)

print("\nCreated:", interactions_file.name)
print("Yes/no examples:", interaction_count)
print("First 5 examples:")
for example in sample_interactions:
    print(example)
    # These are the survey-answer columns whose meanings are documented.
survey_columns = [
    "form_a", "form_b", "form_c", "form_f", "form_g",
    "form_h", "form_i", "form_j", "form_r", "form_rr"
]

training_file = Path("training_data.csv")
training_count = 0
training_samples = []

with file_path.open(encoding="utf-8-sig", newline="") as input_file:
    reader = csv.DictReader(input_file)

    with training_file.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)

        # Each output row will contain the traveler, their survey answers,
        # the destination they reacted to, and their yes/no label.
        writer.writerow(
            ["user_id"] + survey_columns + ["destination_id", "liked"]
        )

        for traveler in reader:
            for column, liked_value in [("yes_swipes", 1), ("no_swipes", 0)]:
                ids = read_id_list(traveler[column])

                for destination_id in ids:
                    training_row = [traveler["id"]]

                    for survey_column in survey_columns:
                        training_row.append(traveler[survey_column])

                    training_row.append(destination_id)
                    training_row.append(liked_value)
                    writer.writerow(training_row)
                    training_count += 1

                    if len(training_samples) < 3:
                        training_samples.append(training_row)

print("\nCreated:", training_file.name)
print("Training examples:", training_count)
print("Training columns:", ["user_id"] + survey_columns + ["destination_id", "liked"])
print("First 3 training examples:")
for example in training_samples:
    print(example)