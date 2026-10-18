import joblib
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from travelwise_features import make_features


data = pd.read_csv("training_data.csv", dtype=str)

# Convert survey answers into separate features the model can learn from.
X = make_features(data)
y = data["liked"].astype(int)

# Keep each traveler entirely in either training or evaluation data.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)
train_rows, test_rows = next(
    splitter.split(X, y, groups=data["user_id"])
)

# Train a model to predict whether a traveler liked a destination.
model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    LogisticRegression(max_iter=500, class_weight="balanced")
)
model.fit(X.iloc[train_rows], y.iloc[train_rows])

# Check how well it predicts reactions from held-out travelers.
predictions = model.predict(X.iloc[test_rows])
scores = model.predict_proba(X.iloc[test_rows])[:, 1]

print("Test accuracy:", round(accuracy_score(y.iloc[test_rows], predictions), 3))
print("Test AUC:", round(roc_auc_score(y.iloc[test_rows], scores), 3))

# Save this updated model over the earlier version.
joblib.dump(model, "travelwise_model.joblib")
print("Saved updated model: travelwise_model.joblib")