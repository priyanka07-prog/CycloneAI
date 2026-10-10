
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from data_loader import load_cyclone_data


# 1. Load the historical dataset
df = load_cyclone_data()

# Prefer WMO measurements for the global dataset when available
wind_col = (
    "WMO_WIND" if "WMO_WIND" in df.columns
    else "USA_WIND"
)

pressure_col = (
    "WMO_PRES" if "WMO_PRES" in df.columns
    else "USA_PRES"
)

# 2. Convert measurements to numeric values
for col in ["LAT", "LON", wind_col, pressure_col]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(-999, np.nan)

df["ISO_TIME"] = pd.to_datetime(
    df["ISO_TIME"], errors="coerce"
)

# 3. Prepare the training data
work = df.dropna(
    subset=["SID", "ISO_TIME", "LAT", "LON", wind_col]
).copy()

# Remove invalid or implausible wind measurements
work = work[
    (work[wind_col] > 0) &
    (work[wind_col] <= 250)
].copy()

work["MONTH"] = work["ISO_TIME"].dt.month

# These are broad experimental intensity classes,
# not official storm classifications.
work["INTENSITY_CLASS"] = np.select(
    [
        work[wind_col] < 34,
        work[wind_col] < 64,
    ],
    [
        "Lower intensity (<34 kt)",
        "Tropical-storm-force (34-63 kt)",
    ],
    default="Hurricane-force (>=64 kt)",
)

# 4. Select input features.
# IMPORTANT: The target wind column is NOT an input feature.
feature_cols = ["LAT", "LON", "MONTH"]

if pressure_col in work.columns:
    feature_cols.append(pressure_col)

X = work[feature_cols]
y = work["INTENSITY_CLASS"]
groups = work["SID"]

# 5. Split by cyclone, not randomly by individual observation.
# This helps prevent observations from the same storm
# appearing in both training and testing.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# 6. Build and train the model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

# 7. Evaluate on unseen cyclone groups
predictions = model.predict(X_test)

print("Training completed!")
print("Wind column:", wind_col)
print("Features:", feature_cols)
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))
print("\nAccuracy:", round(
    accuracy_score(y_test, predictions), 3
))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

# 8. Save the trained model
project_root = Path(__file__).resolve().parents[1]
artifact_dir = project_root / "artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "feature_columns": feature_cols,
        "wind_column": wind_col,
        "pressure_column": pressure_col,
        "target_name": "INTENSITY_CLASS",
    },
    artifact_dir / "cyclone_intensity_model.joblib",
)

print("\nModel saved successfully!")
