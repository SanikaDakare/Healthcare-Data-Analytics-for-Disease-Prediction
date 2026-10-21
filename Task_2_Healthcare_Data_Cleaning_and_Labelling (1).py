# Task 2: Data Cleaning and Labelling
# Project: Healthcare Data Analytics for Disease Prediction
# Dataset: https://www.kaggle.com/datasets/prasad22/healthcare-dataset
#
# Before running:
# 1. Download the dataset CSV from Kaggle.
# 2. Put the CSV in the same folder as this script, or update DATASET_PATH below.
#
# Note: This dataset's "Medical Condition" column can be used as a target label
# for a later prediction task. This script encodes categories for analysis; it
# does not train a disease-prediction model.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Update this path if your downloaded CSV has a different filename.
DATASET_PATH = Path("healthcare_dataset.csv")
OUTPUT_PATH = Path("cleaned_healthcare_data.csv")

# Load the dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {DATASET_PATH}. Download the CSV from Kaggle "
        "and place it beside this script, or change DATASET_PATH."
    )

df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

# Clean column names and remove duplicate records
df.columns = df.columns.str.strip()
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()
print(f"\nDuplicate rows removed: {duplicate_count}")

# Clean whitespace and replace empty strings with missing values
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].apply(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    df[col] = df[col].replace("", np.nan)

print("\nMissing values before filling:")
print(df.isnull().sum())

# Convert columns containing dates, if any
for col in df.columns:
    if "date" in col.lower():
        parsed = pd.to_datetime(df[col], errors="coerce")
        # Convert only when at least one value was parsed successfully
        if parsed.notna().any():
            df[col] = parsed

# Fill missing numeric values using the median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

# Fill missing categorical values using the most frequent value
for col in df.select_dtypes(include=["object", "string"]).columns:
    if df[col].isna().any():
        modes = df[col].mode(dropna=True)
        if not modes.empty:
            df[col] = df[col].fillna(modes.iloc[0])
        else:
            df[col] = df[col].fillna("Unknown")

# Convert date columns into numeric day counts
for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
    if df[col].notna().any():
        minimum_date = df[col].min()
        df[col] = (df[col] - minimum_date).dt.days
    else:
        df[col] = 0
    df[col] = df[col].fillna(0)

# Encode categorical text columns as numeric labels and save mappings
label_encoders = {}
label_mappings = []

for col in df.select_dtypes(include=["object", "string", "category"]).columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    label_encoders[col] = encoder
    for original_label, encoded_label in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        label_mappings.append({
            "Column": col,
            "Original Label": original_label,
            "Encoded Label": int(encoded_label)
        })

print("\nCleaned and labelled data (first five rows):")
print(df.head())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nFinal dataset shape:", df.shape)

# Save the cleaned dataset and category-to-number mapping
df.to_csv(OUTPUT_PATH, index=False)
mapping_path = Path("healthcare_label_mappings.csv")
pd.DataFrame(label_mappings).to_csv(mapping_path, index=False)

print(f"\nCleaned dataset saved to: {OUTPUT_PATH}")
print(f"Label mappings saved to: {mapping_path}")
print("\nTask 2 completed.")
