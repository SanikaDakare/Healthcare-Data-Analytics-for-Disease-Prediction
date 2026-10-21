# Task 3: Complete Code with Visualization Graphs
# Project: Healthcare Data Analytics for Disease Prediction
# Dataset: https://www.kaggle.com/datasets/prasad22/healthcare-dataset
#
# Run in Google Colab:
# 1. Download the CSV file from Kaggle.
# 2. Upload the CSV when prompted.
# 3. The script creates graphs and saves cleaned data.
#
# Important: This script explores the dataset and visualizes medical-condition
# categories. It does not make clinical diagnoses or train a prediction model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from google.colab import files

# Upload dataset
uploaded = files.upload()
file_name = next(iter(uploaded))
df = pd.read_csv(file_name)

print("Original dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# -------------------- DATA CLEANING --------------------
df.columns = df.columns.str.strip()
duplicates = df.duplicated().sum()
df = df.drop_duplicates().copy()
print(f"\nDuplicate rows removed: {duplicates}")

# Clean whitespace and empty strings
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].apply(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    df[col] = df[col].replace("", np.nan)

print("\nMissing values before filling:")
print(df.isnull().sum())

# Convert date-like columns, if present
for col in df.columns:
    if "date" in col.lower():
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().any():
            df[col] = parsed

# Fill numeric missing values with median
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Fill categorical missing values with mode
for col in df.select_dtypes(include=["object", "string"]).columns:
    if df[col].isna().any():
        modes = df[col].mode(dropna=True)
        df[col] = df[col].fillna(modes.iloc[0] if not modes.empty else "Unknown")

# -------------------- VISUALIZATION 1: MISSING VALUES --------------------
missing = df.isnull().sum()
missing = missing[missing > 0]
plt.figure(figsize=(10, 5))
if not missing.empty:
    missing.sort_values(ascending=False).plot(kind="bar")
    plt.title("Missing Values by Column Before Final Encoding")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45, ha="right")
else:
    plt.text(0.5, 0.5, "No missing values remain after cleaning",
             ha="center", va="center", fontsize=12)
    plt.title("Missing Values Check")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Preserve readable values for charts before encoding
chart_df = df.copy()

# -------------------- VISUALIZATION 2: MEDICAL CONDITION --------------------
condition_col = next(
    (c for c in chart_df.columns if c.strip().lower() == "medical condition"),
    None
)
if condition_col:
    counts = chart_df[condition_col].astype(str).value_counts()
    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.title("Number of Records by Medical Condition")
    plt.xlabel("Medical Condition")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.show()
else:
    print("\n' Medical Condition ' column was not found; skipping this graph.")

# -------------------- VISUALIZATION 3: AGE DISTRIBUTION --------------------
age_col = next((c for c in chart_df.columns if c.strip().lower() == "age"), None)
if age_col:
    ages = pd.to_numeric(chart_df[age_col], errors="coerce").dropna()
    plt.figure(figsize=(9, 5))
    plt.hist(ages, bins=15, edgecolor="black")
    plt.title("Age Distribution of Patients")
    plt.xlabel("Age")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.show()
else:
    print("\n' Age ' column was not found; skipping this graph.")

# -------------------- VISUALIZATION 4: GENDER DISTRIBUTION --------------------
gender_col = next((c for c in chart_df.columns if c.strip().lower() == "gender"), None)
if gender_col:
    gender_counts = chart_df[gender_col].astype(str).value_counts()
    plt.figure(figsize=(7, 5))
    gender_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
else:
    print("\n' Gender ' column was not found; skipping this graph.")

# -------------------- VISUALIZATION 5: BILLING AMOUNT --------------------
billing_col = next(
    (c for c in chart_df.columns if c.strip().lower() in
     ["billing amount", "billing_amount", "billingamount"]),
    None
)
if billing_col:
    billing = pd.to_numeric(chart_df[billing_col], errors="coerce").dropna()
    if not billing.empty:
        plt.figure(figsize=(9, 5))
        plt.hist(billing, bins=20, edgecolor="black")
        plt.title("Distribution of Billing Amount")
        plt.xlabel("Billing Amount")
        plt.ylabel("Number of Records")
        plt.tight_layout()
        plt.show()
else:
    print("\nBilling Amount column was not found; skipping this graph.")

# -------------------- ENCODE CATEGORICAL DATA --------------------
label_encoders = {}
label_mappings = []

for col in df.select_dtypes(include=["object", "string", "category"]).columns:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col].astype(str))
    label_encoders[col] = encoder
    for label, code in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        label_mappings.append({
            "Column": col,
            "Original Label": label,
            "Encoded Label": int(code)
        })

# Convert datetime columns to day counts
for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
    if df[col].notna().any():
        min_date = df[col].min()
        df[col] = (df[col] - min_date).dt.days
    else:
        df[col] = 0
    df[col] = df[col].fillna(0)

# -------------------- VISUALIZATION 6: NUMERIC CORRELATION --------------------
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 2:
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(min(12, 1 + corr.shape[1]), min(9, 1 + corr.shape[1])))
    image = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation Heatmap of Numeric/Encoded Columns")
    fig.colorbar(image, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.show()
else:
    print("\nNot enough numeric columns for a correlation heatmap.")

# -------------------- SAVE OUTPUT FILES --------------------
cleaned_path = "cleaned_healthcare_data_task3.csv"
mapping_path = "healthcare_label_mappings_task3.csv"
df.to_csv(cleaned_path, index=False)
pd.DataFrame(label_mappings).to_csv(mapping_path, index=False)

print("\nTask 3 completed.")
print("Cleaned dataset saved as:", cleaned_path)
print("Label mapping saved as:", mapping_path)
print("Graphs displayed above.")

files.download(cleaned_path)
files.download(mapping_path)
