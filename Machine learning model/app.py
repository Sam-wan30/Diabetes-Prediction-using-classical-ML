"""
Pima Indians Diabetes Dataset - Data Loading and Basic Information
==================================================================
This script loads the Pima Indians Diabetes dataset and displays
comprehensive basic information about the dataset.
"""

# Import necessary libraries
# pandas: Used for data manipulation and analysis (DataFrames)
import pandas as pd

# Step 1: Load the dataset from CSV file
# pd.read_csv() reads a comma-separated values file into a pandas DataFrame
# The file path is relative to the script location
# 'data/diabetes.csv' points to the diabetes dataset in the data folder
df = pd.read_csv('data/diabetes.csv')

# Step 2: Display the shape of the dataset
# .shape returns a tuple (number of rows, number of columns)
# This gives us the dimensions of our dataset
print("=" * 60)
print("DATASET SHAPE (Rows, Columns)")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print()

# Step 3: Display the first few rows of the dataset
# .head() displays the first 5 rows by default (or specify number in parentheses)
# This helps us get a quick visual overview of the data structure
print("=" * 60)
print("FIRST 5 ROWS OF THE DATASET")
print("=" * 60)
print(df.head())
print()

# Step 4: Display column names
# .columns returns an Index object containing column names
# This shows us all the features/variables in our dataset
print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())
print()

# Step 5: Display data types of each column
# .dtypes returns the data type of each column
# This helps identify which columns are numeric, categorical, etc.
print("=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)
print()

# Step 6: Display comprehensive dataset information
# .info() provides a concise summary including:
# - Total number of entries
# - Column names
# - Non-null counts (helps identify missing values)
# - Data types
# - Memory usage
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(df.info())
print()

# Step 7: Check for missing values
# .isnull() returns a DataFrame of True/False for each cell
# .sum() counts the number of True values (missing values) per column
# This is crucial for data quality assessment
print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing_values = df.isnull().sum()
print(missing_values)
if missing_values.sum() == 0:
    print("\n✓ No missing values found in the dataset!")
else:
    print(f"\n⚠ Total missing values: {missing_values.sum()}")
print()

# Step 8: Display statistical summary
# .describe() generates descriptive statistics for numeric columns:
# - count: number of non-null values
# - mean: average value
# - std: standard deviation (measure of spread)
# - min: minimum value
# - 25%: first quartile (25th percentile)
# - 50%: median (50th percentile)
# - 75%: third quartile (75th percentile)
# - max: maximum value
print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())
print()

# Step 9: Display basic statistics for the target variable
# The 'Outcome' column is our target variable (0 = no diabetes, 1 = diabetes)
# This shows the distribution of classes in our dataset
print("=" * 60)
print("TARGET VARIABLE DISTRIBUTION (Outcome)")
print("=" * 60)
print(df['Outcome'].value_counts())
print()
print("Percentage distribution:")
print(df['Outcome'].value_counts(normalize=True) * 100)
print()

# Step 10: Display memory usage
# .memory_usage() shows how much memory each column uses
# deep=True gives accurate memory usage (slower but more precise)
print("=" * 60)
print("MEMORY USAGE")
print("=" * 60)
print(df.memory_usage(deep=True))
print(f"\nTotal memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print()

# Step 11: Display unique value counts for each column
# .nunique() returns the number of unique values in each column
# This helps identify categorical vs continuous variables
print("=" * 60)
print("UNIQUE VALUES PER COLUMN")
print("=" * 60)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
print()

# Step 12: Display sample rows (random selection)
# .sample() returns random rows from the dataset
# This gives us a different perspective than just the first rows
print("=" * 60)
print("RANDOM SAMPLE (5 ROWS)")
print("=" * 60)
print(df.sample(5))
print()

print("=" * 60)
print("DATA LOADING AND BASIC INFORMATION COMPLETE!")
print("=" * 60)

