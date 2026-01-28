import pandas as pd

# Load dataset
data = pd.read_csv("movies.csv")

# Show first 5 rows
print(data.head())

# Show column names
print(data.columns)

# Check missing values
print(data.isnull().sum())