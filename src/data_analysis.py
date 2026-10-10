
from data_loader import load_cyclone_data

df = load_cyclone_data()

print("Dataset shape:", df.shape)

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values in the most affected columns:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStorm names:")
print(df["NAME"].dropna().unique()[:10])
