import pandas as pd

df = pd.read_csv("data/diabetes.csv")
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nZeros in each column:\n", (df == 0).sum())