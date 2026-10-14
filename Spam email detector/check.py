import pandas as pd
import os

print("Current folder:", os.getcwd())
print("Files here:", os.listdir())

df = pd.read_csv("emails.csv")
print("Columns:", df.columns.tolist())
print(df.head())
