from joblib import load
import pandas as pd

bundle = load("ml/models/random_forest.joblib")
print("Model classes:", bundle["model"].classes_)
print("Number of features:", len(bundle["features"]))

df = pd.read_csv("ml/data/test.csv", nrows=5000, low_memory=False)
df.columns = df.columns.str.strip()

missing = [f for f in bundle["features"] if f not in df.columns]
print("Features missing from the CSV:", len(missing), missing[:10])
print(df["Label"].astype(str).str.strip().value_counts().head(10))