import pandas as pd
df = pd.read_csv("data/fraud_data.csv")
print(df.drop(columns=["is_fraud"], errors="ignore").shape[1])
