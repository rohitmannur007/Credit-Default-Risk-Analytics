import pandas as pd
from sqlalchemy import create_engine

# Path to Excel dataset
file_path = "data/default of credit card clients.xls"

# Read Excel (skip first row title)
df = pd.read_excel(file_path, header=1)

# Clean column names
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace(".", "_")

# Save to SQLite
engine = create_engine("sqlite:///credit_risk.db")
df.to_sql("credit_data", engine, if_exists="replace", index=False)

print("✅ Data loaded into credit_risk.db!")
print(df.head())