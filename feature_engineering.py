import pandas as pd
from sqlalchemy import create_engine
import sqlite3

# Connect to SQLite
engine = create_engine("sqlite:///credit_risk.db")

# Run SQL aggregation script
with open("sql_scripts/bureau_aggregation.sql") as f:
    sql_script = f.read()

raw_conn = sqlite3.connect("credit_risk.db")
raw_conn.executescript(sql_script)
raw_conn.commit()
raw_conn.close()

# Join aggregated features
query = """
SELECT c.*, p.avg_bill, p.avg_payment, p.total_delay
FROM credit_data c
LEFT JOIN payment_agg p
ON c.ID = p.ID
"""

df = pd.read_sql(query, engine)

# Rename target column
df.rename(columns={
    "default_payment_next_month": "TARGET"
}, inplace=True)

# ===============================
# Feature engineering (SAFE math)
# ===============================

# Avoid division by zero
df["UTILIZATION"] = df["avg_bill"] / (df["LIMIT_BAL"] + 1e-6)

df["PAYMENT_RATIO"] = df["avg_payment"] / (df["avg_bill"] + 1e-6)

# Remove infinities
df.replace([float("inf"), -float("inf")], 0, inplace=True)

# Fill missing values
df.fillna(0, inplace=True)

# Save final dataset
df.to_csv("features_engineered.csv", index=False)

print("✅ Feature engineering complete!")