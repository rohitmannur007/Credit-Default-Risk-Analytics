import pandas as pd
import numpy as np

def psi(expected, actual):
    e = np.histogram(expected, bins=10)[0] / len(expected)
    a = np.histogram(actual, bins=10)[0] / len(actual)
    return np.sum((a - e) * np.log((a + 1e-6) / (e + 1e-6)))

df = pd.read_csv("features_engineered.csv")

split = int(len(df) * 0.7)

train = df.iloc[:split]
test = df.iloc[split:]

for col in ["LIMIT_BAL", "UTILIZATION", "AGE"]:
    print(col, "PSI:", psi(train[col], test[col]))

print("✅ Monitoring complete!")