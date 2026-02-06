import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================

df = pd.read_csv("features_engineered.csv")

X = df.drop(["TARGET", "ID"], axis=1)
y = df["TARGET"]

# =========================
# Train/test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# Feature Scaling
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Logistic Regression
# =========================

log_model = LogisticRegression(max_iter=5000)

log_model.fit(X_train_scaled, y_train)

log_auc = roc_auc_score(
    y_test,
    log_model.predict_proba(X_test_scaled)[:, 1]
)

print("📈 Logistic AUC:", log_auc)

# =========================
# XGBoost Hyperparameter Tuning
# =========================

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [200, 300],
    "subsample": [0.8, 1.0]
}

xgb = XGBClassifier()

grid = GridSearchCV(
    xgb,
    param_grid,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

xgb_auc = roc_auc_score(
    y_test,
    best_xgb.predict_proba(X_test)[:, 1]
)

print("🚀 Tuned XGBoost AUC:", xgb_auc)

# =========================
# SHAP Explainability
# =========================

X_sample = X_test.sample(500, random_state=42)

explainer = shap.Explainer(best_xgb.predict, X_sample)
shap_values = explainer(X_sample)

shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig("shap_summary.png")

# =========================
# Feature Importance Export
# =========================

importance = best_xgb.feature_importances_
feature_names = X.columns

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

fi_df = fi_df.sort_values(by="Importance", ascending=False)
fi_df.to_csv("feature_importance.csv", index=False)

print("✅ Feature importance saved!")
print("✅ Modeling complete!")