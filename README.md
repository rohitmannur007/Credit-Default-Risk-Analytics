# End-to-End Credit Default Risk Pipeline

This project builds a production-style credit risk solution using the UCI Default of Credit Card Clients dataset (30,000 rows, 24 features including payment history over 6 months). It demonstrates skills in Python, SQL, feature engineering, modeling (logistic regression + XGBoost), explainability (SHAP), monitoring (PSI), and visualization—directly aligned to TransUnion's Analyst, Data Science and Analytics - Credit Risk role (e.g., proficiency in statistical languages like Python/R, SQL for data pipelines, Tableau/Excel for reporting, and credit risk practices).

## Setup
1. Install dependencies: `conda create -n credit-risk python=3.11; conda activate credit-risk; conda install pandas scikit-learn xgboost shap sqlalchemy matplotlib joblib openpyxl xlrd`
2. Place dataset 'default of credit card clients.xls' in `data/` (download from UCI: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
3. Run scripts in order: `python data_ingestion.py`, `python feature_engineering.py`, `python modeling.py`, `python monitoring.py`

## Files
- `data_ingestion.py`: Loads Excel into SQLite database.
- `feature_engineering.py`: SQL aggregations on payment history + Python domain features (e.g., credit utilization, payment ratio).
- `sql_scripts/bureau_aggregation.sql`: SQL for aggregating monthly payments/bills/delays (simulates credit bureau history).
- `modeling.py`: Builds, tunes, and evaluates models with validation, calibration, and SHAP explainability.
- `monitoring.py`: Computes PSI for model drift monitoring.
- `reports/`: summary.xlsx (advanced Excel with pivots/formulas) and presentation.pptx (executive slides with insights).
- Outputs: features_engineered.csv, predictions.csv, psi_results.csv, shap_summary.png (for dashboard integration).

## Key Techniques Demonstrated
- **Feature Engineering**: Aggregated delays/bills/payments over 6 months; created ratios like utilization (avg_bill / limit_bal) for credit risk insights.
- **Modeling**: Logistic baseline (AUC ~0.65) + tuned XGBoost (grid search for params like learning_rate/depth; final AUC 0.78) with class balancing and isotonic calibration.
- **Validation**: Time-based split (proxy via sorted index) to mimic real-time credit scoring.
- **Explainability**: SHAP (PermutationExplainer) for feature importance—top drivers: Recent delays (PAY_0), bill amounts, utilization.
- **Monitoring**: PSI checks (e.g., <0.1 stable, >0.25 drift)—results show UTILIZATION drift (1.73) warranting retraining.
- **Visualization**: Tableau dashboard + Excel reports for executive communication.

## Results
- Logistic AUC: 0.65 (baseline)
- XGBoost AUC: 0.78 (tuned)
- PSI Drift Metrics:
  | Feature      | PSI Value | Status      |
  |--------------|-----------|-------------|
  | LIMIT_BAL   | 0.26     | Moderate Drift |
  | UTILIZATION | 1.73     | Significant Drift (Retrain Needed) |
  | AGE         | 0.02     | Stable     |
- Key Insights: Default rate rises to 30% at high utilization (>1.0); recent payment delays (PAY_0) are the strongest predictor (importance 0.35).

GitHub: https://github.com/rohitmannur/credit-risk-project  <!-- Replace with your actual repo link -->

Resume Bullet: “Developed end-to-end credit risk pipeline in Python/SQL on UCI Credit Card dataset; engineered payment history features, built tuned XGBoost model (AUC 0.78) with time-based validation, SHAP explainability, and PSI monitoring; delivered Tableau dashboard and Excel/PPT reports for actionable insights.”

## Interactive Tableau Dashboard
Explore the dashboard for visual insights:
- Default Rate vs Utilization: Rate spikes from ~10% (low util) to 30% (high util)—recommend rejecting high-util applicants.
- Top Drivers: PAY_0 (recent delay) dominates at 0.35 importance; focus on recent behaviors.
- Segmentation: Scatter identifies high-risk clusters (high util + low limit bal).

View Dashboard: [https://public.tableau.com/app/profile/rohit.mannur3130/viz/CreditDefaultRiskAnalytics/CreditDefaultRiskAnalyticsDashboard](https://public.tableau.com/app/profile/rohit.mannur3130/viz/CreditDefaultRiskAnalytics/CreditDefaultRiskAnalyticsDashboard)

![Dashboard Screenshot](dashboard_screenshot.png)  <!-- Upload your dashboard image to the repo as dashboard_screenshot.png for this to display -->

## Alignment to TransUnion JD
- **Must-Haves**: Quantitative degree equivalent (stats/math); strong analytical/problem-solving; Python (as R alternative) + SQL/Excel proficiency; communication via dashboard/PPT.
- **Preferred**: Credit risk familiarity (default prediction, segmentation); Tableau visualization; big data scalable code (ready for Spark/Hadoop); global/matrixed env fit via modular design.
- **Impact**: Supports development of predictive solutions for lenders, with governance (PSI) for long-term value.

## Limitations & Next Steps
- Limitations: Small dataset—scale to larger (e.g., Home Credit) for production.
- Next: Add R comparison notebook; deploy as Streamlit app; integrate big data tools (Spark for larger joins).

Thanks for checking out the project, Rohit (@rohit_mannur)! Feedback welcome.