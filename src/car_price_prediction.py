from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "car_data.csv"
VIZ_DIR = ROOT / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
REFERENCE_YEAR = 2018

# Load and clean

df = pd.read_csv(DATA_PATH)
original_rows = len(df)
duplicate_rows = int(df.duplicated().sum())
df = df.drop_duplicates().copy()
df["Car_Age"] = REFERENCE_YEAR - df["Year"]

# Exploratory visualizations
plt.figure(figsize=(8, 5))
plt.hist(df["Selling_Price"], bins=30)
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price (₹ lakh)")
plt.ylabel("Number of Vehicles")
plt.tight_layout()
plt.savefig(VIZ_DIR / "selling_price_distribution.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["Present_Price"], df["Selling_Price"], alpha=0.7)
plt.title("Selling Price vs Present Price")
plt.xlabel("Present Price (₹ lakh)")
plt.ylabel("Selling Price (₹ lakh)")
plt.tight_layout()
plt.savefig(VIZ_DIR / "selling_vs_present_price.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(df["Car_Age"], df["Selling_Price"], alpha=0.7)
plt.title("Selling Price vs Car Age")
plt.xlabel("Car Age (years)")
plt.ylabel("Selling Price (₹ lakh)")
plt.tight_layout()
plt.savefig(VIZ_DIR / "selling_vs_car_age.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
df.boxplot(column="Selling_Price", by="Fuel_Type")
plt.suptitle("")
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price (₹ lakh)")
plt.tight_layout()
plt.savefig(VIZ_DIR / "price_by_fuel_type.png", dpi=150)
plt.close()

numeric_cols = ["Year", "Selling_Price", "Present_Price", "Driven_kms", "Owner", "Car_Age"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(VIZ_DIR / "correlation_matrix.png", dpi=150)
plt.close()

# Features and preprocessing
X = df.drop(columns="Selling_Price")
y = df["Selling_Price"]
categorical_features = X.select_dtypes(include="object").columns.tolist()
preprocessor = ColumnTransformer(
    [("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)],
    remainder="passthrough",
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    ),
}

# Hold-out evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

holdout_results = []
fitted_pipelines = {}
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    fitted_pipelines[name] = pipe
    holdout_results.append(
        {
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        }
    )

holdout_df = pd.DataFrame(holdout_results).sort_values("MAE")
holdout_df.to_csv(ROOT / "model_results.csv", index=False)

# Five-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2",
        },
        n_jobs=-1,
    )
    cv_rows.append(
        {
            "Model": name,
            "CV_MAE_Mean": -scores["test_MAE"].mean(),
            "CV_MAE_SD": scores["test_MAE"].std(),
            "CV_RMSE_Mean": -scores["test_RMSE"].mean(),
            "CV_RMSE_SD": scores["test_RMSE"].std(),
            "CV_R2_Mean": scores["test_R2"].mean(),
            "CV_R2_SD": scores["test_R2"].std(),
        }
    )

cv_df = pd.DataFrame(cv_rows).sort_values("CV_RMSE_Mean")
cv_df.to_csv(ROOT / "cross_validation_results.csv", index=False)

# Best-model diagnostics
best_model_name = "Gradient Boosting"
best_pipe = fitted_pipelines[best_model_name]
best_pred = best_pipe.predict(X_test)

plt.figure(figsize=(7, 6))
plt.scatter(y_test, best_pred, alpha=0.75)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Actual Selling Price (₹ lakh)")
plt.ylabel("Predicted Selling Price (₹ lakh)")
plt.title("Gradient Boosting: Actual vs Predicted")
plt.tight_layout()
plt.savefig(VIZ_DIR / "actual_vs_predicted.png", dpi=150)
plt.close()

feature_names = best_pipe.named_steps["preprocessor"].get_feature_names_out()
importances = best_pipe.named_steps["model"].feature_importances_
importance_df = (
    pd.DataFrame({"Feature": feature_names, "Importance": importances})
    .sort_values("Importance", ascending=False)
    .head(15)
)
importance_df.to_csv(ROOT / "feature_importance.csv", index=False)

plt.figure(figsize=(9, 6))
plt.barh(importance_df["Feature"].iloc[::-1], importance_df["Importance"].iloc[::-1])
plt.xlabel("Importance")
plt.title("Top 15 Gradient Boosting Features")
plt.tight_layout()
plt.savefig(VIZ_DIR / "gradient_boosting_feature_importance.png", dpi=150)
plt.close()

print(f"Original rows: {original_rows}")
print(f"Duplicate rows removed: {duplicate_rows}")
print(f"Modeling rows: {len(df)}")
print("\nHold-out test results:")
print(holdout_df.round(3).to_string(index=False))
print("\n5-fold cross-validation results:")
print(cv_df.round(3).to_string(index=False))
print(f"\nCross-validation best RMSE model: {cv_df.iloc[0]['Model']}")
