# Task 3 — Car Price Prediction with Machine Learning

**CoreAxis Technology Data Science Internship — Task 3**

## Objective
Build regression models to predict used-vehicle selling prices from vehicle characteristics and compare their performance using standard regression metrics.

## Dataset
The CoreAxis-provided used-car dataset contains **301 observations** and **9 original columns**:

`Car_Name, Year, Selling_Price, Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner`

Data quality checks found **2 exact duplicate rows** and **no missing values**. After removing duplicates, **299 observations** were used for analysis and modeling.

## Methodology
1. Load and inspect the dataset.
2. Check missing values and duplicate observations.
3. Remove exact duplicate rows.
4. Create `Car_Age = 2018 - Year`, using the latest year represented in the dataset as the reference year.
5. Perform exploratory data analysis using distributions, scatter plots, group comparisons, and correlations.
6. One-hot encode categorical variables with `handle_unknown='ignore'`.
7. Use an 80/20 train-test split with `random_state=42`.
8. Train Linear Regression, Random Forest, and Gradient Boosting regressors.
9. Evaluate the hold-out test set using MAE, RMSE, and R².
10. Use 5-fold cross-validation to obtain a more robust model comparison.
11. Inspect actual-vs-predicted values and Gradient Boosting feature importance.

## Hold-out Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | **1.192** | 2.825 | 0.690 |
| Random Forest | 1.384 | 3.334 | 0.569 |
| Linear Regression | 1.412 | **2.431** | **0.771** |

On the single test split, Gradient Boosting has the lowest MAE, while Linear Regression has the lowest RMSE and highest R².

## 5-Fold Cross-Validation Results

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| Gradient Boosting | **0.753** | **1.539** | **0.883** |
| Random Forest | 0.764 | 1.646 | 0.857 |
| Linear Regression | 1.172 | 1.908 | 0.838 |

Gradient Boosting achieved the best mean MAE, RMSE, and R² across the five folds. It is therefore the preferred model for this project based on overall validation performance.

## Key EDA Findings
- `Present_Price` has the strongest linear relationship with `Selling_Price` (correlation ≈ **0.876**).
- `Car_Age` has a negative correlation with selling price (≈ **-0.234**).
- `Driven_kms` has a weak direct correlation with selling price (≈ **0.029**).
- Mean selling price is highest for Diesel vehicles in this dataset, followed by Petrol and CNG.
- Dealer-listed vehicles have a higher mean selling price than Individual listings in this dataset.
- Automatic vehicles have a higher mean selling price than Manual vehicles in this dataset.

Group averages describe this dataset's composition and should not be interpreted as causal effects.

## Limitations
- The dataset is relatively small: 299 unique observations after duplicate removal.
- The dataset contains a mixture of cars and motorcycles.
- Validation results are dataset-specific and should not be treated as production performance.
- `Car_Age` is referenced to 2018 because 2018 is the latest year in the supplied dataset; it is not calculated using the current year.

## Tools
Python, Pandas, NumPy, Matplotlib, Scikit-learn, Jupyter Notebook.

## Project Structure
```text
Task-3-Car-Price-Prediction/
├── README.md
├── IMPORTANT_FINDINGS.md
├── data/
│   └── README.md
├── notebooks/
│   └── Car_Price_Prediction.ipynb
├── src/
│   └── car_price_prediction.py
├── visualizations/
│   └── README.md
├── model_results.csv
├── cross_validation_results.csv
└── feature_importance.csv
```

The downloadable final project package contains the generated PNG/SVG visualizations and the original local dataset used for the internship analysis.

## Run the Project
From the repository root:

```bash
python src/car_price_prediction.py
```
