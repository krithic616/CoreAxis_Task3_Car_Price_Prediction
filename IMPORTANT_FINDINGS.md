# Important Findings — Task 3: Car Price Prediction

## 1. Data Quality
- Original observations: **301**
- Exact duplicate rows: **2**
- Missing values: **None**
- Modeling observations after duplicate removal: **299**
- Original features: **9**
- Engineered feature: **Car_Age**

## 2. Feature Engineering
`Car_Age` was calculated as `2018 - Year` because 2018 is the latest vehicle year represented in the supplied dataset. This keeps the feature consistent with the historical dataset rather than using the current calendar year.

## 3. Exploratory Data Analysis
- `Present_Price` has the strongest linear relationship with `Selling_Price` (**≈ 0.876**).
- `Car_Age` has a negative relationship with selling price (**≈ -0.234**).
- `Driven_kms` has a weak direct correlation with selling price (**≈ 0.029**).
- Mean selling price by fuel type: **Diesel ≈ ₹10.10 lakh, Petrol ≈ ₹3.26 lakh, CNG ≈ ₹3.10 lakh**.
- Mean selling price by selling type: **Dealer ≈ ₹6.63 lakh, Individual ≈ ₹0.87 lakh**.
- Mean selling price by transmission: **Automatic ≈ ₹9.07 lakh, Manual ≈ ₹3.92 lakh**.

These group averages reflect the composition of the supplied dataset and should not be interpreted as causal effects.

## 4. Hold-out Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | **1.192** | 2.825 | 0.690 |
| Random Forest | 1.384 | 3.334 | 0.569 |
| Linear Regression | 1.412 | **2.431** | **0.771** |

## 5. Five-Fold Cross-Validation

| Model | Mean MAE | Mean RMSE | Mean R² |
|---|---:|---:|---:|
| Gradient Boosting | **0.753** | **1.539** | **0.883** |
| Random Forest | 0.764 | 1.646 | 0.857 |
| Linear Regression | 1.172 | 1.908 | 0.838 |

Gradient Boosting performs best across all three mean validation metrics and is therefore the preferred model among the tested approaches.

## 6. Interpretation
The analysis suggests that present price is the strongest linear signal associated with selling price. Vehicle age and categorical characteristics also contribute information. Gradient Boosting can capture nonlinear relationships and interactions, which helps explain its strong cross-validation performance.

## 7. Limitations
- Only 299 unique observations are available after duplicate removal.
- The dataset contains a mixture of cars and motorcycles.
- Results are specific to this dataset and validation setup.
- Cross-validation reduces dependence on a single split but does not replace evaluation on an independent external dataset.

## 8. Conclusion
The project demonstrates a complete machine-learning regression workflow: data quality checks, feature engineering, exploratory analysis, categorical encoding, model training, hold-out evaluation, cross-validation, and model interpretation. Based on five-fold validation, **Gradient Boosting is the strongest overall model among the three tested approaches**.
