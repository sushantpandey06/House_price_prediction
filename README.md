# House Price Prediction using Machine Learning

## Project Overview

This project builds an end-to-end machine learning pipeline to predict residential house prices using the Ames Housing dataset.

The goal was not only to build an accurate regression model, but also to understand how different housing characteristics interact to influence pricing.

During exploratory analysis, one pattern became very clear:

Large houses were not automatically expensive.

The most expensive houses consistently combined:
- large living area
- high construction quality
- strong neighborhood value

That observation later led to the creation of the interaction feature:

```python
QualityScore = OverallQual × GrLivArea
```

This engineered feature eventually became the single strongest predictor in the entire project.

---

# Dataset

Dataset: Ames Housing Dataset

The dataset contains:
- 1,460 training examples
- 81 original features
- numerical and categorical housing attributes

Target variable:
- `SalePrice`

The project uses:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

---

# Project Workflow

The complete workflow included:

1. Data loading and exploration
2. Missing value treatment
3. Exploratory data analysis (EDA)
4. Outlier removal
5. Feature engineering
6. Preprocessing and encoding
7. Model training and comparison
8. Residual analysis
9. Kaggle submission pipeline

---

# Exploratory Data Analysis

Key findings during EDA:

- `OverallQual` showed the strongest correlation with `SalePrice` (0.79)
- `GrLivArea` also had a strong positive relationship with price (0.71)
- SalePrice was heavily right-skewed
- Neighborhood created major pricing differences
- Quality and size repeatedly appeared together among the strongest predictors

One important discovery was that buyers were not simply paying for:
- larger houses
- or better-quality houses separately

Instead, the highest-priced houses consistently combined both.

That insight later motivated the creation of the `QualityScore` interaction feature.

---

# Missing Value Treatment

Missing values were handled differently depending on what they represented in the real world.

Examples:
- missing `PoolQC` usually meant no pool
- missing garage features often meant no garage
- missing basement features often meant no basement

Different imputation strategies were used:
- `"None"` for missing categorical feature absence
- `0` for non-existent numerical features
- neighborhood-wise median imputation for `LotFrontage`

---

# Feature Engineering

Several engineered features were created:

| Feature | Description |
|---|---|
| `TotalSF` | Total house square footage |
| `HouseAge` | Age of house when sold |
| `RemodAge` | Years since remodel |
| `TotalBath` | Combined bathroom utility |
| `HasPool` | Pool indicator |
| `HasGarage` | Garage indicator |
| `QualityScore` | OverallQual × GrLivArea |
| `LogSalePrice` | Log-transformed target |

`QualityScore` became the strongest engineered feature in the project.

---

# Models Tested

The following regression models were compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Models were evaluated using:
- Cross-validation RMSE
- Train RMSE
- Test RMSE
- R² Score

---

# Final Model Selection

Lasso Regression was selected as the final model because it:
- generalized more consistently
- reduced overfitting
- performed automatic feature selection
- remained interpretable
- achieved performance nearly identical to Gradient Boosting

---

# Final Model Performance

| Metric | Result |
|---|---|
| R² | 0.9105 |
| RMSE | $20,005 |
| MAE | $14,368 |
| MAPE | 8.8% |

Prediction accuracy:
- 49% of predictions within $10K
- 79% within $25K
- 97% within $50K

---

# Key Findings

- Construction quality influenced pricing more strongly than expected
- Interaction effects mattered more than isolated variables
- Log transformation significantly improved regression stability
- Random Forest showed clear overfitting
- Lasso generalized more consistently on unseen data
- Feature engineering improved performance more than model complexity alone

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

---


# Future Improvements

Possible future improvements include:
- XGBoost / LightGBM
- hyperparameter optimization
- stacking/blending models
- SHAP explainability
- adding economic and demographic features
- separate models for luxury vs mid-range homes

One interesting finding from the residual analysis was that expensive houses behaved differently enough that they may benefit from specialized modelling approaches.

---

# Conclusion

This project reinforced how important:
- feature engineering
- preprocessing
- and understanding data relationships

can be for structured machine learning problems.

One thing I found especially interesting was how often the data contradicted my initial assumptions.

I originally expected location to dominate pricing behavior much more heavily, but construction quality repeatedly emerged as the strongest driver across almost every stage of the analysis.

That made the project feel much more like discovering how housing characteristics interact rather than simply training regression models.
