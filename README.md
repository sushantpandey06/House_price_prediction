# House Price Prediction

Predicts house sale prices using the Ames Housing dataset. 1,460 houses, 79 features.

## What I did

- Handled missing values in 3 categories: "not present" (missing PoolQC = no pool, not unknown pool), zero fills (no garage = 0 garage area), and grouped fills (LotFrontage filled by neighbourhood median)
- Created 8 features. The one that mattered most was QualityScore = OverallQual × GrLivArea. I made it as an experiment — didn't expect much. It ended up with 47.24% feature importance, beating every single original column
- Compared Linear Regression, Ridge, Lasso, Random Forest, and Gradient Boosting. Lasso and GB tied at R²=0.91. Picked Lasso because it also cuts features from 220 to 82
- Random Forest overfit badly: train RMSE 0.05 vs test 0.14

## Results

- R²: 0.91
- RMSE: $20,005
- MAPE: 8.8%
- 79% of predictions within $25K of actual price

Quality 10 houses have a median of $432,390 — that's 8.6x the $50,150 of quality 1. I expected location to be the strongest predictor (the old "location, location, location"), but construction quality beat it.

## Tools

Python, pandas, scikit-learn, matplotlib, seaborn

## How to run

Open `House_price_prediction.ipynb` in Colab or Jupyter. Dataset: Ames Housing from Kaggle.

## What I'd improve

- Hyperparameter tuning for Lasso alpha and GB parameters
- Stack Lasso + GB predictions since they capture different patterns
- Add external data like school ratings and crime rates
