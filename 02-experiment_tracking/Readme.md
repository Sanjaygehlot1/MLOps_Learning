# Duration Prediction — Key Learnings

This notebook builds a **trip duration prediction** workflow on NYC Green Taxi data and focuses on the core ML pipeline, not fancy side quests.

## What was learned

- Loaded taxi trip data from **Parquet** files using pandas.
- Created the target variable **duration (in minutes)** from pickup and dropoff timestamps.
- Cleaned the dataset by keeping trips only in the **1 to 60 minute** range.
- Treated location IDs as **categorical features** and trip distance as a **numerical feature**.
- Converted categorical features into model-ready vectors using **`DictVectorizer`**.
- Trained and evaluated a baseline **Linear Regression** model.
- Improved feature representation by combining pickup and dropoff into a single route feature: **`PU_DO`**.
- Saved the trained preprocessing + model pipeline with **pickle**.
- Used **MLflow** to track:
  - experiment name
  - input data paths
  - model hyperparameters
  - RMSE metric
- Trained a **Lasso** model and logged its performance in MLflow.
- Moved to **XGBoost** for better non-linear modeling.
- Used **Hyperopt** to tune XGBoost hyperparameters automatically.
- Enabled **MLflow autologging** for XGBoost runs to capture params, metrics, and artifacts.

## Main takeaway

The notebook shows the full path from **raw trip data → preprocessing → feature engineering → baseline model → experiment tracking → hyperparameter tuning**.

In one line: this is a clean intro to a practical ML workflow with **pandas, scikit-learn, MLflow, XGBoost, and Hyperopt**.
