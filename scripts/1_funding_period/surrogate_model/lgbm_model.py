import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from itertools import product

PATH = "/scratch/j_mens07/data/1_funding_period/phase_space_sample/"

# --- 1. Load Your Data ---
df  = pd.read_csv(f"{PATH}Nx=9_Ny=9_Ne=8.csv")
X   = df.iloc[:,:7].values
y   = df['Observable'].values
y_e = df['Error'].values

# --- 2. Split Data into Training, Validation, and Test Sets ---
# 70% for training, 15% for validation, 15% for testing.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42  # 0.1765 * 0.85 ≈ 0.15
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

# --- 3. Define the Focused Hyperparameter Grid ---
param_grid = {
    'learning_rate': [0.01],
    'num_leaves': [120],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [5/7, 6/7, 1.0]
}

# --- 4. Manual Grid Search with Early Stopping and Full Logging ---
all_results             = []
N_ESTIMATORS            = 50000
EARLY_STOPPING_PATIENCE = 50

print(f"\nStarting advanced tuning with n_estimators={N_ESTIMATORS} and early stopping patience={EARLY_STOPPING_PATIENCE}...")

param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]

for i, params in enumerate(param_combinations):
    print(f"\n--- Running combination {i+1}/{len(param_combinations)} ---")
    print(params)

    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=N_ESTIMATORS,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_PATIENCE, verbose=True)]
    )

    # --- Evaluate on both training and validation sets ---
    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)
    
    val_score = mean_squared_error(y_val, y_val_pred)
    train_score = mean_squared_error(y_train, y_train_pred)

    print(f"Validation MSE: {val_score:.4f}")
    print(f"Training MSE:   {train_score:.4f}")

    # --- Store all relevant results for this run ---
    run_result = params.copy() # Start with the hyperparameters
    run_result['validation_mse'] = val_score
    run_result['training_mse'] = train_score
    run_result['best_iteration'] = model.best_iteration_
    all_results.append(run_result)

print("\n--- Tuning Complete ---")

# --- 5. Save All Results to a Single CSV ---
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by='validation_mse', ascending=True)

# Add columns for RMSE and the train/validation gap for easier analysis
results_df['validation_rmse'] = np.sqrt(results_df['validation_mse'])
results_df['training_rmse'] = np.sqrt(results_df['training_mse'])
results_df['overfitting_gap_rmse'] = results_df['validation_rmse'] - results_df['training_rmse']

output_filepath = os.path.join(PATH, "lgbm_grid_search_results.csv")
results_df.to_csv(output_filepath, index=False)

print(f"\nSaved all tuning results to '{output_filepath}'")
print("\nTop 5 performing models:")
print(results_df.head(5).to_string())