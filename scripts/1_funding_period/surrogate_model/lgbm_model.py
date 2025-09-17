import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

PATH    = "/home/jonasmensing/bagheera/data/1_funding_period/phase_space_sample/"
PATH2   = "/mnt/c/Users/jonas/Desktop/phd/nanonets/scripts/1_funding_period/surrogate_model/"

# --- 1. Load Your Data ---
print("Loading data...")
df  = pd.read_csv(f"{PATH}Nx=9_Ny=9_Ne=8.csv")
X   = df.iloc[:,:7].values
y   = df['Observable'].values

# --- 2. Split Data into Full Training and Final Test Sets ---
# We combine the original training and validation sets for the final model.
print("Splitting data...")
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

print(f"Full training set size: {len(X_train_full)}")
print(f"Test set size: {len(X_test)}")

## --- 3. Define the Champion Hyperparameters ---
# These are from Combination 18 of your last run.
best_params = {
    'learning_rate': 0.01,
    'num_leaves': 120,
    'subsample': 0.7,
    'colsample_bytree': 6/7,
    'reg_lambda': 100.0,
    'reg_alpha': 20.0,
    'min_child_samples': 200,
    'objective': 'regression',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# The optimal number of boosting rounds found during tuning
BEST_ITERATION = 45129

# --- 4. Train the Final Model ---
print(f"\nTraining final model with {BEST_ITERATION} rounds...")
final_model = lgb.LGBMRegressor(n_estimators=BEST_ITERATION, **best_params)

# Train on the combined training and validation data
final_model.fit(X_train_full, y_train_full)

print("Training complete.")

# --- 5. Evaluate on the Hold-Out Test Set ---
print("\nEvaluating final model on the test set...")
y_test_pred = final_model.predict(X_test)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print(f"--- Final Test Set Performance ---")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# --- 6. Save the Trained Model ---
model_filepath = os.path.join(PATH2, "final_lgbm_model.txt")
final_model.booster_.save_model(model_filepath)

print(f"\nSuccessfully saved the final model to '{model_filepath}'")
