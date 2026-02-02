import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
FOLDER = "/home/j/j_mens07/phd/data/1_funding_period/phase_space_sample/"
FILENAME = "Nx=9_Ny=9_Ne=8.csv"
USE_JITTER = True
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 1e-3
MAX_ERROR = 0.1 #np.inf
X_RANGE = 50*1e-3
Y_RANGE = 500e6


# NETWORK ARCHITECTURE: [Nodes, L2_Reg, Dropout_Rate]
LAYER_INFO = [
    [128, 0.0001, 0.0],
    [128, 0.0001, 0.0],
    [128, 0.0001, 0.0],
    [128, 0.0001, 0.0],
    [128, 0.0001, 0.0],
    [128, 0.0001, 0.0]
]

def load_and_process_data(filepath, max_error=np.inf):
    df = pd.read_csv(filepath)
    df = df[(df['Error']/df['Observable']).abs() < max_error].reset_index(drop=True)
    X = df.iloc[:, :7].values
    y = df['Observable'].values.reshape(-1, 1)
    y_e = df['Error'].values.reshape(-1, 1)/1.96
    return X, y, y_e

def get_model(input_dim, layer_info):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for nodes, reg, drop in layer_info:
        model.add(Dense(nodes, kernel_regularizer=l2(reg)))
        model.add(LeakyReLU(alpha=0.01))
        if drop > 0:
            model.add(Dropout(drop))

    # Output Layer
    model.add(Dense(1, activation='linear'))
    
    return model

class JitterDataGenerator(tf.keras.utils.Sequence):
    """
    Generates data batches with target jittering.
    The y-values are resampled from N(y_mean, y_error) every epoch.
    """
    def __init__(self, X_scaled, y_scaled, y_e_scaled, batch_size=32, shuffle=True):
        self.X = X_scaled
        self.y = y_scaled
        self.y_e = y_e_scaled # This must be the error scaled to the same range as y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices of the batch
        idxs = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Select data
        X_batch = self.X[idxs]
        y_mean_batch = self.y[idxs]
        y_err_batch = self.y_e[idxs]

        # Jitter: Sample from Gaussian N(mean, std)
        # We assume y_err_batch is the standard deviation
        noise = np.random.normal(0, 1, size=y_mean_batch.shape) * y_err_batch
        y_jittered = y_mean_batch + noise

        return X_batch, y_jittered

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def main():
    # 1. Load Data
    print("Loading data...")
    path = os.path.join(FOLDER, FILENAME)
    X, y, y_e = load_and_process_data(path, MAX_ERROR)

    # 2. Split Data
    X_train, X_temp, y_train, y_temp, y_e_train, y_e_temp = train_test_split(
        X, y, y_e, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test, y_e_val, y_e_test = train_test_split(
        X_temp, y_temp, y_e_temp, test_size=0.5, random_state=42
    )

    # 3. Scale Data
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    # x_scaler.fit([[-X_RANGE]*7,[X_RANGE]*7])
    y_scaler.fit([[-Y_RANGE],[Y_RANGE]])

    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.transform(y_train)

    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    X_test_scaled = x_scaler.transform(X_test)
    
    # CRITICAL: We must scale the error bars too!
    # For MinMaxScaler: scaled_std = original_std * scale_factor
    # scale_factor is stored in scaler.scale_
    y_scale_factor = y_scaler.scale_[0] 
    y_e_train_scaled = y_e_train * y_scale_factor

    steps_per_epoch = len(X_train) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS

    # Define the schedule
    lr_schedule = CosineDecay(
        initial_learning_rate=1e-3, 
        decay_steps=total_steps, 
        alpha=0.01  # Minimum LR will be 1% of initial_learning_rate
    )

    # 4. Define Model
    model = get_model(X_train_scaled.shape[1], LAYER_INFO)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')

    # 5. Define Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7)
    ]

    # 6. Train
    print(f"Starting training (Method: {'Target Jittering' if USE_JITTER else 'Sample Weights'})...")
    
    if USE_JITTER:
        # Method A: Data Generator with Jitter
        train_gen = JitterDataGenerator(X_train_scaled, y_train_scaled, y_e_train_scaled, batch_size=BATCH_SIZE)
        history = model.fit(
            train_gen,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=2
        )
    else:
        # Method B: Sample Weights (Inverse Variance)
        epsilon = 1e-6
        # FIX: Square the error to get variance
        variance = (y_e_train.flatten()**2) + epsilon
        weights = 1 / variance
        
        # Clip and Normalize
        weights = np.clip(weights, a_min=0, a_max=10.0)
        weights /= np.mean(weights)
        
        history = model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            sample_weight=weights,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=2
        )

    # 7. Save Artifacts for Jupyter Analysis
    print("Saving model and artifacts...")
    model.save('trained_model.keras') # Save model
    
    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
        
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Save Test set for evaluation in notebook
    np.savez('test_data.npz', X_test=X_test, y_test=y_test, X_test_scaled=X_test_scaled)
    
    print("Done. You can now open the Jupyter Notebook to analyze results.")

if __name__ == "__main__":
    main()