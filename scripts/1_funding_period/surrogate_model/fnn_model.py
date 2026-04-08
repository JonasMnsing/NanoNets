import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
FOLDER          = "/home/j/j_mens07/bagheera/data/1_funding_period/phase_space_sample/"
FILENAME        = "Nx=9_Ny=9_Ne=8.csv"

BATCH_SIZE      = 128
EPOCHS          = 1000
LEARNING_RATE   = 5e-5

# Physical scale for Coulomb Blockade (0.16 aA ≈ 1 electron / second)
# Defines where the math transitions from linear to logarithmic.
SCALE_FACTOR    = 0.16 

# Network Architecture: [Nodes, L2_Regularization, Dropout_Rate]
LAYER_INFO      = [
    [512, 0.0001, 0.0],
    [512, 0.0001, 0.0],
    [256, 0.0001, 0.0],
    [256, 0.0001, 0.0]
]

# ==========================================
# DATA PROCESSING & MODEL ARCHITECTURE
# ==========================================

def load_and_process_data(filepath, scale_factor):
    """
    Loads data from CSV and applies the initial arcsinh transformation.
    Returns both the transformed targets (for fitting scalers) and 
    the raw targets (for dynamic jittering).
    """
    df = pd.read_csv(filepath)

    mask_blocked = df['Error'] == 0.0
    df.loc[mask_blocked, ['Observable', 'Error']] = 0.0

    # 1. Extract raw features and targets
    X = df.iloc[:, :7].values
    y_raw = df['Observable'].values.reshape(-1, 1)
    
    # Convert 95% Confidence Interval to 1-sigma standard error
    y_e_raw = df['Error'].values.reshape(-1, 1)
    
    # 2. Mathematical Transform for 14-Decade Physics
    # Arcsinh acts linearly near zero and logarithmically for large values.
    y_trans = np.arcsinh(y_raw / scale_factor)
    
    return X, y_trans, y_raw, y_e_raw


def get_model(input_dim, layer_info):
    """Dynamically builds the Sequential model based on LAYER_INFO."""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    for nodes, reg, drop in layer_info:
        model.add(Dense(nodes, kernel_regularizer=l2(reg)))
        model.add(Activation('swish'))  # Smooth, non-monotonic activation
        if drop > 0:
            model.add(Dropout(drop))

    # Output Layer (Linear for regression)
    model.add(Dense(1, activation='linear'))
    
    return model


class JitterDataGenerator(tf.keras.utils.Sequence):
    """
    Custom Data Generator that applies physical noise to the target variables
    every epoch to prevent memorization and ensure generalized physics learning.
    """
    def __init__(self, X_scaled, y_raw, y_e_raw, y_scaler, scale_factor, batch_size=32, shuffle=True):
        self.X = X_scaled
        self.y_raw = y_raw          # Raw physical current (aA)
        self.y_e_raw = y_e_raw      # Raw physical error (aA)
        self.y_scaler = y_scaler    # Fitted MinMaxScaler
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        # 1. Get batch indices
        idxs = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # 2. Extract batch data
        X_batch = self.X[idxs]
        y_mean_raw = self.y_raw[idxs]
        y_err_raw = self.y_e_raw[idxs]

        # 3. Inject physical noise in raw domain (Target Jittering)
        noise = np.random.normal(0, 1, size=y_mean_raw.shape) * y_err_raw
        y_jitter_raw = y_mean_raw + noise

        # 4. Transform the newly jittered data via arcsinh
        y_jitter_trans = np.arcsinh(y_jitter_raw / self.scale_factor)

        # 5. Scale to neural network range [-1, 1]
        y_jitter_scaled = self.y_scaler.transform(y_jitter_trans)

        return X_batch, y_jitter_scaled

    def on_epoch_end(self):
        """Shuffle data at the end of every epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

class PeriodicCheckpoint(tf.keras.callbacks.Callback):
    """
    Saves the model and the training history every N epochs.
    """
    def __init__(self, filepath_base, save_every=50):
        super().__init__()
        self.filepath_base = filepath_base
        self.save_every = save_every
        self.custom_history = {} # Manually track history

    def on_epoch_end(self, epoch, logs=None):
        # 1. Append the current epoch's metrics to our history dictionary
        for k, v in logs.items():
            self.custom_history.setdefault(k, []).append(v)
            
        # 2. Check if it is a checkpoint epoch (e.g., 50, 100, 150...)
        if (epoch + 1) % self.save_every == 0:
            # Save the Model
            model_path = f"{self.filepath_base}_epoch_{epoch+1}.keras"
            self.model.save(model_path)
            
            # Save the History
            history_path = f"{self.filepath_base}_history_epoch_{epoch+1}.pkl"
            with open(history_path, 'wb') as f:
                pickle.dump(self.custom_history, f)
                
            print(f"\n[Checkpoint] Saved backup model and history at Epoch {epoch + 1}")

# ==========================================
# MAIN TRAINING LOOP
# ==========================================

def main():
    print("Loading data...")
    path = os.path.join(FOLDER, FILENAME)
    X, y_trans, y_raw, y_e_raw = load_and_process_data(path, SCALE_FACTOR)

    # ---------------------------------------------------------
    # 1. Data Splitting (Synchronous for Raw and Transformed)
    # ---------------------------------------------------------
    # Split 1: Train (70%) vs Temp (30%)
    X_train, X_temp, y_train_trans, y_temp_trans, y_train_raw, y_temp_raw, y_e_train_raw, y_e_temp_raw = train_test_split(
        X, y_trans, y_raw, y_e_raw, test_size=0.3, random_state=42
    )
    
    # Split 2: Validation (15%) vs Test (15%)
    X_val, X_test, y_val_trans, y_test_trans, y_val_raw, y_test_raw, y_e_val_raw, y_e_test_raw = train_test_split(
        X_temp, y_temp_trans, y_temp_raw, y_e_temp_raw, test_size=0.5, random_state=42
    )

    # ---------------------------------------------------------
    # 2. Data Scaling
    # ---------------------------------------------------------
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    # Fit scaling ONLY on the training data
    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train_trans)

    # Apply scaling to Validation and Test sets
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val_trans)
    X_test_scaled = x_scaler.transform(X_test)

    # ---------------------------------------------------------
    # 3. Model Setup & Scheduling
    # ---------------------------------------------------------
    # Calculate steps for Cosine Decay
    steps_per_epoch = len(X_train) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS

    lr_schedule = CosineDecay(
        initial_learning_rate=LEARNING_RATE, 
        decay_steps=total_steps, 
        alpha=0.01  # Ends at 1% of initial learning rate
    )

    model = get_model(X_train_scaled.shape[1], LAYER_INFO)
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='mse')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True),
        PeriodicCheckpoint(filepath_base='trained_model', save_every=50)
    ]

    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    # ---------------------------------------------------------
    # 4. Training (Always uses Target Jittering)
    # ---------------------------------------------------------
    print("Starting training with continuous Target Jittering...")
    
    train_gen = JitterDataGenerator(
        X_scaled=X_train_scaled, 
        y_raw=y_train_raw,       
        y_e_raw=y_e_train_raw,   
        y_scaler=y_scaler,       
        scale_factor=SCALE_FACTOR,
        batch_size=BATCH_SIZE
    )
    
    history = model.fit(
        train_gen,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )

    # ---------------------------------------------------------
    # 5. Save Artifacts
    # ---------------------------------------------------------
    print("Saving model and artifacts...")
    model.save('trained_model.keras') 
            
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    # Save Test set for Jupyter evaluation
    # Note: We save the *transformed* test targets (arcsinh space) 
    # to maintain consistency with the scaler's output format.
    np.savez('test_data.npz', X_test=X_test, y_test=y_test_trans, X_test_scaled=X_test_scaled)
    
    print("Done. You can now open the Jupyter Notebook to analyze results.")

if __name__ == "__main__":
    main()