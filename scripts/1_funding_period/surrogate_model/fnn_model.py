import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

# --- DEFINE GPU PATHS (WSL2 Fix) ---
nvidia_libs = "/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cusolver/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/nccl/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/nvtx/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cublas/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cufft/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib:/home/jonasmensing/.local/lib/python3.10/site-packages/nvidia/curand/lib"
wsl_lib = "/usr/lib/wsl/lib"
os.environ['LD_LIBRARY_PATH'] = f"{nvidia_libs}:{wsl_lib}:" + os.environ.get('LD_LIBRARY_PATH', '')

# --- Verify GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS: TensorFlow is using {tf.config.experimental.get_device_details(gpus[0])['device_name']}")

from tensorflow.keras.models import Sequential, load_model
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
FOLDER          = "/home/jonasmensing/bagheera/data/1_funding_period/phase_space_sample/"
FILENAME        = "Nx=9_Ny=9_Ne=8.csv"

BATCH_SIZE      = 1024
EPOCHS          = 1000
LEARNING_RATE   = 3e-4
SCALE_FACTOR    = 0.16 

# RESUME SETTINGS
RESUME_TRAINING = False
START_EPOCH     = 400 
CHECKPOINT_PATH = f'trained_model_epoch_{START_EPOCH}.keras'
WEIGHT_TOGGLE_EPOCH = 100 # When to start 10x penalty for blockade errors

# Network Architecture
LAYER_INFO = [[256, 0.0001, 0.0]] * 8

# ==========================================
# DATA PROCESSING
# ==========================================

def load_and_process_data(filepath, scale_factor):
    df = pd.read_csv(filepath)
    mask_blocked = df['Observable'].abs() < 0.16
    df.loc[mask_blocked, ['Observable', 'Error']] = 0.0
    mask_blocked = df['Error'] == 0.0
    df.loc[mask_blocked, ['Observable', 'Error']] = 0.0
    # mask_blocked = df['Observable'] == 0.0
    # df.loc[mask_blocked, ['Observable', 'Error']] = 0.0

    V = df.iloc[:, :7].values 
    
    # 2. Physics-Informed Feature Engineering
    # We calculate the difference between the 'gatekeeper' electrodes
    v4_minus_v6 = (V[:, 4] - V[:, 6]).reshape(-1, 1)
    
    # Interaction term
    v4_v6_inter = (V[:, 4] * V[:, 6]).reshape(-1, 1)
    
    # Combine original features with the new 'hints'
    X = np.hstack([V, v4_minus_v6, v4_v6_inter])

    # X = df.iloc[:, :7].values
    y_raw = df['Observable'].values.reshape(-1, 1)
    y_e_raw = df['Error'].values.reshape(-1, 1)
    y_trans = np.arcsinh(y_raw / scale_factor)
    
    return X, y_trans, y_raw, y_e_raw

def get_model(input_dim, layer_info, lr_schedule):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for nodes, reg, drop in layer_info:
        model.add(Dense(nodes, kernel_regularizer=l2(reg)))
        model.add(Activation('swish'))
        if drop > 0: model.add(Dropout(drop))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='huber')
    return model


# ==========================================
# GENERATOR & CALLBACKS
# ==========================================

class JitterDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_scaled, y_raw, y_e_raw, y_scaler, scale_factor, batch_size=32, shuffle=True, weight_start=100, **kwargs):
        super().__init__(**kwargs)
        self.X = X_scaled
        self.y_raw = y_raw
        self.y_e_raw = y_e_raw
        self.y_scaler = y_scaler
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weight_start = weight_start
        self.current_epoch = 0
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        idxs = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X_batch = self.X[idxs]
        y_mean_raw = self.y_raw[idxs]
        y_err_raw = self.y_e_raw[idxs]

        # Artificial Noise Floor
        safe_errors = np.maximum(y_err_raw, 0.1)
        noise = np.random.normal(0, 1, size=y_mean_raw.shape) * safe_errors
        y_jitter_raw = y_mean_raw + noise
        y_jitter_trans = np.arcsinh(y_jitter_raw / self.scale_factor)
        y_jitter_scaled = self.y_scaler.transform(y_jitter_trans)

        # Weighting Logic
        if self.current_epoch >= self.weight_start:
            # 3x Penalty for blockade errors
            weights = 1.0 + 1.0 * np.exp(-np.abs(y_mean_raw) / 50.0)
        else:
            weights = np.ones_like(y_mean_raw)

        return X_batch, y_jitter_scaled, weights.flatten()

    def on_epoch_end(self):
        self.current_epoch += 1
        if self.shuffle:
            np.random.shuffle(self.indices)

class PeriodicCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath_base, save_every=100):
        super().__init__()
        self.filepath_base = filepath_base
        self.save_every = save_every
        self.custom_history = {}

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.custom_history.setdefault(k, []).append(v)
        if (epoch + 1) % self.save_every == 0:
            self.model.save(f"{self.filepath_base}_epoch_{epoch+1}.keras")
            with open(f"{self.filepath_base}_history_epoch_{epoch+1}.pkl", 'wb') as f:
                pickle.dump(self.custom_history, f)
            print(f"\n[Checkpoint] Saved backup at Epoch {epoch + 1}")

# ==========================================
# MAIN TRAINING LOOP
# ==========================================

def main():
    print("Loading data...")
    path = os.path.join(FOLDER, FILENAME)
    X, y_trans, y_raw, y_e_raw = load_and_process_data(path, SCALE_FACTOR)

    X_train, X_temp, y_train_trans, y_temp_trans, y_train_raw, y_temp_raw, y_e_train_raw, y_e_temp_raw = train_test_split(
        X, y_trans, y_raw, y_e_raw, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val_trans, y_test_trans, y_val_raw, y_test_raw, y_e_val_raw, y_e_test_raw = train_test_split(
        X_temp, y_temp_trans, y_temp_raw, y_e_temp_raw, test_size=0.5, random_state=42
    )

    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))

    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train_trans)
    
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val_trans)
    X_test_scaled = x_scaler.transform(X_test)

    # === SAVE ARTIFACTS NOW (Before Training) ===
    print("Saving scalers and test data for evaluation...")
    with open('scalers.pkl', 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    np.savez('test_data.npz', 
             X_test=X_test, 
             y_test=y_test_trans, 
             X_test_scaled=X_test_scaled)

    # Scheduler Setup
    steps_per_epoch = len(X_train) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    lr_schedule = CosineDecay(initial_learning_rate=LEARNING_RATE, decay_steps=total_steps, alpha=0.1)

    # Model Loading/Building
    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}...")
        model = load_model(CHECKPOINT_PATH)
    else:
        print("Starting training from scratch...")
        model = get_model(X_train_scaled.shape[1], LAYER_INFO, lr_schedule)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True),
        PeriodicCheckpoint(filepath_base='trained_model', save_every=100)
    ]

    train_gen = JitterDataGenerator(
        X_train_scaled, y_train_raw, y_e_train_raw, y_scaler, SCALE_FACTOR, 
        batch_size=BATCH_SIZE, weight_start=WEIGHT_TOGGLE_EPOCH,
        workers=6, use_multiprocessing=True
    )

    if RESUME_TRAINING:
        train_gen.current_epoch = START_EPOCH

    print(f"Training. Weights Active: {train_gen.current_epoch >= WEIGHT_TOGGLE_EPOCH}")
    
    history = model.fit(
        train_gen,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=EPOCHS,
        initial_epoch=START_EPOCH if RESUME_TRAINING else 0, # CRITICAL
        callbacks=callbacks,
        verbose=2
    )

    model.save('trained_model_final.keras')
    print("Done.")

if __name__ == "__main__":
    main()