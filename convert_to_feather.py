import pandas as pd
import os

# --- Configuration ---
# Use the absolute path to your data directory
PATH = os.path.expanduser("/home/jonasmensing/bagheera/data/1_funding_period/phase_space_sample/")
CSV_FILENAME = "Nx=9_Ny=9_Ne=8.csv"
FEATHER_FILENAME = "Nx=9_Ny=9_Ne=8.feather"

csv_path = os.path.join(PATH, CSV_FILENAME)
feather_path = os.path.join(PATH, FEATHER_FILENAME)

# --- Conversion ---
print(f"Reading CSV file from: {csv_path}")
try:
    df = pd.read_csv(csv_path)
    print("CSV read successfully. Converting to Feather format...")
    
    # Save the DataFrame in Feather format
    df.to_feather(feather_path)
    
    print(f"\nSuccessfully converted data to Feather format!")
    print(f"Saved new file to: {feather_path}")
    print("\nPlease upload this new .feather file to your scratch directory on the cluster.")
    
except FileNotFoundError:
    print(f"Error: Could not find the CSV file at {csv_path}")
    print("Please make sure the path and filename are correct.")
