import numpy as np



def ndr_config(low : float, high : float, sweep_low : float, sweep_high : float, sweep_col : int, N_sweep : int, N_rows : int, N_cols : int)->np.array:
    
    arr                 = np.repeat(a=uniform_config(low, high, int(N_rows/N_sweep), N_cols), repeats=N_sweep, axis=0)
    arr[:,sweep_col]    = np.tile(np.linspace(sweep_low,sweep_high,N_sweep),int(N_rows/N_sweep))

    return arr

def ndr_config_G(low : float, high : float, N_sweep : int, N_rows : int)->np.array:

    arr = np.repeat(a=uniform_config(low, high, int(N_rows/N_sweep), N_cols=1), repeats=N_sweep)

    return arr




