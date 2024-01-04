import numpy as np

def uniform_config(low : float, high : float, N_rows : int, N_cols : int)->np.array:

    arr = np.random.uniform(low=low, high=high, size=(N_rows, N_cols))

    return arr

def logic_gate_config(low : float, high : float, off_state : float, on_state : float, i1_col : int, i2_col : int, N_rows : int, N_cols : int)->np.array:

    arr = np.repeat(a=uniform_config(low, high, int(N_rows/4), N_cols), repeats=4, axis=0)
    i1  = np.tile([off_state,off_state,on_state,on_state], int(N_rows/4))
    i2  = np.tile([off_state,on_state,off_state,on_state], int(N_rows/4))

    arr[:,i1_col]   = i1
    arr[:,i2_col]   = i2

    return arr

def logic_gate_config_G(low : float, high : float, N_rows : int)->np.array:
    
    arr = np.repeat(a=uniform_config(low, high, int(N_rows/4), N_cols=1), repeats=4)

    return arr

def ndr_config(low : float, high : float, sweep_low : float, sweep_high : float, sweep_col : int, N_sweep : int, N_rows : int, N_cols : int)->np.array:
    
    arr                 = np.repeat(a=uniform_config(low, high, int(N_rows/N_sweep), N_cols), repeats=N_sweep, axis=0)
    arr[:,sweep_col]    = np.tile(np.linspace(sweep_low,sweep_high,N_sweep),int(N_rows/N_sweep))

    return arr

def ndr_config_G(low : float, high : float, N_sweep : int, N_rows : int)->np.array:

    arr = np.repeat(a=uniform_config(low, high, int(N_rows/N_sweep), N_cols=1), repeats=N_sweep)

    return arr




