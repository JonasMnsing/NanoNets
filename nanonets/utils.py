import matplotlib.pyplot as plt
import matplotlib.animation as animation
import colorednoise as cn
from matplotlib import cm
from matplotlib.colors import Normalize

import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing
import logging
import ast

from . import simulation
from typing import Any, Callable, Union, Set, Tuple, List, Dict, Optional
from scipy.stats import entropy
from pathlib import Path
from scipy import signal
from scipy.spatial import KDTree

# Constants
BLUE_COLOR          = '#4477AA'
RED_COLOR           = '#EE6677'
GREEN_COLOR         = '#228833'
NO_CONNECTION       = -100
ELECTRODE_RADIUS    = 10.0

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAMPLING METHODS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def uniform_sample(U_e : Union[float, List[float]], N_samples : int, topology_parameter : dict,
                   U_g : Union[float, List[float]]=0.0, target_electrode : int=-1)->np.ndarray:
    """Returns a uniform sample of electrode voltages, with floating and target electrodes set Zero and Gate electrode defined individually.

    Parameters
    ----------
    U_e : Union[float, List[float]]
        Electrode Voltage range 
    N_samples : int
        Number of Samples
    topology_parameter : dict
        Network topology dictonary
    U_g : Union[float, List[float]], optional
        Gate voltage range, by default 0.0
    target_electrode : int
        Electrode set to be grounded at start

    Returns
    -------
    np.array
        Uniform sample
    """

    # Parameter based on electrode specification
    N_electrodes    = len(topology_parameter["e_pos"])
    electrode_types = np.array(topology_parameter["electrode_type"])
    floating_idx    = np.where(electrode_types=="floating")[0]
    
    # Sample Control Voltages
    if isinstance(U_e, list):
        sample  = np.random.uniform(low=U_e[0], high=U_e[1], size=(N_samples,N_electrodes+1))
    else:
        sample  = np.random.uniform(low=-U_e, high=U_e, size=(N_samples,N_electrodes+1))  
    
    # Sample Gate Voltages
    if isinstance(U_g, list):
        sample[:,-1]    = np.random.uniform(low=U_g[0], high=U_g[1], size=N_samples)
    else:
        sample[:,-1]    = np.random.uniform(low=-U_g, high=U_g, size=N_samples)

    # Set target and floating electrodes to 0V
    sample[:,floating_idx]          = 0.0
    sample[:,target_electrode-1]    = 0.0

    return sample

def lhs_sample(U_e : Union[float, List[float]], N_samples : int, topology_parameter : dict,
               U_g : Union[float, List[float]]=0.0, target_electrode : int=-1)->np.ndarray:
    """Returns a Latin Hypercube sample of electrode voltages, with floating and target electrodes set Zero and Gate electrode defined individually.

    Parameters
    ----------
    U_e : Union[float, List[float]]
        Electrode Voltage range
    N_samples : int
        Number of Samples
    topology_parameter : dict
        Network topology dictonary
    U_g : Union[float, List[float]], optional
        Gate voltage range, by default 0.0
    target_electrode : int
        Electrode set to be grounded at start

    Returns
    -------
    np.array
        LHS sample
    """

    # Parameter based on electrode specification
    N_electrodes    = len(topology_parameter["e_pos"])
    electrode_types = np.array(topology_parameter["electrode_type"])
    floating_idx    = np.where(electrode_types=="floating")[0]

    # lhs sample between [0,1]
    lhs_samples = np.zeros((N_samples, N_electrodes+1))
    for i in range(N_electrodes+1):
        intervals   = np.linspace(0, 1, N_samples + 1)
        points      = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(points)
        lhs_samples[:, i] = points
    scaled_samples  = np.zeros_like(lhs_samples)

    # Sample Control Voltages
    if isinstance(U_e, list):
        scaled_samples[:,:N_electrodes] = U_e[0] + (U_e[1] - U_e[0]) * lhs_samples[:,:N_electrodes]
    else:
        scaled_samples[:,:N_electrodes] = -U_e + (U_e + U_e) * lhs_samples[:,:N_electrodes]
    
    # Sample Gate Voltages
    if isinstance(U_g, list):
        scaled_samples[:,-1]    = U_g[0] + (U_g[1] - U_g[0]) * lhs_samples[:,-1]
    else:
        scaled_samples[:,-1]    = -U_g + (U_g + U_g) * lhs_samples[:,-1]

    # Set floating electrodes to 0V
    scaled_samples[:,floating_idx]          = 0.0
    scaled_samples[:,target_electrode-1]    = 0.0

    return scaled_samples

def logic_gate_sample(U_e : Union[float, List[float]], input_pos : List[int], N_samples : int, topology_parameter : dict,
                      U_i : Union[float, List[float]]=0.01, U_g : Union[float, List[float]]=0.0, sample_technique='lhs', target_electrode : int=-1)->np.array:
    """Returns a sample of electrode voltages, with floating electrodes set Zero and Gate electrodes defined individually.
    At input_pos electrode values are set to Boolean logic states defined in U_i

    Parameters
    ----------
    U_e : Union[float, List[float]]
        Electrode Voltage range
    input_pos : List[int]
        Input Electrode positions
    N_samples : int
        Number of Samples
    topology_parameter : dict
        Network topology dictonary
    U_i : Union[float, List[float]], optional
        Input Voltages, by default 0.01
    U_g : Union[float, List[float]], optional
        Gate voltage range, by default 0.0, by default 0.0
    sample_technique : str, optional
        Sampling technique, either lhs or uniform, by default 'lhs'
    target_electrode : int
        Electrode set to be grounded at start

    Returns
    -------
    np.array
        logic gate sample

    Raises
    ------
    ValueError
        If sample_technique isn't either 'lhs' or 'uniform'
    """

    # Define sampling technique
    if sample_technique == 'lhs':
        sample = lhs_sample(U_e=U_e, N_samples=N_samples, topology_parameter=topology_parameter, U_g=U_g, target_electrode=target_electrode)
    elif sample_technique == 'uniform':
        sample = uniform_sample(U_e=U_e, N_samples=N_samples, topology_parameter=topology_parameter, U_g=U_g, target_electrode=target_electrode)
    else:
        raise ValueError("Sample technique 'lhs' or 'uniform' supported")

    # Repeat Sample 4 times for each logic state
    sample = np.repeat(a=sample, repeats=4, axis=0)

    # Put logic states into sample
    if isinstance(U_i, list):
        sample[:,input_pos[0]]  = np.tile([U_i[0],U_i[0],U_i[1],U_i[1]], N_samples)
        sample[:,input_pos[1]]  = np.tile([U_i[0],U_i[1],U_i[0],U_i[1]], N_samples)
    else:
        sample[:,input_pos[0]]  = np.tile([0.0,0.0,U_i,U_i], N_samples)
        sample[:,input_pos[1]]  = np.tile([0.0,U_i,0.0,U_i], N_samples)

    return sample

def sinusoidal_voltages(N_samples : int, topology_parameter : dict, amplitudes : Union[float, List[float]], frequencies : Union[float, List[float]]=0.0,
                        phase : Union[float, List[float]]=0.0, offset : Union[float, List[float]]=0.0, time_step : float=1e-10)->Tuple[np.array,np.array]:
    """Return voltage array containing sinusoidal signals of given frequencies and amplitudes

    Parameters
    ----------
    N_samples : int
        Number of voltage values
    topology_parameter : dict
        Network topology dictonary
    amplitudes : Union[float, List[float]]
        Single amplitude for all constant electrodes or individual amplitude for each electrode
    frequencies : Union[float, List[float]]
        Single frequency for all constant electrodes or individual frequency for each electrode in Hz, if set to zero voltages are constant
    phase: Union[float, List[float]]
        Single phase for all constant electrodes or individual phase for each electrode
    offset: Union[float, List[float]]
        Single offset for all constant electrodes or individual offset for each electrode
    time_step : float, optional
        Time step size in seconds, by default 1e-10

    Returns
    -------
    Tuple[np.array,np.array]
        Time and Voltages
    """
    # Voltages and Time Scale
    N_electrodes    = len(topology_parameter["electrode_type"])
    voltages        = np.zeros(shape=(N_samples, N_electrodes+1))
    time_steps      = time_step*np.arange(N_samples)
    
    # Parameter based on electrode specification
    electrode_types = np.array(topology_parameter["electrode_type"])
    floating_idx    = np.where(electrode_types=="floating")[0]

    # Signal properties
    frequencies = N_electrodes*[frequencies] if isinstance(frequencies, (int, float)) else frequencies
    amplitudes  = N_electrodes*[amplitudes] if isinstance(amplitudes, (int, float)) else amplitudes
    phase       = N_electrodes*[phase] if isinstance(phase, (int, float)) else phase
    offset      = N_electrodes*[offset] if isinstance(offset, (int, float)) else offset

    # Voltages for each electrode
    for i in range(N_electrodes):
        voltages[:,i]   = amplitudes[i]*np.sin(2*np.pi*frequencies[i]*time_steps+phase[i]) + offset[i]
    
    # Set floating electrodes to 0V
    voltages[:,floating_idx] = 0.0
    
    return time_steps, voltages

def distribute_array_across_processes(process : int, data : np.ndarray, N_processes : int) -> np.ndarray:
    """Returns part of an array to be simulated by a certain process

    Parameters
    ----------
    process : int
        Process index
    data : np.array
        Data as array 
    N_processes : int
        Number of total processes

    Returns
    -------
    np.array
        Sub-array for given process
    """

    # Simulated rows for each process
    index           = [i for i in range(len(data))]
    rows            = [index[i::N_processes] for i in range(N_processes)]
    process_rows    = rows[process]

    return data[process_rows,:]

def logic_gate_time_series(U_e : List[float], input_pos : List[int], U_i : float=0.01,
                           U_g : float=0.0, step_size : float = 1e-9, N_samples : int = 10000)->Tuple[np.array,np.array]:

    voltages    = np.tile(U_e+[U_g],(N_samples,1))
    time_steps  = step_size*np.arange(N_samples)
    U_i1        = [0.0,U_i,0.0,U_i]
    U_i2        = [0.0,0.0,U_i,U_i]

    for i in range(4):
        voltages[i*N_samples//4:((i+1)*N_samples)//4, input_pos[0]] = U_i1[i]
        voltages[i*N_samples//4:((i+1)*N_samples)//4, input_pos[1]] = U_i2[i]
    
    return time_steps, voltages

def generate_band_limited_noise(duration_s: float, max_amplitude: float = 20e-3, bandwidth_hz: float = 4.35e9, dt_s: float = 1e-11) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates time and white noise arrays with a specified sampling rate (dt) and bandwidth.

    Parameters
    ----------
    duration_s: float
        Total length of the simulation time series (seconds).
    bandwidth_hz: float
        The maximum frequency (Hz) to represent in the signal. (Ensures adequate sampling rate for the physical signal).
    dt_s: float, optional
        The discrete time step (seconds). Must be small enough to satisfy Nyquist. Default: 1e-11
    
    Returns:
        A tuple: (time_series, noise_series)
    """

    # 1. Setup Time
    n_samples   = int(duration_s / dt_s)
    time_series = np.linspace(0, duration_s, n_samples, endpoint=False)

    # 2. Generate Raw White Noise (Beta=0)
    raw_noise = cn.powerlaw_psd_gaussian(0, n_samples)

    # 3. Apply Low-Pass Filter (To enforce 4.35 GHz limit)
    # Nyquist frequency
    nyquist = 0.5 / dt_s

    # Design 4th order Butterworth filter
    # normalized_cutoff = cutoff_hz / nyquist
    sos = signal.butter(4, bandwidth_hz / nyquist, btype='low', output='sos')

    # Apply filter
    filtered_noise = signal.sosfiltfilt(sos, raw_noise)
    
    # 4. Normalize (Robust Method)
    # Instead of dividing by random max, we scale so that 3*Sigma = Max Amplitude.
    # This ensures 99.7% of data is within range, and power is constant.
    sigma = np.std(filtered_noise)
    target_sigma = max_amplitude / 3.0
    
    scaled_noise = filtered_noise * (target_sigma / sigma)
    
    # 5. Hard Clip (Safety)
    # Strictly enforce that no outlier exceeds the voltage limit
    final_noise = np.clip(scaled_noise, -max_amplitude, max_amplitude)
    
    return time_series, final_noise

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOAD SIMULATION RESULTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_simulation_results(folder : str, N : Union[int, List[int]], N_e : Union[int, List[int]], disordered : bool=False, data : Union[None, dict]=None)->Tuple[pd.DataFrame,dict]:
    """Load simulation results. For N as list, returns a dict of DataFrames for each particle count. For N_e as list, returns a dict of DataFrames for each electrode count.
    For N and N_e as int returns DataFrame.

    Parameters
    ----------
    folder : str
        Path to data folder
    N : Union[int, List[int]]
        Number of NPs in x,y direction or Number of NPs is disordered Network
    N_e : Union[int, List[int]]
        Number of Electrodes
    disordered : bool, optional
        Disorderd network, by default False
    data : Union[None, dict], optional
        Previous dict of data to be appended if provided, by default None

    Returns
    -------
    Union[pd.DataFrame,dict]
        Simulation data
    """

    # Add slash to folder path
    folder += '/' if not folder.endswith('/') else ''

    # For variable numbers of nanoparticles
    if (isinstance(N, list) and isinstance(N_e, int)):

        # Without dictonary input
        if data is None:
            if disordered:
                data = {key : pd.read_csv(f"{folder}Np={key}_Nj=4_Ne={N_e}.csv") for key in N}
            else:
                data = {key : pd.read_csv(f"{folder}Nx={key}_Ny={key}_Ne={N_e}.csv") for key in N}

        else:
            for key in N:
                if disordered:
                    data[key] = pd.read_csv(f"{folder}Np={key}_Nj=4_Ne={N_e}.csv")
                else:
                    data[key] = pd.read_csv(f"{folder}Nx={key}_Ny={key}_Ne={N_e}.csv")

    # For variable numbers of electrodes
    elif (isinstance(N, int) and isinstance(N_e, list)):

        # Without dictonary input
        if data is None:
            if disordered:
                data = {key : pd.read_csv(f"{folder}Np={N}_Nj=4_Ne={key}.csv") for key in N_e}
            else:
                data = {key : pd.read_csv(f"{folder}Nx={N}_Ny={N}_Ne={key}.csv") for key in N_e}

        else:
            for key in N_e:
                if disordered:
                    data[key] = pd.read_csv(f"{folder}Np={N}_Nj=4_Ne={key}.csv")
                else:
                    data[key] = pd.read_csv(f"{folder}Nx={N}_Ny={N}_Ne={key}.csv")

    else:

        if disordered:
            data  = pd.read_csv(f"{folder}Np={N}_Nj=4_Ne={N_e}.csv")
        else:
            data  = pd.read_csv(f"{folder}Nx={N}_Ny={N}_Ne={N_e}.csv")

    return data

def prepare_for_fitness_calculation(df: pd.DataFrame, N_e: int, input_cols: List[str], drop_zero=True, off_state=0.0, on_state=0.01)->pd.DataFrame:
    """Prepares a pandas DataFrame for the calculation of logic gate fitness, ensuring that all input state combinations are present for each set of rows.

    Parameters
    ----------
    df : pd.DataFrame
        Boolean logic simulation DataFrame
    N_e : int
        Number of Electrodes
    input_cols : List[str]
        List of input voltage columns (e.g., ['E0', 'E1']).
    drop_zero : bool, optional
        Whether to drop rows with zero observable current, by default True
    off_state : float, optional
        Value representing the off-state voltage, by default 0.0
    on_state : float, optional
        Value representing the on-state voltage, by default 0.01

    Returns
    -------
    pd.DataFrame
        The filtered and cleaned DataFrame
    """

    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()

    # Initial filtering based on 'Error' and 'Observable'
    data = data[data['Error'] != 0.0]
    if drop_zero:
        data = data[data['Observable'].abs() > 0.0]

    # 1. Define the columns that should be constant for a group
    # These are all electrode columns that are NOT inputs, plus the 'G' column.
    group_cols = [f'E{i}' for i in range(N_e) if f'E{i}' not in input_cols] + ['G']
    
    # 2. Define the set of required input states we're looking for
    required_states: Set[Tuple[float, float]] = {
        (off_state, off_state),
        (off_state, on_state),
        (on_state, off_state),
        (on_state, on_state),
    }

    # 3. Group by the constant columns and filter the groups
    # A group is kept only if it has exactly 4 rows and contains all required input combinations.
    def is_complete_group(group):
        # Quick check for size
        if len(group) != 4:
            return False
        # Get the actual input states present in the group
        actual_states = set(zip(group[input_cols[0]], group[input_cols[1]]))
        # Compare the set of actual states to the set of required states
        return actual_states == required_states

    # Apply the filter function to the grouped data
    filtered_data = data.groupby(group_cols).filter(is_complete_group)

    # Sort the final result for consistent ordering and reset the index
    sort_cols = group_cols + input_cols
    return filtered_data.sort_values(by=sort_cols).reset_index(drop=True)

def load_boolean_results(folder : str, N : Union[int, List[int]], N_e : Union[int, List[int]], input_cols: List[str],
                         disordered: bool = False, off_state: float = 0.0, on_state: float = None, max_error=np.inf,
                         drop_zero=True) -> Union[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """Load and prepare simulation results.

    Parameters
    ----------
    folder : str
        Path to data folder
    N : Union[int, List[int]]
        Number of NPs in x,y direction or Number of NPs is disordered Network
    N_e : Union[int, List[int]]
        Number of Electrodes
    input_cols : List[str], optional
        List of input voltage columns
    disordered : bool, optional
        Disorderd network, by default False
    off_state : float, optional
        Value representing the off-state voltage, by default 0.0
    on_state : float, optional
        Value representing the on-state voltage, by default 0.01
    max_error : float, optional
        Maximum relative error to be considered, by default np.inf

    Returns
    -------
    Union[pd.DataFrame, dict]
        Prepared simulation data
    """
    
    data = load_simulation_results(folder, N, N_e, disordered)

    if isinstance(data, dict):
        for key, df in data.items():
            data[key] = df[(df['Error']/df['Observable']).abs() < max_error].reset_index(drop=True)
    else:
        data = data[(data['Error']/data['Observable']).abs() < max_error].reset_index(drop=True)

    if isinstance(N, list) and isinstance(N_e, int):
        if isinstance(on_state, float):
            on_state = [on_state for _ in range(len(N))]
        if on_state is None:
            on_state = [data[n_val].loc[:,input_cols[0]].max() for n_val in N]         
        prepared_data = {key: prepare_for_fitness_calculation(data[key], N_e, input_cols, off_state=off_state, on_state=on_state[i], drop_zero=drop_zero) for i, key in enumerate(data.keys())}
    elif isinstance(N, int) and isinstance(N_e, list):
        if isinstance(on_state, float):
            on_state = [on_state for _ in range(len(N_e))]
        if on_state is None:
            on_state = [data[n_val].loc[:,input_cols[0]].max() for n_val in N]
        prepared_data = {key: prepare_for_fitness_calculation(data[key], key, input_cols, off_state=off_state, on_state=on_state[i], drop_zero=drop_zero) for i, key in enumerate(data.keys())}
    else:
        prepared_data = prepare_for_fitness_calculation(data, N_e, input_cols, off_state=off_state, on_state=on_state, drop_zero=drop_zero)

    return prepared_data


def vary_currents_by_error(data: Union[pd.DataFrame, Dict[any, pd.DataFrame]], M: int, current_col: str = 'Observable', error_col: str = 'Error'
) -> Union[pd.DataFrame, Dict[any, pd.DataFrame]]:
    """
    M times vary currents by error to get M separate datasets.
    Handles both single DataFrames and Dictionaries of DataFrames.
    """
    
    # 1. Handle Dictionary Input recursively
    if isinstance(data, dict):
        return {
            key: vary_currents_by_error(df, M, current_col, error_col) 
            for key, df in data.items()
        }

    # 2. Optimized Resampling Logic for a single DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame or a Dictionary of DataFrames.")

    # We use a list to collect dataframes; this is much faster than concat in a loop
    frames = []
    
    # Extract values as numpy arrays once for speed
    currents = data[current_col].values
    # Ensure error is positive and handle NaNs if any
    errors = np.nan_to_num(np.abs(data[error_col].values))
    
    # Drop the error column from the template since it's not needed in the output
    template_df = data.drop(columns=[error_col]).reset_index(drop=True)

    for _ in range(M):
        # Vectorized normal distribution generation
        perturbed_currents = np.random.normal(loc=currents, scale=errors/1.96)
        
        # Create a copy of the template and update the current column
        new_df = template_df.copy()
        new_df[current_col] = perturbed_currents
        frames.append(new_df)

    # Combine everything at once
    return pd.concat(frames, ignore_index=True)

def get_on_off_rss(df00 : pd.DataFrame, df01 : pd.DataFrame, df10 : pd.DataFrame, df11 : pd.DataFrame, gate : str) -> pd.DataFrame:
    """
    Calculate the on and off states and the residual sums of squares (RSS) for the specified logic gate.

    Parameters
    ----------
    df00 : pd.DataFrame
        DataFrame containing input states I1=0, I2=0 (off state).
    df01 : pd.DataFrame
        DataFrame containing input states I1=0, I2=1 (on state).
    df10 : pd.DataFrame
        DataFrame containing input states I1=1, I2=0 (on state).
    df11 : pd.DataFrame
        DataFrame containing input states I1=1, I2=1 (on state).
    gate : str
        Type of logic gate ('AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'P', 'notP', 'Q', 'notQ', 'PnotQ', 'notPQ', 'notPandQ', 'PandnotQ').

    Returns
    -------
    pd.DataFrame
        DataFrame containing off and on state currents, and residual sum of squares (RSS) for each gate.
    """

    df = pd.DataFrame()

    if gate == 'AND':

        df['off']   = (df00['Observable'] + df01['Observable'] + df10['Observable'])/3
        df['on']    = df11['Observable']
        df['res']   = np.sqrt(((df00['Observable'] - df['off'])**2 +
                               (df01['Observable'] - df['off'])**2 +
                               (df10['Observable'] - df['off'])**2 +
                               (df11['Observable'] - df['on'])**2)/4)

    elif gate == 'OR':

        df['off']   = df00['Observable']
        df['on']    = (df01['Observable'] + df10['Observable'] + df11['Observable'])/3
        df['res']   = np.sqrt(((df00['Observable'] - df['off'])**2 +
                               (df01['Observable'] - df['on'])**2 +
                               (df10['Observable'] - df['on'])**2 +
                               (df11['Observable'] - df['on'])**2)/4)

    elif gate == 'XOR':

        df['off']   = (df00['Observable'] + df11['Observable'])/2
        df['on']    = (df01['Observable'] + df10['Observable'])/2
        df['res']   = np.sqrt(((df00['Observable'] - df['off'])**2 +
                               (df01['Observable'] - df['on'])**2 +
                               (df10['Observable'] - df['on'])**2 +
                               (df11['Observable'] - df['off'])**2)/4)

    elif gate == 'NAND':

        df['on']    = (df00['Observable'] + df01['Observable'] + df10['Observable'])/3
        df['off']   = df11['Observable']
        df['res']   = np.sqrt(((df00['Observable'] - df['on'])**2 +
                               (df01['Observable'] - df['on'])**2 +
                               (df10['Observable'] - df['on'])**2 +
                               (df11['Observable'] - df['off'])**2)/4)

    elif gate == 'NOR':

        df['on']    = df00['Observable']
        df['off']   = (df01['Observable'] + df10['Observable'] + df11['Observable'])/3
        df['res']   = np.sqrt(((df00['Observable'] - df['on'])**2 +
                               (df01['Observable'] - df['off'])**2 +
                               (df10['Observable'] - df['off'])**2 +
                               (df11['Observable'] - df['off'])**2)/4)

    elif gate == 'XNOR':

        df['on']    = (df00['Observable'] + df11['Observable'])/2
        df['off']   = (df01['Observable'] + df10['Observable'])/2
        df['res']   = np.sqrt(((df00['Observable'] - df['on'])**2 +
                               (df01['Observable'] - df['off'])**2 +
                               (df10['Observable'] - df['off'])**2 +
                               (df11['Observable'] - df['on'])**2)/4)
    
    elif gate == 'P':

        df['on']    = (df10['Observable'] + df11['Observable'])/2
        df['off']   = (df00['Observable'] + df01['Observable'])/2
        df['res']   = np.sqrt(((df00['Observable'] - df['off'])**2 +
                               (df01['Observable'] - df['off'])**2 +
                               (df10['Observable'] - df['on'])**2 +
                               (df11['Observable'] - df['on'])**2)/4)
    
    elif gate == 'notP':

        df['on']    = (df00['Observable'] + df01['Observable'])/2
        df['off']   = (df10['Observable'] + df11['Observable'])/2
        df['res']   = np.sqrt(((df00['Observable'] - df['on'])**2 +
                               (df01['Observable'] - df['on'])**2 +
                               (df10['Observable'] - df['off'])**2 +
                               (df11['Observable'] - df['off'])**2)/4)

    elif gate == 'Q':

        df['on']    = (df01['Observable'] + df11['Observable'])/2
        df['off']   = (df00['Observable'] + df10['Observable'])/2
        df['res']   = np.sqrt(((df00['Observable'] - df['off'])**2 +
                               (df01['Observable'] - df['on'])**2 +
                               (df10['Observable'] - df['off'])**2 +
                               (df11['Observable'] - df['on'])**2)/4)
    
    elif gate == 'notQ':

        df['on']    = (df00['Observable'] + df10['Observable'])/2
        df['off']   = (df01['Observable'] + df11['Observable'])/2
        df['res']   = np.sqrt(((df00['Observable'] - df['on'])**2 +
                               (df01['Observable'] - df['off'])**2 +
                               (df10['Observable'] - df['on'])**2 +
                               (df11['Observable'] - df['off'])**2)/4)

    elif gate == 'PnotQ':

        df['on']    = df10['Observable']
        df['off']   = (df00['Observable'] + df01['Observable'] + df11['Observable'])/3
        df['res']   = np.sqrt(((df00['Observable'] - df['off'])**2 +
                               (df01['Observable'] - df['off'])**2 +
                               (df10['Observable'] - df['on'])**2 +
                               (df11['Observable'] - df['off'])**2)/4)
    
    elif gate == 'notPQ':

        df['on']    = df01['Observable']
        df['off']   = (df00['Observable'] + df10['Observable'] + df11['Observable'])/3
        df['res']   = np.sqrt(((df00['Observable'] - df['off'])**2 +
                               (df01['Observable'] - df['on'])**2 +
                               (df10['Observable'] - df['off'])**2 +
                               (df11['Observable'] - df['off'])**2)/4)
    
    elif gate == 'notPandQ':

        df['on']    = (df00['Observable'] + df01['Observable'] + df11['Observable'])/3
        df['off']   = df10['Observable']
        df['res']   = np.sqrt(((df00['Observable'] - df['on'])**2 +
                               (df01['Observable'] - df['on'])**2 +
                               (df10['Observable'] - df['off'])**2 +
                               (df11['Observable'] - df['on'])**2)/4)
    
    elif gate == 'PandnotQ':

        df['on']    = (df00['Observable'] + df10['Observable'] + df11['Observable'])/3
        df['off']   = df01['Observable']
        df['res']   = np.sqrt(((df00['Observable'] - df['on'])**2 +
                               (df01['Observable'] - df['off'])**2 +
                               (df10['Observable'] - df['on'])**2 +
                               (df11['Observable'] - df['on'])**2)/4)

    return df

# def fitness(df: pd.DataFrame, input_cols: List[str], delta: float = 0.0, off_state: float = 0.0, on_state: float = None,
#     gates: List[str] = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']) -> pd.DataFrame:
#     """
#     Calculate the fitness of a set of logic gates based on their residual sum of squares (RSS).

#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing electric currents.
#     N_controls : int
#         Number of control signals.
#     input_cols : List[str]
#         List containing the names of the columns representing the two inputs.
#     delta : float, optional
#         A small value to regularize the division in the fitness calculation (default is 0.01).
#     off_state : float, optional
#         The current value considered as the 'off' state (default is 0.0).
#     on_state : float, optional
#         The current value considered as the 'on' state (default is 0.01).
#     gates : List[str], optional
#         List of gate types to calculate the fitness for (default is ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']).

#     Returns
#     -------
#     pd.DataFrame
#         DataFrame containing the fitness values for each gate.
#     """
#     if gates is None:
#         gates = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'P', 'notP', 'Q', 'notQ', 'PnotQ', 'notPQ', 'notPandQ', 'PandnotQ']
#     if on_state is None:
#         on_state = df.loc[:,input_cols[0]].max()
#     df00    = df[(df[input_cols[0]] == off_state) & (df[input_cols[1]] == off_state)].reset_index(drop=True)
#     df01    = df[(df[input_cols[0]] == off_state) & (df[input_cols[1]] == on_state)].reset_index(drop=True)
#     df10    = df[(df[input_cols[0]] == on_state) & (df[input_cols[1]] == off_state)].reset_index(drop=True)
#     df11    = df[(df[input_cols[0]] == on_state) & (df[input_cols[1]] == on_state)].reset_index(drop=True)

#     fitness = pd.DataFrame(0, index=np.arange(len(df00)), columns=list(df00.columns) + [g + ' Fitness' for g in gates])
#     fitness = fitness.drop(columns=['Observable','Error'])

#     for col in list(df00.columns):
#         if col != 'Observable' and col != 'Error':
#             fitness[col]    = (df00[col].values + df01[col].values + df10[col].values + df11[col].values)/4

#     for gate in gates:
#         gate_states                 = get_on_off_rss(df00=df00, df01=df01, df10=df10, df11=df11, gate=gate)
#         gate_states['m']            = gate_states['on'] - gate_states['off']
#         gate_states['denom']        = gate_states['res'] + delta*(gate_states['off'].abs())
#         fitness[gate + ' Fitness']  = gate_states['m']/gate_states['denom']

#     fitness = fitness.reset_index(drop=True)

#     return fitness

def get_displacement_currents(pots:np.ndarray, C_U: np.ndarray, dt: float, output_idx=-1):
    """
    Calculates the total displacement current flowing into the output electrode.
    
    This function computes I_disp(t) = C_out_vec · d(phi_vec)/dt.
    It accounts for the contribution of every nanoparticle in the network, 
    weighted by its mutual capacitance to the output.

    Parameters:
    -----------
    pots : np.ndarray
        Array of shape (steps, N_p) containing the time series of potentials 
        for all N_p nanoparticles.
    C_U : np.ndarray
        The electrode capacitance matrix of shape (N_p, N_e).
    dt : float
        The simulation time step size (in seconds, or consistent units).
    output_idx : int, optional
        The column index in C_U corresponding to the output electrode. 
        Default is -1 (the last column).

    Returns:
    --------
    I_disp : np.ndarray
        Array of shape (steps - 1,). The displacement current time series.
        Note: The length is one less than 'pots' because differentiation 
        consumes one data point.
    """

    # 1. Extract the coupling vector for the output electrode
    # Shape: (N_p,) - This is the vector \vec{C}_{out}
    C_out_vec = C_U[:, output_idx]

    # 2. Calculate the time derivative of the potential vector
    # We use numerical differentiation: d_phi/dt approx (phi(t+1) - phi(t)) / dt
    # Shape: (steps - 1, N_p)
    d_phi_dt = np.gradient(pots, dt, axis=0)

    # 3. Compute the displacement current
    # We perform a dot product sum over the N_p dimension for each time step.
    # Mathematical operation: I(t) = \sum_{i} C_{i,out} * \dot{\phi}_i(t)
    # Using matrix multiplication (@) handles this sum efficiently.
    # (steps-1, N_p) @ (N_p,) -> (steps-1,)
    I_disp = d_phi_dt @ C_out_vec

    return I_disp

def get_SET_TAU_F0():
    topo        = {"Nx" : 1, "Ny" : 1, "electrode_type" : ['constant','constant']}
    sim_c       = simulation.Simulation(topo)
    C_total     = sim_c.get_capacitance_matrix()[0][0]*1e-18
    R_junction  = 25*1e6
    TAU_SET     = R_junction * C_total
    F0_SET      = (1e-6)/(2*np.pi*TAU_SET)
    return TAU_SET, F0_SET

def fitness(df: pd.DataFrame, input_cols: List[str], M: int = 0, delta: float = 0.0, 
            off_state: float = 0.0, on_state: float = None,
            gates: List[str] = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR']) -> pd.DataFrame:
    """
    Calculate fitness and expand the dataset by M resamples per row.
    """
    if gates is None:
        gates = ['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR', 'P', 'notP', 'Q', 'notQ', 'PnotQ', 'notPQ', 'notPandQ', 'PandnotQ']
    
    if on_state is None:
        on_state = df.loc[:, input_cols[0]].max()

    # 1. Split into logic states
    df00 = df[(df[input_cols[0]] == off_state) & (df[input_cols[1]] == off_state)].reset_index(drop=True)
    df01 = df[(df[input_cols[0]] == off_state) & (df[input_cols[1]] == on_state)].reset_index(drop=True)
    df10 = df[(df[input_cols[0]] == on_state) & (df[input_cols[1]] == off_state)].reset_index(drop=True)
    df11 = df[(df[input_cols[0]] == on_state) & (df[input_cols[1]] == on_state)].reset_index(drop=True)

    # 2. Resampling Logic (Expansion)
    if M >= 2:
        def expand_and_resample(target_df, m_samples):
            # Repeat each row index M times (0,0,0, 1,1,1, ...)
            repeated_indices = np.repeat(target_df.index.values, m_samples)
            expanded_df = target_df.iloc[repeated_indices].copy().reset_index(drop=True)
            
            # Apply noise to the 'Observable' column based on the 'Error' column
            noise = np.random.normal(loc=0.0, scale=expanded_df['Error'].values)
            expanded_df['Observable'] += noise
            
            return expanded_df

        df00 = expand_and_resample(df00, M)
        df01 = expand_and_resample(df01, M)
        df10 = expand_and_resample(df10, M)
        df11 = expand_and_resample(df11, M)

    # 3. Initialize Output DataFrame
    # We take metadata columns from df00 (voltages, Resample_idx, etc.)
    exclude_from_output = ['Observable', 'Error']
    fitness_results = df00.drop(columns=[c for c in exclude_from_output if c in df00.columns])

    # 4. Calculate Fitness for each gate
    # Your get_on_off_rss function will receive the expanded dfs (length N*M)
    for gate in gates:
        gate_states = get_on_off_rss(df00=df00, df01=df01, df10=df10, df11=df11, gate=gate)
        
        # Vectorized calculation across all N*M rows
        m = gate_states['on'] - gate_states['off']
        denom = gate_states['res'] + delta * (gate_states['off'].abs()) + 1
        
        fitness_results[gate + ' Fitness'] = m / denom

    return fitness_results.reset_index(drop=True)

def abundance(df: pd.DataFrame, gates: List[str] = ['AND Fitness', 'OR Fitness', 'XOR Fitness', 'NAND Fitness', 'NOR Fitness', 'XNOR Fitness'],
              bins: Union[int, np.ndarray] = 0) -> pd.DataFrame:
    """
    Calculate empirical abundance based on fitness values.

    This function computes the abundance of logic gates based on their fitness values. The abundance
    is derived either through cumulative distribution functions (CDFs) or by sorting and ranking values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fitness values for different logic gates.
    gates : List[str], optional
        List of column names in the DataFrame corresponding to the fitness values of the gates 
        (default is ['AND Fitness', 'OR Fitness', 'XOR Fitness', 'NAND Fitness', 'NOR Fitness', 'XNOR Fitness']).
    bins : Union[int, np.ndarray], optional
        Number of bins to use for histogram-based abundance calculation. If bins is 0, abundance is calculated
        by ranking the sorted fitness values. If an array is provided, it is used as bin edges (default is 0).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the fitness values and their corresponding abundance values for each gate.
    """

    df_abundance = pd.DataFrame()
    
    if isinstance(bins, int):  
        if bins == 0:
            for gate in gates:

                df_curr = df.copy()

                x = np.sort(np.array(df_curr[gate]))
                
                p_x = 1. * np.arange(len(x)) / float(len(x) - 1)
                
                abundance = 100 - 100*p_x
                
                df_abundance[gate]     = x
                df_abundance[gate+' Abundance']   = abundance
        
        else:
            for gate in gates:
            
                df_curr = df.copy()
                count, bins_count = np.histogram(np.array(df_curr[gate]), bins=bins)#, range=range)
                pdf = count / np.sum(count)
                cdf = np.cumsum(pdf)

                abundance = 100 - 100*cdf
                df_abundance[gate]     = bins_count[:-1]
                df_abundance[gate+' Abundance']   = abundance
    else:

        for gate in gates:
            
            df_curr = df.copy()
            count, bins_count = np.histogram(np.array(df_curr[gate]), bins=bins)#, range=range)
            pdf = count / np.sum(count)
            cdf = np.cumsum(pdf)

            abundance = 100 - 100*cdf
            df_abundance[gate]     = bins_count[:-1]
            df_abundance[gate+' Abundance']   = abundance

    return df_abundance

def get_frequency_spectrum(signal, dt):
    """
    Computes the calibrated one-sided amplitude spectrum.
    
    Parameters:
    - signal: 1D numpy array (time domain)
    - dt: Time step in seconds
    
    Returns:
    - freqs: Array of frequencies [Hz]
    - amplitudes: Array of physical amplitudes [A] (or whatever unit signal is in)
    """
    N_samples = len(signal)
    
    # 1. Apply Window (Consistent with your previous analysis)
    window = np.blackman(N_samples)
    y_val = signal - np.mean(signal)
    y_windowed = y_val * window
    
    # Window Coherent Gain Correction
    w_gain = np.sum(window) / N_samples
    
    # 2. Compute FFT
    fft_vals = np.fft.rfft(y_windowed)
    freqs = np.fft.rfftfreq(N_samples, dt)
    
    # 3. Normalize Magnitude
    # Scale for one-sided (*2), normalize by N, correct for window loss
    amplitudes = (np.abs(fft_vals) * 2 / N_samples) / w_gain
    
    return freqs, amplitudes

def extract_harmonic_features(y_val, n_vals, N_periods=20, search_range=3, mode='complex', pad_len=None, interpolate=True):
    """
    Extracts harmonic features from a time series using FFT with optional zero-padding
    and parabolic interpolation for high-precision peak estimation.

    Parameters
    ----------
    y_val : array-like
        Input time series (e.g., current or voltage).
    n_vals : list or array-like of int
        List of harmonic orders to extract (e.g., [1, 2, 3] or [3, 5, 7]).
    N_periods : float, optional (default=20)
        Number of full periods of the fundamental frequency contained in y_val.
        Used to identify the theoretical FFT bin index of the fundamental.
    search_range : int, optional (default=3)
        Number of FFT bins to search left/right of the ideal index to account for 
        spectral leakage or small frequency deviations.
    mode : str, optional (default='complex')
        Format of the output features:
        - 'complex':   [Re, Im] pairs relative to fundamental phase. (Length = 2 * len(n_vals))
                       Preserves full shape info; ideal for Volumetric analysis.
        - 'abs':       [Magnitude] only. (Length = len(n_vals))
                       Ideal for THD or Centroid analysis.
        - 'projected': [Magnitude * cos(delta_phi)]. (Length = len(n_vals))
                       Scalar projection onto fundamental axis (lossy).
        - 'phase':     [Phase Difference] in normalized radians/pi [-1, 1].
    pad_len : int or None, optional (default=None)
        Length of the FFT padding. If None, uses len(y_val). 
        Higher values (e.g., 4096) increase spectral density, reducing discretization error.
    interpolate : bool, optional (default=True)
        If True, uses parabolic interpolation on the spectral peak to estimate 
        magnitude and frequency more accurately than the discrete FFT bin resolution.
        Recommended to avoid "bimodality" in error distributions.

    Returns
    -------
    np.ndarray
        Array of feature values. Structure and length depend on 'mode'.
    """
    
    # --- 1. Preprocessing: Detrend and Window ---
    y_val = np.array(y_val)
    # Remove DC offset to prevent 0-Hz peak leakage
    y_val = y_val - np.mean(y_val)
    
    N_samples = len(y_val)
    
    # Blackman window minimizes spectral leakage (low sidelobes)
    window = np.blackman(N_samples)
    y_windowed = y_val * window
    
    # Coherent Gain Correction: compensates for energy amplitude lost due to windowing
    w_gain = np.sum(window) / N_samples
    
    # --- 2. FFT Calculation ---
    # Determine padding
    if pad_len is None:
        n_fft = N_samples
        pad_ratio = 1.0
    else:
        n_fft = pad_len
        pad_ratio = n_fft / N_samples
    
    # rfft is efficient for real-valued inputs; returns positive freqs only
    fft_vals = np.fft.rfft(y_windowed, n=n_fft)

    # Helper: Peak Extraction Logic
    def get_peak_metrics(neighborhood):
        """
        Internal helper to find peak magnitude and phase from a spectral slice.
        Handles both discrete argmax and parabolic interpolation.
        """
        # Safety check for empty or zero-signal neighborhoods
        if len(neighborhood) == 0 or np.max(np.abs(neighborhood)) < 1e-15:
            return 0.0, 0.0

        # Magnitude spectrum of the neighborhood
        mags = np.abs(neighborhood)
        idx_local = np.argmax(mags)
        
        # 1. Discrete Estimates (Base)
        mag_peak = mags[idx_local]
        # Phase at the discrete peak bin
        phase_peak = np.angle(neighborhood[idx_local])
        
        # 2. Parabolic Interpolation (Refinement)
        if interpolate:
            # We need 3 points: (left, center, right)
            # Check bounds to ensure we have neighbors
            if 0 < idx_local < len(mags) - 1:
                alpha = mags[idx_local - 1]
                beta  = mags[idx_local]
                gamma = mags[idx_local + 1]
                
                # Calculate fractional peak shift 'delta' (-0.5 to 0.5)
                # Formula: Parabolic peak location
                denominator = (alpha - 2 * beta + gamma)
                if denominator != 0:
                    delta = 0.5 * (alpha - gamma) / denominator
                    
                    # Refined Magnitude Estimate
                    mag_peak = beta - 0.25 * (alpha - gamma) * delta
                    
                    # Optional: We stick to the discrete bin phase for robustness, 
                    # but the magnitude is now "physics-accurate".

        return mag_peak, phase_peak

    # --- 3. Find Fundamental Phase (Reference Frame) ---
    k_fund_ideal = int(round(1.0 * N_periods * pad_ratio))
    
    # Define search window (scaled by pad_ratio)
    s_width = int(search_range * pad_ratio)
    f_start = max(0, k_fund_ideal - s_width)
    f_end   = min(len(fft_vals), k_fund_ideal + s_width + 1)
    
    fund_neighborhood = fft_vals[f_start:f_end]
    
    # Get Fundamental Metrics
    amp_fund_raw, phi_1 = get_peak_metrics(fund_neighborhood)
    
    # Note: amp_fund_raw is currently in "FFT units". 
    # We don't convert it to physical units yet because we only need phi_1 here.
    # (Though for consistency, it's good to know amp_fund is available).

    features = []

    # Scaling Factor: Convert FFT amplitude to Physical Amplitude
    #   * 2.0: Accounts for negative frequencies dropped by rfft
    #   * / N_samples: Normalization by original signal length (not padded length)
    #   * / w_gain: Correction for window attenuation
    phys_scale = (2.0 / N_samples) / w_gain

    # --- 4. Extract Harmonics ---
    for m in n_vals:
        # Ideal bin for m-th harmonic
        k_ideal = int(round(m * N_periods * pad_ratio))
        
        # Search window
        start = max(0, k_ideal - s_width)
        end   = min(len(fft_vals), k_ideal + s_width + 1)
        
        # Check if harmonic is within Nyquist limit
        if start < len(fft_vals) and start < end:
            neighborhood = fft_vals[start:end]
            
            # Get Harmonic Metrics
            mag_raw, phi_m = get_peak_metrics(neighborhood)
            
            # Convert to Physical Magnitude
            mag = mag_raw * phys_scale
            
            # Phase relative to fundamental (Shift Invariant)
            delta_phi = phi_m - (m * phi_1)
            
            # --- Feature Formatting ---
            if mode == 'complex':
                # Map to 2D plane: (x, y) coordinates
                features.append(mag * np.cos(delta_phi))
                features.append(mag * np.sin(delta_phi))

            elif mode == 'phase':
                # Normalized Wrapped Phase [-1, 1]
                wrapped = (delta_phi + np.pi) % (2 * np.pi) - np.pi
                features.append(wrapped / np.pi)
                
            elif mode == 'projected':
                # Scalar projection onto Real axis
                features.append(mag * np.cos(delta_phi))
                
            else: # mode == 'abs'
                # Pure Magnitude
                features.append(mag)
                
        else:
            # Handle out-of-bounds (e.g., frequencies > Nyquist)
            if mode == 'complex':
                features.extend([0.0, 0.0])
            else:
                features.append(0.0)
                
    return np.array(features)


def compute_spectral_centroid(amplitudes, harmonic_orders, exclude_fundamental=True):
    """
    Calculates the Spectral Centroid (Center of Mass of the Harmonic Spectrum).
    
    Formula: C = Sum(n * A_n^2) / Sum(A_n^2)
    (Weighted by Power, consistent with energy distribution)
    
    Parameters:
    - amplitudes: Array of magnitudes [A_1, A_3, A_5...]
    - harmonic_orders: Array of harmonic indices [1, 3, 5...]
    - exclude_fundamental: If True, calculates the centroid of the DISTORTION only (n > 1).
                           If False, includes the fundamental frequency.
    
    Returns:
    - centroid: The weighted average harmonic order (e.g., 3.5 means energy is between n=3 and n=5).
    """
    # Ensure inputs are numpy arrays
    amps = np.asarray(amplitudes)
    ords = np.asarray(harmonic_orders)
    
    # 1. Filter: Decide whether to include n=1
    if exclude_fundamental:
        # Only keep harmonics where n > 1
        mask = ords > 1
        valid_amps = amps[mask]
        valid_ords = ords[mask]
    else:
        valid_amps = amps
        valid_ords = ords
        
    # 2. Calculate Power (Square of Amplitude)
    power = valid_amps**2
    total_power = np.sum(power)
    
    # 3. Safety Check for Zero Distortion
    if total_power < 1e-20:
        return np.nan # No harmonic energy exists
        
    # 4. Calculate Centroid
    # Sum(n * Power) / Sum(Power)
    centroid = np.sum(valid_ords * power) / total_power
    
    return centroid

def compute_thd(amplitudes):
    """
    Calculates Total Harmonic Distortion (THD).
    Input: Array of amplitudes [A_fund, A_harm1, A_harm2, ...]
    """
    if len(amplitudes) < 2 or amplitudes[0] == 0:
        return 0.0
        
    power_fund = amplitudes[0]**2
    power_harm = np.sum(amplitudes[1:]**2)
    
    return np.sqrt(power_harm) / np.sqrt(power_fund)

def MC_effective_volume(points, M_samples, fixed_radius, global_bounds):
    """
    Calculates the 'Effective Volume' of a high-dimensional point cloud using 
    Monte Carlo integration with a fixed probe radius.

    This function estimates the volume of the union of hyperspheres centered at each 
    data point. It is useful for quantifying the "reachable state space" or 
    "expressivity" of a reservoir computer in harmonic space.

    Parameters:
    -----------
    points : np.ndarray
        Array of shape (N_points, D_dimensions) containing the coordinates of the 
        reachable states (e.g., normalized harmonic phasors).
    M_samples : int
        Number of Monte Carlo samples to generate. Higher values reduce variance 
        but increase computation time. Suggest > 100,000 for D > 3.
    fixed_radius : float
        The radius of the hypersphere surrounding each point. This defines the 
        "resolution" of the volume.
        - Small radius: Measures point density (Volume ~ N * V_sphere).
        - Large radius: Approaches the Convex Hull volume (ignoring holes).
    global_bounds : tuple of (np.ndarray, np.ndarray)
        A tuple (min_bounds, max_bounds) defining the hyper-rectangle to sample within.
        Ensure these bounds fully enclose the 'points' + 'fixed_radius' to avoid clipping.

    Returns:
    --------
    float
        The estimated effective volume occupied by the point cloud.
    """
    
    # 1. Setup Bounding Box Dimensions
    min_bounds, max_bounds = global_bounds
    
    # Calculate the side lengths of the sampling box
    # (Vector of length D, allowing for non-cubic bounds)
    side_lengths = max_bounds - min_bounds
    
    # Calculate total volume of the sampling box (V_0)
    # This serves as the reference volume for the Monte Carlo integration
    v0_volume = np.prod(side_lengths)

    # 2. Generate Random Samples (Monte Carlo)
    # Create random points uniformly distributed within the [0, 1] hypercube
    raw_samples = np.random.rand(M_samples, points.shape[1])
    
    # Scale and shift samples to fit inside the 'global_bounds' box
    samples = raw_samples * side_lengths + min_bounds

    # 3. Neighbor Search (The "Hit" Test)
    # Build a KDTree for efficient nearest-neighbor lookup
    tree = KDTree(points)
    
    # Query the tree: Find the distance to the single nearest neighbor for each sample
    # k=1 returns only the nearest neighbor distance
    d_to_nn, _ = tree.query(samples, k=1)
    
    # 4. Count Hits
    # A sample is a "hit" if it falls within 'fixed_radius' of ANY point in the set
    hits = np.sum(d_to_nn <= fixed_radius)

    # 5. Calculate Effective Volume
    # V_eff = (Fraction of Hits) * (Total Box Volume)
    p_hit_rate = hits / M_samples
    v_mc_eff = p_hit_rate * v0_volume

    return v_mc_eff

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# MISC
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_best_logic_gate(df: pd.DataFrame, fitness : pd.DataFrame, gate: str, input_cols=['E1','E3'])->pd.DataFrame:
    """Return best performing logic gate

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing Currents/Voltages
    fitness : pd.DataFrame
        Fitness
    gate : str
        Name of the gate
    input_cols : list, optional
        _description_, by default ['E1','E3']

    Returns
    -------
    pd.DataFrame
        Best gate
    """
    
    df          = df.astype(float).round(6)
    df_f        = fitness.astype(float).round(6)
    volt        = df_f.sort_values(by=f'{gate} Fitness', ascending=False).reset_index(drop=True).loc[0,'E0':'E6'].values
    df_gate     = df.copy()

    for i, col in enumerate(['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6']):
        if col not in input_cols:
            df_gate = df_gate[df_gate[col] == volt[i]]

    df_gate = df_gate.reset_index(drop=True)

    return df_gate

def nonlinear_parameter(df: pd.DataFrame, input1_column: str = 'E1', input2_column: str = 'E3',
                        off_state: float=0.0, on_state: float=0.01, n_bootstrap: int=0)->pd.DataFrame:
    """
    Compute nonlinear parameters based on input and output values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing current and input state data.
    input1_column : str, optional
        Column name for the first input state, by default 'E1'.
    input2_column : str, optional
        Column name for the second input state, by default 'E2'.
    off_state : float, optional
        The value representing the 'off' state, by default 0.0.
    on_state : float, optional
        The value representing the 'on' state, by default 0.01.
    n_bootstrap : int, optional
        Add error n_bootstrap times on top of the current and collect nonlinear parameter those in a list, by default 0. 

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated nonlinear parameters:
        - 'Iv': Average current.
        - 'Ml': Difference between inputs 10/11 and 00/01 (left margin).
        - 'Mr': Difference between inputs 01/11 and 00/10 (right margin).
        - 'X': Nonlinear cross-term.
    """

    if on_state is None:
        on_state = df[input1_column].max()

    # Collect values and errors
    values_input = pd.DataFrame()
    errors_input = pd.DataFrame()

    combos = {
        "I00": (off_state, off_state),
        "I01": (off_state, on_state),
        "I10": (on_state, off_state),
        "I11": (on_state, on_state),
    }

    for key, (v1, v2) in combos.items():
        df_tmp = df[
            (df[input1_column] == v1) &
            (df[input2_column] == v2)
        ].reset_index(drop=True)

        values_input[key] = df_tmp["Observable"]
        errors_input[key] = df_tmp["Error"]

    def compute_params(data: pd.DataFrame) -> pd.DataFrame:
        out         = pd.DataFrame()
        out["Iv"]   = (data["I11"] + data["I10"] + data["I01"] + data["I00"]) / 4
        out["Ml"]   = (data["I11"] + data["I10"] - data["I01"] - data["I00"]) / 4
        out["Mr"]   = (data["I11"] - data["I10"] + data["I01"] - data["I00"]) / 4
        out["X"]    = (data["I11"] - data["I10"] - data["I01"] + data["I00"]) / 4
        return out

    # No bootstrap
    if n_bootstrap == 0:
        return compute_params(values_input)

    # Bootstrap
    bootstrap_dfs = []
    sigma = errors_input / 1.96  # assumes 95% CI

    for _ in range(n_bootstrap):
        perturbed = values_input + np.random.normal(0, sigma)
        bootstrap_dfs.append(compute_params(perturbed))

    return bootstrap_dfs

def stat_moment(arr: np.ndarray, order: int = 1, bins: int = 100)->float:
    """
    Compute the statistical moment of an array.

    Parameters
    ----------
    arr : np.array
        Input array containing numerical values.
    order : int, optional
        The order of the moment to compute (default is 1, i.e., the mean).
    bins : int, optional
        Number of bins for histogram-based approximation (default is 100).

    Returns
    -------
    float
        Expected value of the input array raised to the specified order.
    """
    count, bins = np.histogram(a=arr, bins=bins)
    probs       = count / np.sum(count)
    mids        = 0.5*(bins[1:]+ bins[:-1])
    exp_val     = np.sum(probs * mids**order)

    return exp_val

def return_ndr(arr: np.ndarray)->np.ndarray:
    """
    Compute the Negative differential resistance for a given array.

    Parameters
    ----------
    arr : np.array
        Input array of numerical values.

    Returns
    -------
    np.array
        NDR value
    """

    return (1 - np.tanh(np.mean(arr)/np.std(arr)))/2

def return_nls(df: pd.DataFrame, ml_col: str = 'Ml', mr_col: str = 'Mr', x_col: str = 'X', bins: int = 1000)->np.ndarray:
    """
    Compute Nonlinear Seperability (NLS) based on input columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the relevant columns for computation.
    ml_col : str, optional
        Column name representing the left-margin (Ml) values (default is 'Ml').
    mr_col : str, optional
        Column name representing the right-margin (Mr) values (default is 'Mr').
    x_col : str, optional
        Column name representing the cross-term (X) values (default is 'X').
    bins : int, optional
        Number of bins for histogram-based computation of expected values (default is 1000).

    Returns
    -------
    np.array
        NLS value
    """
    return stat_moment(df[x_col].values, order=2, bins=bins)/(stat_moment(df[ml_col].values, order=2, bins=bins) + stat_moment(df[mr_col].values, order=2, bins=bins))

def standard_scale(arr: np.ndarray)->np.ndarray:
    """Standard scale: y = ( y - mean(y) ) / std(y) a Numpy array

    Parameters
    ----------
    arr : np.array
        Numpy array to be scaled

    Returns
    -------
    np.array
        Standard scaled data
    """
    return (arr - np.mean(arr))/np.std(arr)

def min_max_scale(arr: np.ndarray, min_val: float = 0, max_val: float = 1)->np.ndarray:
    """Min-Max scale arr to [min_val,max_val]

    Parameters
    ----------
    arr : np.array
        Numpy array to be scaled
    min_val : float, optional
        Minimum, by default 0
    max_val : float, optional
        Maximum, by default 1

    Returns
    -------
    np.array
        Min Max scaled Array
    """
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))
    arr = arr * (max_val - min_val) + min_val
    return arr

def standard_norm(arr: np.ndarray, min_val: float = 0, max_val: float = 1)->np.ndarray:
    """Standard normalize an array to [min_val,max_val] range

    Parameters
    ----------
    arr : np.array
        Numpy array to be scaled

    Returns
    -------
    np.array
        Scaled Numpy array
    """
    arr = standard_scale(arr)
    arr = min_max_scale(arr, min_val, max_val)
    return arr

def error_norm(arr1: np.ndarray, arr2: np.ndarray, norm:Union[int,str] = 1)->float:

    diff = arr1 - arr2
    if norm == 1:
        return np.linalg.norm(diff, 1)  # L1 norm
    elif norm == 2:
        return np.linalg.norm(diff, 2)  # L2 norm
    elif norm == 'inf':
        return np.linalg.norm(diff, np.inf)  # L∞ norm
    else:
        raise ValueError(f"Unsupported norm type: {norm}. Choose 1, 2, or 'inf'.")

def poincare_map_zero_corssing(arr : np.ndarray)->np.ndarray:
    """For a 1D array, return all idx at which array crosses zero line

    Parameters
    ----------
    arr : np.array
        Data

    Returns
    -------
    np.array
        Zero crossings
    """
    vals        = standard_scale(arr)
    crossing    = np.where(np.diff(np.sign(vals)))[0]
    
    return crossing

def shannon_entropy(state: np.ndarray, bins: int = 20)->np.ndarray:
    """Shannon entropy for a 2D array representing (Time Step, Node Signal)

    Parameters
    ----------
    state : np.array
        2D array where each column represents the signal of a single neuron across time steps.
    bins : int, optional
        Number of bins for histogram, by default 10

    Returns
    -------
    np.array
        Entropy for each node (neuron) in the system.
    """
    entropies   = np.zeros(state.shape[1])

    for i in range(state.shape[1]):
        arr             = state[:,i]
        hist, _         = np.histogram(arr, bins=bins, density=True)
        entropies[i]    = entropy(hist)

    return entropies

def shannon_rank(state : np.ndarray)->float:
    """Calculate the Shannon rank of a 2D array representing (Time Step, Node Signal)

    The Shannon rank is based on the Shannon entropy of the normalized singular values 
    from the Singular Value Decomposition (SVD) of the input matrix. It quantifies 
    the diversity or richness of the system's dynamics.

    Parameters
    ----------
    state : np.array
        2D array where each column represents the signal of a single neuron across time steps.
    bins : int, optional
        Number of bins for histogram, by default 10

    Returns
    -------
    float
        Shannon rank, a measure of the complexity/diversity of the system.
    """

    U, S, Vt    = np.linalg.svd(state)
    p           = S / np.sum(S)
    rank        = np.exp(-np.sum(p*np.log(p)))

    return rank

def autocorrelation(x : np.ndarray, y : np.ndarray, lags : int)->np.ndarray:
    """Compute autocorrelation between two arrays "x" and "y" for a range of lags

    Parameters
    ----------
    x : np.array
        First time series array
    y : np.array
        Second time series array
    lags : int
        Number of lags

    Returns
    -------
    np.array
        Autocorrelation for each lag from 0 to lags-1
    """

    return np.array([np.corrcoef(x, y)[0,1] if l==0 else np.corrcoef(x[:-l], y[l:])[0,1] for l in range(lags)])

def cross_correlation(x: np.ndarray, y: np.ndarray, lags: int) -> np.ndarray:
    """Compute cross-correlation between two arrays "x" and "y" for a range of lags.

    Parameters
    ----------
    x : np.ndarray
        First time series array.
    y : np.ndarray
        Second time series array. Must have the same length as x.
    lags : int
        Number of lags to compute on each side (from -lags to +lags).

    Returns
    -------
    np.ndarray
        Cross-correlation for each lag from -lags to +lags.
    """
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    if lags >= len(x):
        raise ValueError("Lags must be smaller than the length of the arrays.")

    corrs = []
    for l in range(-lags, lags + 1):
        if l == 0:
            # For lag 0, use the full arrays
            c = np.corrcoef(x, y)[0, 1]
        elif l > 0:
            # For positive lags, y is shifted left
            c = np.corrcoef(x[:-l], y[l:])[0, 1]
        else: # l < 0
            # For negative lags, y is shifted right
            # abs(l) is used to get a positive index for slicing
            c = np.corrcoef(x[abs(l):], y[:-abs(l)])[0, 1]
        corrs.append(c)
        
    return np.array(corrs)

def fft(signal: np.ndarray, dt: float,n_padded: int = 0, use_hann: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one‐sided FFT amplitude spectrum of a real-signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    dt : float
        Sampling interval (seconds).
    n_padded : int, optional
        Length to zero-pad signal to before FFT. If 0 or less, no padding.
    use_hann : bool, optional
        If True, apply a Hann window (with coherent gain correction).

    Returns
    -------
    freq : np.ndarray
        Frequencies corresponding to the amplitude bins (Hz).
    mag : np.ndarray
        One-sided amplitude spectrum, such that a pure sine of amplitude A
        appears with magnitude A at its frequency.
    """
    # Ensure array
    x   = np.asarray(signal, dtype=float)
    N0  = signal.size

    # Window
    if use_hann:
        w       = np.hanning(N0)
        x       = x * w
        gain    = np.sum(w) / N0
    else:
        gain    = 1.0

    # Zero-padding (only if n_padded > N0)
    N = max(N0, n_padded)
    if N > N0:
        x = np.pad(x, (0, N-N0), mode='constant')

    # FFT
    fft_vals = np.fft.rfft(x)
    freq     = np.fft.rfftfreq(len(x), dt)

    # Scale to one-sided amplitude and correct for window gain
    mag =   np.abs(fft_vals) * 2 / len(x)
    mag /=  gain

    return freq, mag

def harmonic_strength(signal: np.ndarray, f0: float, dt: float, N_f: int,use_hann: bool = False, n_padded: int = 0, snr_threshold: float = 10.0) -> np.ndarray:
    """
    Compute the relative amplitudes of harmonics using an SNR-based validity check.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    f0 : float
        Fundamental frequency (Hz).
    dt : float
        Sampling interval (seconds).
    N_f : int
        Number of harmonics to include (excludes the fundamental).
    use_hann : bool, optional
        If True, apply a Hann window before FFT.
    n_padded : int, optional
        Length to zero-pad signal to before FFT.
    snr_threshold : float, optional
        Minimum SNR of the fundamental peak relative to the noise floor
        for the calculation to be considered valid.

    Returns
    -------
    h_strength : np.ndarray
        Relative amplitudes of harmonics 2 through N_f+1, normalized to the
        fundamental amplitude: A_n / A_1. Returns zeros if SNR is too low.
    """
    # Compute maximum allowable harmonics
    fs = 1.0 / dt
    if (N_f + 1) * f0 > fs / 2:
        max_harmonics = int(np.floor((fs / 2) / f0)) - 1
        raise ValueError(
            f"N_f={N_f} exceeds maximum detectable harmonics "
            f"({max_harmonics}) for dt={dt} and f0={f0}."
        )

    # Remove DC component
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)

    # Compute spectrum
    freqs, amps = fft(x, dt, n_padded=n_padded, use_hann=use_hann)

    # --- Robustness Improvement ---
    
    # 1. Estimate the noise floor
    # We create a mask to exclude the regions around the fundamental and harmonics
    # from our noise calculation.
    is_peak_region = np.zeros_like(freqs, dtype=bool)
    freq_resolution = freqs[1] - freqs[0]
    # Define a bandwidth around each harmonic to exclude, e.g., 5 bins wide
    exclude_width_bins = 5 
    
    for n in range(1, N_f + 2):
        harmonic_freq = n * f0
        # Find the index of the closest frequency bin
        idx = np.argmin(np.abs(freqs - harmonic_freq))
        # Exclude a small window around this index
        start_idx = max(0, idx - exclude_width_bins // 2)
        end_idx = min(len(freqs), idx + exclude_width_bins // 2 + 1)
        is_peak_region[start_idx:end_idx] = True

    # Calculate noise floor as the median of the spectrum *outside* the peak regions
    noise_floor = np.median(amps[~is_peak_region])
    
    # Avoid division by zero in the unlikely case of a perfectly silent signal
    if noise_floor == 0:
        noise_floor = 1e-12 

    # 2. Find fundamental amplitude and calculate its SNR
    idx1 = np.argmin(np.abs(freqs - f0))
    A1 = amps[idx1]
    snr_f0 = A1 / noise_floor

    # 3. Apply the SNR threshold
    if snr_f0 < snr_threshold:
        # The fundamental is not distinct enough from the noise
        return np.zeros(N_f)

    # --- End of Improvement ---

    # Proceed with calculation as before
    harmonics = np.arange(2, N_f + 2)
    h_strength = np.empty(N_f, dtype=float)

    for i, n in enumerate(harmonics):
        idx = np.argmin(np.abs(freqs - n * f0))
        h_strength[i] = amps[idx] / A1

    return h_strength

def total_harmonic_distortion(signal: np.ndarray, f0: float, dt: float, N_f: int, use_hann: bool = False,
                              n_padded: int = 0, snr_threshold: float = 10, dB: bool = False) -> float:
    """
    Compute the Total Harmonic Distortion (THD) of a signal, based on the
    first N_f harmonics beyond the fundamental.

    THD is given in percent:
        THD = 100 * sqrt(sum_{n=2..N_f+1} (A_n/A1)^2)
    where A1 is the fundamental amplitude and A_n are the harmonic amplitudes.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    f0 : float
        Fundamental frequency (Hz).
    dt : float
        Sampling interval (seconds).
    N_f : int
        Number of harmonics to include (excludes the fundamental).
    use_hann : bool, optional
        If True, apply a Hann window before FFT.
    n_padded : int, optional
        Length to zero-pad signal to before FFT.
    dB : bool, optional
        If True, return THD in decibels: 20 * log10(ratio).
        Otherwise, return THD in percent: 100 * ratio.
    snr_threshold : float, optional
        Minimum SNR of the fundamental peak relative to the noise floor
        for the calculation to be considered valid.

    Returns
    -------
    thd_pct : float
        Total Harmonic Distortion in percent.
    """
    # Get normalized harmonic strengths
    # h       = harmonic_strength(signal, f0, dt, N_f, use_hann=use_hann, n_padded=n_padded)
    h       = harmonic_strength(signal, f0, dt, N_f, use_hann, n_padded, snr_threshold)
    ratio   = np.sqrt(np.sum(h**2))

    # Output
    if dB:
        return 20.0 * np.log10(ratio)
    else:
        return ratio

def harmonic_richness_index(signal: np.ndarray, f0: float, dt: float, N_f: int,
                            threshold: float = 0.01, use_hann: bool = False, n_padded: int = 0) -> float:
    """
    Compute the Harmonic Richness Index (HRI) of a signal driven by a known
    fundamental frequency f0.

    HRI is defined as the fraction of the first N_f harmonics whose relative
    amplitude (A_n/A1) exceeds a given threshold.

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    f0 : float
        Fundamental frequency (Hz).
    dt : float
        Sampling interval (seconds).
    N_f : int
        Number of harmonics to test (excludes the fundamental).
    threshold : float, optional
        Relative amplitude threshold (e.g., 0.01 for -40 dB) above which a
        harmonic is considered significant.
    use_hann : bool, optional
        If True, apply a Hann window before FFT.
    n_padded : int, optional
        Length to zero-pad signal to before FFT.

    Returns
    -------
    hri : float
        Harmonic Richness Index: number_of_significant_harmonics.
    """
    # Get relative strengths
    h = harmonic_strength(signal, f0, dt, N_f,
                          use_hann=use_hann,
                          n_padded=n_padded)
    # Count above threshold
    count = np.sum(h > threshold)
    return float(count)

def spectral_entropy(signal: np.ndarray, dt: float, n_padded: int = 0, use_hann: bool = False) -> float:
    """
    Compute the spectral entropy (Shannon entropy of the normalized PSD).

    Spectral entropy = -sum(p * log2(p)), where p = P / sum(P).

    Parameters
    ----------
    signal : np.ndarray
        Input time-domain signal.
    dt : float
        Sampling interval (seconds).
    n_padded : int, optional
        Length to zero-pad signal to before FFT.
    use_hann : bool, optional
        If True, apply a Hann window before FFT.

    Returns
    -------
    se : float
        Spectral entropy in bits.
    """

    # Compute amplitude spectrum
    freqs, mags = fft(signal - np.mean(signal), dt,
                      n_padded=n_padded, use_hann=use_hann)
    
    # Power spectrum
    P = mags**2

    # Normalize to probability distribution
    P_sum = np.sum(P)
    if P_sum <= 0:
        return 0.0
    p = P / P_sum

    # Avoid log2(0)
    p = np.where(p <= 0, 0, p)

    # Compute entropy
    se = -np.sum(p * np.log2(p))

    return se / np.log2(len(freqs))

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# SIMULATION HELPER FUNCTIONS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
def run_static_simulation(voltages: np.ndarray, topology: Dict[str, Any], out_folder: Union[Path, str],
                          net_kwargs: Optional[Dict[str,Any]] = None, sim_kwargs: Optional[Dict[str,Any]] = None)->None:
    """Instantiate and run one nanonets static simulation.

    Parameters
    ----------
    voltages : np.ndarray
        2D array of shape (n_volt, n_electrodes) of applied voltages.
    topology : dict
        Network topology
    out_folder : pathlib.Path or str
        Directory where simulation outputs are saved.
    net_kwargs : dict, optional
        Additional keyword arguments for nanonets.simulation(). Defaults to None
    sim_kwargs : dict, optional
        Additional keyword arguments for nanonets.run_static_voltages(). Defaults to None.

    Returns
    -------
    None
    """
    net_kwargs = net_kwargs or {}
    sim_kwargs = sim_kwargs or {}
    try:
        # target = len(topology["e_pos"]) - 1
        target = voltages.shape[1] - 2
        sim = simulation.Simulation(
            topology_parameter=topology,
            folder=str(out_folder)+"/",
            **net_kwargs,
        )
        sim.run_static_voltages(
            voltages=voltages,
            target_electrode=target,
            **sim_kwargs,
        )
        logging.info(f"Done {topology}")
    except Exception:
        logging.exception(f"Error for topology {topology}")

def run_dynamic_simulation(time_steps: np.ndarray, voltages: np.ndarray, topology: Dict[str, Any], out_folder: Union[Path, str],
                           net_kwargs: Optional[Dict[str,Any]] = None, sim_kwargs: Optional[Dict[str,Any]] = None)->None:
    """Instantiate and run one nanonets simulation.

    Parameters
    ----------
    time_steps : np.ndarray
        1D array of time points (in seconds) for simulation steps.
    voltages : np.ndarray
        2D array of shape (n_steps, n_electrodes) of applied voltages.
    topology : dict
        Network topology with keys 'Nx', 'Ny', 'Nz', 'e_pos', and 'electrode_type'.
    out_folder : pathlib.Path or str
        Directory where simulation outputs are saved.
    net_kwargs : dict, optional
        Additional keyword arguments for nanonets.simulation(). Defaults to None
    sim_kwargs : dict, optional
        Additional keyword arguments for nanonets.run_static_voltages(). Defaults to None.

    Returns
    -------
    None
    """
    sim_kwargs = sim_kwargs or {}
    net_kwargs = net_kwargs or {}

    try:
        target = voltages.shape[1] - 2
        sim = simulation.Simulation(
            topology_parameter=topology,
            folder=str(out_folder)+"/",
            **net_kwargs,
        )
        sim.run_dynamic_voltages(
            voltages=voltages,
            time_steps=time_steps,
            target_electrode=target,
            **sim_kwargs,
        )
        logging.info(f"Done {topology}")
    except Exception:
        logging.exception(f"Error for topology {topology}")

def batch_launch(func: Callable[..., Any], tasks: List[Tuple[Tuple[Any,...]]], max_procs: int)->None:
    """
    Launch tasks in parallel, limiting concurrency to max_procs.

    Parameters
    ----------
    func : callable
        Function to run in each process. Must accept args and kwargs as given.
    tasks : list of (args, kwargs)
        Each task is a tuple: (args, kwargs). `args` is a tuple of positional
        arguments for func; `kwargs` is a dict of keyword arguments.
    max_procs : int
        Maximum number of concurrent processes.

    Returns
    -------
    None
    """
    running = []
    for args, kwargs in tasks:
        while len(running) >= max_procs:
            running.pop(0).join()
        p = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        p.start()
        running.append(p)
    for p in running:
        p.join()

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Network Currents
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def get_net_currents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduces a DataFrame by subtracting symmetric column pairs.

    This function identifies column pairs named '(u, v)' and '(v, u)',
    calculates the difference (u,v) - (v,u), and creates a new
    DataFrame with the results.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame where column names are strings of the format
        '(item1, item2)'. It is expected that for each '(u, v)', a
        corresponding '(v, u)' column exists.

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the subtracted values. The new columns
        are named 'u_minus_v' to reflect the operation. The resulting
        DataFrame will have half the number of columns of the original.
    """
    # A set to keep track of columns we've already included in a calculation
    processed_columns = set()
    
    # A dictionary to build the new DataFrame from
    new_df_data = {}

    # Iterate over each column name in the original DataFrame
    for col_name in df.columns:
        # If we have already processed this column as a counterpart, skip it
        if col_name in processed_columns:
            continue

        # --- Parse the column name to find its counterpart ---
        try:
            # 1. Remove the parentheses: '(u, v)' -> 'u, v'
            # 2. Split by the comma and space: 'u, v' -> ['u', 'v']
            u_str, v_str = col_name.strip("()").split(', ')
        except ValueError:
            # This handles cases where a column name doesn't fit the format
            print(f"Warning: Skipping column '{col_name}' as it doesn't match the '(u, v)' format.")
            continue

        # Construct the name of the counterpart column
        counterpart_col_name = f'({v_str}, {u_str})'

        # --- Check for the counterpart and perform subtraction ---
        if counterpart_col_name in df.columns:
            # Define a clean name for the new column
            new_col_name = f'({u_str}, {v_str})'
            
            # Perform the subtraction and store it in our dictionary
            new_df_data[new_col_name] = df[col_name] - df[counterpart_col_name]

            # Add both the original and counterpart columns to our set of
            # processed columns so we don't calculate the inverse.
            processed_columns.add(col_name)
            processed_columns.add(counterpart_col_name)
        else:
            # This handles cases where a column doesn't have a symmetric pair
            print(f"Warning: Counterpart for '{col_name}' not found. Skipping.")

    # Create the new DataFrame from our dictionary of results
    return pd.DataFrame(new_df_data)

def create_weighted_undirected_graph(data):
    
    G = nx.Graph()
    for key_str, weight in data.items():
        u, v = ast.literal_eval(key_str)
        G.add_edge(u, v, weight=weight)

    return G

def display_net_flow_graph(net_graph: nx.DiGraph, ax=None, pos=None, node_color='#348ABD',
                           node_size=40, font_size=None, cmap=plt.cm.Reds, vmin=None, vmax=None, log_scale=False):
    
    # --- 1. Setup Figure and Axes ---
    if ax is None:
        fig, ax = plt.subplots(dpi=200)
    
    # --- 2. Calculate Node Positions ---
    if pos is None:
        pos = nx.kamada_kawai_layout(net_graph)

    # --- 3. Get Edge Weights for Coloring ---
    weights = [net_graph[u][v]['weight'] for u, v in net_graph.edges()]
    # Use log scale for better color distribution with wide-ranging data
    if log_scale:
        new_weights = np.log1p(np.array(weights))
        norm_min    = np.log1p(vmin) if vmin is not None else min(new_weights)
        norm_max    = np.log1p(vmax) if vmax is not None else max(new_weights)
    else:
        new_weights = np.array(weights)
        norm_min    = vmin if vmin is not None else min(new_weights)
        norm_max    = vmax if vmax is not None else max(new_weights)

    # --- 4. Draw the Graph Components ---
    nx.draw(net_graph, pos, node_color=node_color, node_size=node_size, ax=ax,
            edge_color=new_weights, edge_cmap=cmap, width=1,
            edge_vmin=norm_min, edge_vmax=norm_max)
    
    if font_size is not None:
        nx.draw_networkx_labels(net_graph, pos, font_size=font_size, ax=ax)
        
    ax.axis('off')
    
    return ax

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLOTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def abundance_plot(df: pd.DataFrame, gates: List[str] = ['AND', 'OR', 'XOR', 'XNOR', 'NAND', 'NOR'], 
    dpi: int = 200, x_limits: List[float] = [0.45, 10], y_limits: List[float] = [1.0, 100],
    xlabel: str = 'Fitness', ylabel: str = 'Abundance', style_sheet: List[str] = ["science","bright","grid"]) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the abundance of logic gates as a function of their fitness values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fitness and abundance values for logic gates.
    gates : List[str], optional
        List of logic gate names to include in the plot (default is ['AND', 'OR', 'XOR', 'XNOR', 'NAND', 'NOR']).
    dpi : int, optional
        Resolution of the plot in dots per inch (default is 200).
    x_limits : List[float], optional
        List specifying the x-axis limits [min, max] (default is [0.1, 10]).
    y_limits : List[float], optional
        List specifying the y-axis limits [min, max] (default is [0.1, 100]).
    xlabel : str, optional
        Label for the x-axis (default is 'Fitness').
    ylabel : str, optional
        Label for the y-axis (default is 'Abundance').
    style_sheet : List[str], optional
        List of style constraints

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects for the plot.
    """

    marker = ['o','s','^','<','>','v','P']

    with plt.style.context(style_sheet):
        fig, ax = plt.subplots(dpi=dpi)
        for i, gate in enumerate(gates):
            ax.plot(df[f'{gate} Fitness'], df[f'{gate} Fitness Abundance'], label=f'{gate}')
            # ax.plot(df[f'{gate} Fitness'], df[f'{gate} Fitness Abundance'], marker=marker[i % len(marker)], markevery=0.1, label=f'{gate}')

        ax.set_xlim(x_limits[0], x_limits[1])
        ax.set_ylim(y_limits[0], y_limits[1])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize='x-small', loc='upper right')

    return fig, ax

def display_network(G, pos: dict, radius: np.ndarray, fig: plt.Figure, ax: plt.Axes):
    """
    Visualize a nanoparticle network, including particles and electrodes.

    Parameters
    ----------
    G : networkx.Graph
        The network graph.
    pos : dict
        Node positions {index: (x, y)}.
    radius : np.ndarray
        Radii for each nanoparticle.
    fig, ax : matplotlib Figure and Axes
        Output axes for plotting.

    Returns
    -------
    fig, ax : tuple
        The matplotlib Figure and Axes with the plot.
    """
    ax.set_aspect('equal')

    # Draw network edges
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ax.plot([x0,x1],[y0,y1], 'black', lw=1)

    # Draw nanoparticles
    for i in range(len(radius)):
        x, y = pos[i]
        circle = plt.Circle((x, y), radius[i], fill=True,
                            edgecolor='black', lw=1, zorder=2, facecolor=BLUE_COLOR)
        ax.add_patch(circle)

    # Draw electrodes
    N_e = len(G.nodes) - len(radius)
    for i in range(1,N_e+1):
        e_node = -int(i)
        x, y = pos[-i]
        # Draw electrode circle
        circ = plt.Circle((x, y), ELECTRODE_RADIUS, fill=True,
                        edgecolor='black', lw=1, zorder=2, facecolor=RED_COLOR)
        ax.add_patch(circ)

    # Autoscale and padding
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = max(np.max(radius), ELECTRODE_RADIUS) + 1
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    
    return fig, ax

def update_circle_colors(ax: plt.Axes, pot: np.ndarray, vlim=None, cmap="coolwarm", colorbar=False):
    """
    Update nanoparticle circles colored by potential.
    
    Parameters
    ----------
    ax : matplotlib Axes
    pot : np.ndarray
        Potentials for each node.
    vlim : float, optional
        Color scale limit. If None, +- max|pot| used.
    cmap : str
        Matplotlib colormap name.
    """
    vlim = np.max(np.abs(pot)) if vlim is None else vlim
    norm = Normalize(-vlim,vlim)
    cmap = cm.get_cmap(cmap)
    n    = len(pot)
    for i in range(n):
        ax.patches[i].set_facecolor(cmap(norm(pot[i])))
    
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='$\phi$ [mV]')

def display_network_currents(df : pd.DataFrame, row, N_electrodes : int, charge_landscape=None, pos=None, fig=None, ax=None,
                             arrowsize=12, node_size=300, blue_color='#348ABD', red_color='#A60628',
                             position_by_currents=False, display_path=None, edge_vmin=None, edge_vmax=None):
    
    # Initialize figure and axis if not provided
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()
        
    ax.axis('off')

    # Extract values based on row specification
    if isinstance(row, tuple):
        values  = df.loc[row[0]:row[1],:].mean().values
    else:
        values  = df.loc[row,:].values

    junctions                   = np.array([val for val in df.columns])
    values_new, junctions_new   = [], []

    # Process the currents and prepare for plotting
    for n1, junction in enumerate(junctions):

        i       = junction[0]
        j       = junction[1]
        val1    = values[n1]
        n2      = np.where(((junctions[:,0]==j) & (junctions[:,1]==i)))[0][0]

        if n2 > n1:
            
            val2  = values[n2]
            values_new.append(np.abs(val2-val1))
            
            if val1 > val2:
                junctions_new.append([i-N_electrodes,j-N_electrodes])
            else:
                junctions_new.append([j-N_electrodes,i-N_electrodes])

    values_new  = np.log1p(values_new)
    values_new  = (values_new - np.min(values_new))/(np.max(values_new) - np.min(values_new))

    if edge_vmin is None:
        edge_vmin = np.min(values_new)
    
    if edge_vmax is None:
        edge_vmax = np.max(values_new)

    G = nx.DiGraph()
    G.add_nodes_from(np.arange(np.min(junctions)-N_electrodes, np.max(junctions)+1-N_electrodes))

    if charge_landscape is not None:

        if isinstance(row, tuple):
            states  = charge_landscape.loc[row[0]:row[1],:].mean().values
        else:
            states  = charge_landscape.loc[row,:].values
        colors  = np.repeat(BLUE_COLOR, len(G.nodes)-N_electrodes)
        colors[np.where(states < 0)] = RED_COLOR
        colors  = np.insert(colors, 0, np.repeat(BLUE_COLOR, N_electrodes))
        states  = np.abs(states)
        states  = node_size*(states - np.min(states))/(np.max(states)-np.min(states))
        states  = np.insert(states, 0, np.repeat(node_size, N_electrodes))
    else:
        states  = np.repeat(node_size, len(G.nodes))
        colors  = np.repeat(BLUE_COLOR, len(G.nodes))

    for val, junction in zip(values_new, junctions_new):

        G.add_edge(junction[0], junction[1], width=val)

    widths = [G[u][v]['width'] for u, v in G.edges]

    if pos is None:
        if position_by_currents:
            pos = nx.kamada_kawai_layout(G=G, weight='width', seed=42)
        else:
            pos = nx.kamada_kawai_layout(G=G, seed=42)
    else:
        pos         = pos.copy()
        keys        = [-i for i in range(1, N_electrodes+1)]
        key_vals    = [pos[i] for i in keys]
        new_keys    = keys[::-1]

        for key in keys:
            pos.pop(key)
        
        for i, key in enumerate(new_keys):
            pos[key] = key_vals[i]

    nx.draw(G=G, pos=pos, ax=ax, edge_color=widths, arrowsize=arrowsize, node_size=states,
            edge_cmap=plt.cm.Reds, node_color=colors, edge_vmin=edge_vmin, edge_vmax=edge_vmax)

    return fig, ax

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# PLOTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def store_average_time_results(folder, Nx, Ny, Nz, Ne, N_stat, N_threads):
    
    values          = [pd.read_csv(folder+f"/Nx={Nx}_Ny={Ny}_Nz={Nz}_Ne={Ne}_t{j}_s{k}.csv") for j in range(N_threads) for k in range(N_stat)]
    means           = pd.DataFrame(np.mean(values, axis=0),columns=values[0].columns)
    means['Error']  = np.std(values,axis=0)[:,-2]/np.sqrt(len(values))

    means.to_csv(folder+f"/Nx={Nx}_Ny={Ny}_Nz={Nz}_Ne={Ne}.csv", index=0)

def store_average_time_states(folder, Nx, Ny, Nz, Ne, N_stat, N_threads):

    values  = [pd.read_csv(folder+f"/mean_state_Nx={Nx}_Ny={Ny}_Nz={Nz}_Ne={Ne}_t{j}_s{k}.csv") for j in range(N_threads) for k in range(N_stat)]
    means   = pd.DataFrame(np.mean(values, axis=0),columns=values[0].columns).round(3)
    
    means.to_csv(folder+f"/mean_state_Nx={Nx}_Ny={Ny}_Nz={Nz}_Ne={Ne}.csv", index=0)

def store_average_time_currents(folder, Nx, Ny, Nz, Ne, N_stat, N_threads):

    values  = [pd.read_csv(folder+f"/net_currents_Nx={Nx}_Ny={Ny}_Nz={Nz}_Ne={Ne}_t{j}_s{k}.csv") for j in range(N_threads) for k in range(N_stat)]
    means   = pd.DataFrame(np.mean(values, axis=0),columns=values[0].columns).round(3)
    
    means.to_csv(folder+f"/net_currents_Nx={Nx}_Ny={Ny}_Nz={Nz}_Ne={Ne}.csv", index=0)


# def display_network(np_network_sim : simulation.Simulation, fig=None, ax=None, blue_color='#348ABD', red_color='#A60628', save_to_path=False, 
#                     style_context=['science','bright'], node_size=300, edge_width=1.0, font_size=12, title='', title_size='small',
#                     arrows=False, provide_electrode_labels=None, np_numbers=False, height_scale=1, width_scale=1, margins=None):

#     n_floatings = len(np_network_sim.floating_indices)
#     colors      = np.repeat(blue_color, np_network_sim.N_particles+np_network_sim.N_electrodes)
    
#     if np_numbers:
#         node_labels = {i : i for i in np_network_sim.G.nodes}
#     else:
#         node_labels = {i : '' for i in np_network_sim.G.nodes}

#     if provide_electrode_labels == None:
#         if n_floatings == 0:
#             colors[np_network_sim.N_particles:] = red_color
#         else: 
#             colors[np_network_sim.N_particles-n_floatings:-n_floatings] = red_color
#     else:
#         colors[np_network_sim.N_particles-n_floatings:-n_floatings] = None
    
#         for i, electrode_label in enumerate(provide_electrode_labels):
#             node_labels[-1-i] = electrode_label

#     with plt.style.context(style_context):

#         if fig == None:
#             fig = plt.figure()
#             fig.set_figheight(fig.get_figheight()*height_scale)
#             fig.set_figwidth(fig.get_figwidth()*width_scale)
#         if ax == None:
#             ax = fig.add_subplot()

#         ax.axis('off')
#         ax.set_title(title, size=title_size)

#         nx.draw_networkx(G=np_network_sim.G, pos=np_network_sim.pos, ax=ax, node_color=colors, arrows=arrows,
#                          node_size=node_size, font_size=font_size, width=edge_width, labels=node_labels,
#                          clip_on=False, margins=margins, bbox={"color":"white"})

#         if save_to_path != False:

#             fig.savefig(save_to_path, bbox_inches='tight', transparent=True)

#     return fig, ax

def display_landscape(path : str, row, Nx, Ny, fig=None, ax=None, cmap='coolwarm', vmin=None, vmax=None,
                        x_label='$x_{NP}$', y_label='$y_{NP}$', colorbar=False, interpolation=None, cbar_label=''):

    if type(path) == str:
        arr = pd.read_csv(path).loc[row,:].values
    else:
        arr = path.loc[row,:].values
    
    arr = arr.reshape(Nx, Ny)
    
    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot()
    
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', interpolation=interpolation)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if colorbar:
        fig.colorbar(im, label=cbar_label)

    return fig, ax

def animate_landscape(landscape : np.array, Nx, Ny, N_rows=None, fig=None, ax=None, cmap='coolwarm', vmin=None, vmax=None,
                        x_label='$x_{NP}$', y_label='$y_{NP}$', interpolation=None, delay_between_frames=200, cbar_width=0.05, cbar_label='', plot_steps=False):

    arr = landscape
    
    if N_rows == None:
        N_rows = arr.shape[0]

    if vmin==None:
        vmin = np.min(arr)
    if vmax==None:
        vmax = np.max(arr)
    
    if fig == None:
        fig = plt.figure()
        fig.set_tight_layout(True)
    if ax == None:
        ax = fig.add_subplot()

    cax = ax.inset_axes([1.03, 0, cbar_width, 1], transform=ax.transAxes)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ims = []

    for i in range(N_rows):
        
        cax.clear()
        
        im  = ax.imshow(arr[i,:].reshape(Nx, Ny), cmap=cmap, vmin=vmin, vmax=vmax,
                        origin='lower', interpolation=interpolation, animated=True)
        cb  = fig.colorbar(im, ax=ax, cax=cax, label=cbar_label)

        if plot_steps:
            ax.set_title(f"Step: {i+1}/{N_rows}")
        
        ims.append([im])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ani = animation.ArtistAnimation(fig, ims, interval=delay_between_frames, repeat_delay=delay_between_frames*10)

    return ani

##################################################################################
################################ TRAINING ########################################
##################################################################################

def sim_run_for_gradient_decent(thread : int, return_dic : dict, voltages : np.array, time_steps : np.array,
                                target_electrode : int, stat_size : int, network_topology : str,
                                topology_parameter : dict, initial_charge_vector=None):
    """Simulation execution for gradient decent algorithm. Inits a class and runs simulation for variable voltages.

    Parameters
    ----------
    thread : int
        Thread if
    return_dic : dict
        Dictonary containing simulation results
    voltages : np.array
        Voltage values
    time_steps : np.array
        Time Steps
    target_electrode : int
        Electrode associated to target observable
    stat_size : int
        Number of individual runs for statistics
    network_topology : str
        Network topology, either 'cubic' or 'random'
    topology_parameter : dict
        Dictonary containing information about topology
    initial_charge_vector : _type_, optional
        If not None, initial_charge_vector is used as the first network state, by default None
    """

    if initial_charge_vector is None:
        class_instance  = simulation.Simulation(network_topology=network_topology, topology_parameter=topology_parameter)
        class_instance.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                                        stat_size=stat_size, save=False, output_potential=True, init=True)
    else:
        class_instance  = simulation.Simulation(network_topology=network_topology, topology_parameter=topology_parameter)
        class_instance.init_based_on_charge_vector(voltages=voltages, initial_charge_vector=initial_charge_vector)
        class_instance.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                                        stat_size=stat_size, save=False, output_potential=True, init=False)
    
    # Get target observable 
    output_values       = class_instance.return_output_values()
    observable          = output_values[:,2]
    error_values        = output_values[:,3]
    return_dic[thread]  = observable

    if thread == 0:
        charge_values   = class_instance.return_microstates()[-1,:]
        # return_dic[-1]  = class_instance.ele_charge*np.round(charge_values/class_instance.ele_charge)
        return_dic[-1]  = charge_values

def loss_function(y_pred : np.array, y_real : np.array, transient=0)->float:
    """Root mean square error.

    Parameters
    ----------
    y_pred : np.array
        Predicted values
    y_real : np.array
        Actual values
    transient : int, optional
        Neglect the first transient steps, by default 0

    Returns
    -------
    float
        RMSE
    """
    return np.mean((y_pred[transient:]-y_real[transient:])**2)

def time_series_gradient_decent(x_vals : np.array, y_target : np.array, learning_rate : float, batch_size : int, N_epochs : int, network_topology : str, topology_parameter : dict,
                                epsilon=0.001, adam=False, time_step=1e-10, stat_size=500, Uc_init=0.05, transient_steps=0, print_nth_epoch=1,
                                save_nth_epoch=1, path=''):

    # Parameter
    N_voltages          = len(x_vals)
    N_batches           = int(N_voltages/batch_size)
    N_electrodes        = len(topology_parameter["e_pos"])
    N_controls          = N_electrodes - 2
    time_steps          = time_step*np.arange(N_voltages)
    control_voltages    = np.random.uniform(-Uc_init, Uc_init, N_controls)
    target_electrode    = N_electrodes - 1
        
    # Multiprocessing Manager
    with multiprocessing.Manager() as manager:

        # Storage for simulation results
        return_dic = manager.dict()
        current_charge_vector = None
        
        # ADAM Optimization
        if adam:
            m = np.zeros_like(control_voltages)
            v = np.zeros_like(control_voltages)

        for epoch in range(1, N_epochs+1):
            
            # Set charge vector to None
            predictions = np.zeros(N_voltages)
            
            for batch in range(N_batches):

                start   = batch*batch_size
                stop    = (batch+1)*batch_size
                
                # Voltage array containing input at column 0 
                voltages            = np.zeros(shape=(batch_size, N_electrodes+1))
                voltages[:,0]       = x_vals[start:stop]
                voltages[:,1:-2]    = np.tile(control_voltages, (batch_size,1))
                voltages_list       = []
                voltages_list.append(voltages)

                # Set up a list of voltages considering small deviations
                for i in range(N_controls):

                    voltages_tmp        = voltages.copy()
                    voltages_tmp[:,i+1] += epsilon
                    voltages_list.append(voltages_tmp)

                    voltages_tmp        = voltages.copy()
                    voltages_tmp[:,i+1] -= epsilon
                    voltages_list.append(voltages_tmp)

                # Container for processes
                procs = []

                # For each set of voltages assign start a process
                for thread in range(2*N_controls+1):

                    # Start process
                    process = multiprocessing.Process(target=sim_run_for_gradient_decent, args=(thread, return_dic, voltages_list[thread], time_steps,
                                                                                                target_electrode, stat_size, network_topology,
                                                                                                topology_parameter, current_charge_vector))
                    process.start()
                    procs.append(process)
                
                # Wait for all processes
                for p in procs:
                    p.join()

                # Current charge vector given the last iteration
                current_charge_vector = return_dic[-1]
                
                # Gradient Container
                gradients = np.zeros_like(control_voltages)

                # Calculate gradients for each control voltage 
                for i in np.arange(1,2*N_controls+1,2):

                    y_pred_eps_pos          = return_dic[i]
                    y_pred_eps_neg          = return_dic[i+1]
                    y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
                    y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
                    perturbed_loss_pos      = loss_function(y_pred=y_pred_eps_pos, y_real=y_target[(start+1):stop], transient=transient_steps)
                    perturbed_loss_neg      = loss_function(y_pred=y_pred_eps_neg, y_real=y_target[(start+1):stop], transient=transient_steps)
                    gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

                # Current prediction and loss
                y_pred  = return_dic[0]
                y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
                loss    = loss_function(y_pred=y_pred, y_real=y_target[(start+1):stop], transient=transient_steps)
                
                predictions[(start+1):stop] = y_pred

                # ADAM Optimization
                if adam:

                    beta1 = 0.9         # decay rate for the first moment
                    beta2 = 0.999       # decay rate for the second moment

                    # Update biased first and second moment estimate
                    m = beta1 * m + (1 - beta1) * gradients
                    v = beta2 * v + (1 - beta2) * (gradients ** 2)

                    # Compute bias-corrected first and second moment estimate
                    m_hat = m / (1 - beta1 ** epoch)
                    v_hat = v / (1 - beta2 ** epoch)

                    # Update control voltages given the gradients
                    control_voltages -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

                
                # Update control voltages given the gradients
                else:
                    control_voltages -= learning_rate * gradients

            # Print infos
            if epoch % print_nth_epoch == 0:
                print(f'Run : {epoch}')
                print(f'U_C : {control_voltages}')
                print(f"Loss : {loss}")

            # Save prediction
            if epoch % save_nth_epoch == 0:
                np.savetxt(fname=f"{path}ypred_{epoch}.csv", X=predictions)
                np.savetxt(fname=f"{path}charge_{epoch}.csv", X=current_charge_vector)

def metropolis_criterion(delta_func, beta):
    return np.exp(-beta * delta_func)

def return_res(np_network_sim : simulation.Simulation, time : np.array, voltages : np.array, stat_size=20, I_to_U=1e-4, fit_after_n=0):

    currents = []

    for i in range(stat_size):
        np_network_sim.run_var_voltages(voltages=voltages, time_steps=time, target_electrode=(np_network_sim.N_electrodes-1), save_th=0.1, init=True, eq_steps=0)
        currents.append(np_network_sim.return_output_values()[:,2])
    
    I_mean  = np.mean(currents, axis=0)
    I_std   = np.std(currents, axis=0)/np.sqrt(stat_size)
    res     = np.sum((I_to_U*I_mean[fit_after_n:] - voltages[fit_after_n+1:,0])**2)

    return I_mean[fit_after_n:], I_std[fit_after_n:], res

def metropolis_optimization(np_network_sim : simulation.Simulation, time : np.array, voltages : np.array, n_runs : int,
                            gamma : float, beta : float, V_std=0.01, Vg_std=0.01, I_to_U=1e-4, stat_size=20, save_best_at=None, fit_after_n=0):

    N_voltages          = voltages.shape[0]
    current_voltages    = voltages
    best_voltages       = voltages
    new_voltages        = voltages
    I_mean, I_std, res  = return_res(np_network_sim, time, current_voltages, stat_size, I_to_U, fit_after_n)
    f_val               = 1.
    best_res            = res

    for i in range(n_runs):

        V_sample                                            = np.tile(np.random.normal(loc=0, scale=V_std, size=(np_network_sim.N_electrodes-2)), (N_voltages,1))
        Vg_sample                                           = np.repeat(np.random.normal(loc=0, scale=Vg_std), N_voltages)
        new_voltages[:,1:(np_network_sim.N_electrodes-1)]   = current_voltages[:,1:(np_network_sim.N_electrodes-1)] + V_sample
        new_voltages[:,-1]                                  = current_voltages[:,-1] + Vg_sample
        I_mean, I_std, res                                  = return_res(np_network_sim, time, new_voltages, stat_size, I_to_U, fit_after_n)
        f_val_new                                           = 1 - np.exp(-gamma*(res - best_res))
        delta_f                                             = f_val_new - f_val

        if ((delta_f < 0) or (np.random.rand() < metropolis_criterion(delta_f, beta))):

            current_voltages    = new_voltages
            best_res            = res
            best_voltages       = current_voltages
            f_val               = f_val_new

            if save_best_at != None:

                np.savetxt(save_best_at+"best_volt_{i}.csv", best_voltages)
                np.savetxt(save_best_at+"I_mean_{i}.csv", I_mean)
                np.savetxt(save_best_at+"I_std_{i}.csv", I_std)

    return best_voltages, best_res, I_mean, I_std

def train_test_split(time, voltages, train_length, test_length, prediction_distance):

    u_train = voltages[:train_length]
    y_train = voltages[prediction_distance:(train_length+prediction_distance)]

    u_test  = voltages[(train_length):(train_length+test_length)]
    y_test  = voltages[(train_length+prediction_distance):(train_length+test_length+prediction_distance)]

    t_train = time[:(train_length)]
    t_test  = time[(train_length):(train_length+test_length)]

    return t_train, u_train, y_train, t_test, u_test, y_test

def train_test_split_memory(time, voltages, train_length, test_length, remember_distance, offset=100):

    u_train = voltages[offset:(train_length+offset)]
    y_train = voltages[(offset-remember_distance):(train_length+offset-remember_distance)]

    u_test  = voltages[(train_length+offset):(train_length+offset+test_length)]
    y_test  = voltages[(train_length+offset-remember_distance):(train_length+offset+test_length-remember_distance)]

    t_train = time[offset:(train_length+offset)]
    t_test  = time[(train_length+offset):(train_length+offset+test_length)]

    return t_train, u_train, y_train, t_test, u_test, y_test

def train_test_split(time, voltages, train_length, test_length, prediction_distance=1):

    u_train = voltages[:(train_length)]
    y_train = voltages[(prediction_distance):(train_length+prediction_distance)]

    u_test  = voltages[(train_length):(train_length+test_length)]
    y_test  = voltages[(train_length+prediction_distance):(train_length+test_length+prediction_distance)]

    t_train = time[:(train_length)]
    t_test  = time[(train_length):(train_length+test_length)]

    return t_train, u_train, y_train, t_test, u_test, y_test

def select_electrode_currents(np_network_sim : simulation.Simulation):

    # Return Network Currents
    jump_paths, network_I   = np_network_sim.return_network_currents()
    network_I_df            = pd.DataFrame(network_I)
    network_I_df.columns    = jump_paths
    network_I_df            = network_I_df.reset_index(drop=True)

    # Select Electrode Currents
    electrode_currents  = pd.DataFrame()

    for i in range(np_network_sim.N_electrodes):

        for col in jump_paths:
            if col[0] == i:
                a = network_I_df[col]
            if col[1] == i:
                b = network_I_df[col]
        
        diff                        = b - a
        electrode_currents[f'I{i}'] = diff

    return electrode_currents

def memory_capacity_simulation(time, voltages, train_length, test_length, remember_distance, network_topology, topology_parameter, np_info=None,
                               save_th=10, folder='data/', regularization_coeff=1e-8, R=25, Rstd=0, path_info='', init=True):

    # Train Test Split
    t_train, u_train, y_train, t_test, u_test, y_test = train_test_split_memory(time=time, voltages=voltages, train_length=train_length,
                                                                                test_length=test_length, remember_distance=remember_distance)
    
    # Run Train Simulation
    np_network_sim = simulation.Simulation(network_topology=network_topology, topology_parameter=topology_parameter, np_info=np_info,
                                         folder=folder, add_to_path=f'_mc_train_{remember_distance}{path_info}')
    np_network_sim.run_var_voltages(voltages=u_train, time_steps=t_train, target_electrode=(np_network_sim.N_electrodes-1),
                                    save_th=save_th, R=R, Rstd=Rstd, init=init)

    # Return Electrode Currents
    electrode_currents      = select_electrode_currents(np_network_sim)
    electrode_currents['t'] = t_train[1:]
    electrode_currents['U'] = u_train[1:,0]

    # Save Train Electrode Currents 
    electrode_currents.to_csv(f"{folder}mc_train_results_{remember_distance}{path_info}.csv")

    # Regression of Electrode Currents and Train Target
    X       = electrode_currents.loc[:,'I1':f'I{np_network_sim.N_electrodes-1}'].T.values
    y       = y_train[1:,0].copy()
    W_out   = np.linalg.solve(np.dot(X,X.T) + regularization_coeff*np.eye(X.shape[0]), np.dot(X,y.T)).T

    # Store Wout Matrix
    np.savetxt(f"{folder}Wout_{remember_distance}{path_info}.csv", W_out)

    # Run Test Simulation
    np_network_sim.run_var_voltages(voltages=u_test, time_steps=t_test, target_electrode=(np_network_sim.N_electrodes-1),
                                    save_th=save_th, R=R, Rstd=Rstd, init=False)

    # Return Electrode Currents
    electrode_currents      = select_electrode_currents(np_network_sim)
    electrode_currents['t'] = t_test[1:]
    electrode_currents['U'] = u_test[1:,0]
    X                       = electrode_currents.loc[:,'I1':f'I{np_network_sim.N_electrodes-1}'].values

    # Predict y_test based on W_out
    y_pred  = []

    for val in X:

        y_val   = np.dot(W_out,val[:,np.newaxis])[0]
        y_pred.append(y_val)

    electrode_currents['y_test'] = y_test[1:,0]
    electrode_currents['y_pred'] = np.array(y_pred)
    
    # Save Test Electrode Currents 
    electrode_currents.to_csv(f"{folder}mc_test_results_{remember_distance}{path_info}.csv")

def train_reservoir(time, voltages, train_length, test_length, prediction_distance, network_topology, topology_parameter, np_info=None,
                    save_th=10, folder='data/', regularization_coeff=1e-8, R=25, Rstd=0, path_info=''):

    # Train Test Split
    t_train, u_train, y_train, t_test, u_test, y_test = train_test_split(time=time, voltages=voltages, train_length=train_length,
                                                                        test_length=test_length, prediction_distance=prediction_distance)
    
    # Run Train Simulation
    np_network_sim = simulation.Simulation(network_topology=network_topology, topology_parameter=topology_parameter, np_info=np_info,
                                         folder=folder, add_to_path=f'_train_{prediction_distance}{path_info}')
    np_network_sim.run_var_voltages(voltages=u_train, time_steps=t_train, target_electrode=(np_network_sim.N_electrodes-1), save_th=save_th, R=R, Rstd=Rstd)

    # Return Electrode Currents
    electrode_currents      = select_electrode_currents(np_network_sim)
    electrode_currents['t'] = t_train[1:]
    electrode_currents['U'] = u_train[1:,0]

    # Save Train Electrode Currents 
    electrode_currents.to_csv(f"{folder}train_results_{prediction_distance}{path_info}.csv")

    # Regression of Electrode Currents and Train Target
    X       = electrode_currents.loc[:,'I1':f'I{np_network_sim.N_electrodes-1}'].T.values
    y       = y_train[1:,0].copy()
    W_out   = np.linalg.solve(np.dot(X,X.T) + regularization_coeff*np.eye(X.shape[0]), np.dot(X,y.T)).T

    # Store Wout Matrix
    np.savetxt(f"{folder}Wout_{prediction_distance}{path_info}.csv", W_out)

# def predict(time, voltages, n_predictions, np_net)

#     # Run Test Simulation
#     np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, np_info=np_info,
#                                          folder=folder, add_to_path=f'_test_{prediction_distance}{path_info}')
#     np_network_sim.run_var_voltages(voltages=u_test, time_steps=t_test, target_electrode=(np_network_sim.N_electrodes-1), save_th=save_th, R=R, Rstd=Rstd)

#     # Return Electrode Currents
#     electrode_currents      = select_electrode_currents(np_network_sim)
#     electrode_currents['t'] = t_test[1:]
#     electrode_currents['U'] = u_test[1:,0]
#     X                       = electrode_currents.loc[:,'I1':f'I{np_network_sim.N_electrodes-1}'].values

#     # Predict y_test based on W_out
#     y_pred  = []

#     for val in X:

#         y_val   = np.dot(W_out,val[:,np.newaxis])[0]
#         y_pred.append(y_val)

#     electrode_currents['y_test'] = y_test[1:,0]
#     electrode_currents['y_pred'] = np.array(y_pred)
    
#     # Save Test Electrode Currents 
#     electrode_currents.to_csv(f"{folder}test_results_{prediction_distance}{path_info}.csv")