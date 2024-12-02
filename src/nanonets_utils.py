import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import networkx as nx
import nanonets
import scienceplots
import multiprocessing
from typing import Union, Tuple, List
from scipy.signal.windows import hann
from pyDOE import lhs

blue_color  = '#348ABD'
red_color   = '#A60628'

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# SAMPLING METHODS
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def uniform_sample(U_e : Union[float, List[float]], N_samples : int, topology_parameter : dict, U_g : Union[float, List[float]]=0.0)->np.array:
    """Returns a uniform sample of electrode voltages, with floating electrodes set Zero and Gate electrode defined individually.

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

    # Set floating electrodes to 0V
    sample[:,floating_idx]  = 0.0

    return sample

def lhs_sample(U_e : Union[float, List[float]], N_samples : int, topology_parameter : dict, U_g : Union[float, List[float]]=0.0)->np.array:
    """Returns a Latin Hypercube sample of electrode voltages, with floating electrodes set Zero and Gate electrode defined individually.

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
    lhs_samples     = lhs(N_electrodes+1, samples=N_samples)
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
    scaled_samples[:,floating_idx]  = 0.0

    return scaled_samples

def logic_gate_sample(U_e : Union[float, List[float]], input_pos : List[int], N_samples : int, topology_parameter : dict,
                      U_i : Union[float, List[float]]=0.01, U_g : Union[float, List[float]]=0.0, sample_technique='lhs')->np.array:
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
        sample = lhs_sample(U_e=U_e, N_samples=N_samples, topology_parameter=topology_parameter, U_g=U_g)
    elif sample_technique == 'uniform':
        sample = uniform_sample(U_e=U_e, N_samples=N_samples, topology_parameter=topology_parameter, U_g=U_g)
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

def sinusoidal_voltages(frequencies : Union[float, List[float]], amplitudes : Union[float, List[float]], N_samples : int,
                        topology_parameter : dict, time_step : float=1e-10)->Tuple[np.array,np.array]:
    """Return voltage array containing sinusoidal signals of given frequencies and amplitudes

    Parameters
    ----------
    frequencies : Union[float, List[float]]
        Single frequency for all constant electrodes or individual frequency for each electrode
    amplitudes : Union[float, List[float]]
        Single amplitude for all constant electrodes or individual amplitude for each electrode
    N_samples : int
        Number of voltage values
    topology_parameter : dict
        Network topology dictonary
    time_step : float, optional
        Time step size, by default 1e-10

    Returns
    -------
    Tuple[np.array,np.array]
        Time and Voltages
    """
    # Voltages and Time Scale
    voltages        = np.zeros(shape=(N_samples, len(topology_parameter["e_pos"])+1))
    time_steps      = time_step*np.arange(N_samples)
    
    # Parameter based on electrode specification
    electrode_types = np.array(topology_parameter["electrode_type"])
    floating_idx    = np.where(electrode_types=="floating")[0]

    # For variable frequencies
    if isinstance(frequencies, list):
        # For variable amplitudes
        if isinstance(amplitudes, list):
            for i in range(voltages.shape[1]-1):
                voltages[:,i]   = amplitudes[i]*np.cos(frequencies[i]*time_steps*1e8)
        else:
            for i in range(voltages.shape[1]-1):
                voltages[:,i]   = amplitudes*np.cos(frequencies[i]*time_steps*1e8)
    else:
        if isinstance(amplitudes, list):
            for i in range(voltages.shape[1]-1):
                voltages[:,i]   = amplitudes[i]*np.cos(frequencies*time_steps*1e8)
        else:
            for i in range(voltages.shape[1]-1):
                voltages[:,i]   = amplitudes*np.cos(frequencies*time_steps*1e8)

    # Set floating electrodes to 0V
    voltages[:,floating_idx] = 0.0
    
    return time_steps, voltages 

def distribute_array_across_processes(process : int, data : np.array, N_processes : int)->np.array:
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
                data = {key : pd.read_csv(f"{folder}Nx={key}_Ny={key}_Nz=1_Ne={N_e}.csv") for key in N}

        else:
            for key in N:
                if disordered:
                    data[key] = pd.read_csv(f"{folder}Np={key}_Nj=4_Ne={N_e}.csv")
                else:
                    data[key] = pd.read_csv(f"{folder}Nx={key}_Ny={key}_Nz=1_Ne={N_e}.csv")

    # For variable numbers of electrodes
    elif (isinstance(N, int) and isinstance(N_e, list)):

        # Without dictonary input
        if data is None:
            if disordered:
                data = {key : pd.read_csv(f"{folder}Np={N}_Nj=4_Ne={key}.csv") for key in N_e}
            else:
                data = {key : pd.read_csv(f"{folder}Nx={N}_Ny={N}_Nz=1_Ne={N_e}.csv") for key in N_e}

        else:
            for key in N_e:
                if disordered:
                    data[key] = pd.read_csv(f"{folder}Np={N}_Nj=4_Ne={key}.csv")
                else:
                    data[key] = pd.read_csv(f"{folder}Nx={N}_Ny={N}_Nz=1_Ne={key}.csv")

    else:

        if disordered:
            data  = pd.read_csv(f"{folder}Np={N}_Nj=4_Ne={N_e}.csv")
        else:
            data  = pd.read_csv(f"{folder}Nx={N}_Ny={N}_Nz=1_Ne={N_e}.csv")

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

    # Prepare column names for sorting
    sort_cols   = [f'E{i}' for i in range(N_e) if 'E{i}' not in input_cols] + ['G'] + input_cols

    # Copy DataFame and drop Zeros
    data    = df.copy()
    data    = data[data['Observable'].abs() > 0.0] if drop_zero else data

    # Sort the DataFrame by relevant columns
    data    = data.sort_values(by=sort_cols)
    data    = data.reset_index(drop=True)
    N_data  = len(data)
    
    # Collect indices of rows to drop
    rows_to_drop    = []
    i_input1        = input_cols[0]
    i_input2        = input_cols[1]

    while i < N_data-3:

        cond1   = (data[i_input1][i]      == off_state)   and (data[i_input2][i]      == off_state)
        cond2   = (data[i_input1][i + 1]  == off_state)   and (data[i_input2][i + 1]  == on_state)
        cond3   = (data[i_input1][i + 2]  == on_state)    and (data[i_input2][i + 2]  == off_state)
        cond4   = (data[i_input1][i + 3]  == on_state)    and (data[i_input2][i + 3]  == on_state)
        
        cond5   = data['G'][i] == data['G'][i + 1]
        cond6   = data['G'][i] == data['G'][i + 2]
        cond7   = data['G'][i] == data['G'][i + 3]

        # If any condition fails, mark the rows for dropping
        if not (cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7):
            rows_to_drop.extend([i])
            i += 1
        else:
            i += 4
    
    # Drop the rows at once and reset the index
    data = data.drop(rows_to_drop).reset_index(drop=True)

    return data

def load_boolean_results(folder : str, N : Union[int, List[int]], N_e : Union[int, List[int]], disordered=False, boot_steps=0, i1_col=1, i2_col=3, o_col=7,
                         min_currents=0.0, min_error=0.0, max_error=np.inf, dic=None, dic_nc=None, off_state=[0.0], on_state=[0.01])->Tuple[dict,dict]:

    # For variable numbers of nanoparticles
    if (isinstance(N, list) and isinstance(N_e, int)):

        # DataFrame Columns
        columns = [f'C{i}' for i in range(1,N_e-2)] + ['G','Jumps_eq','Jumps','Current','Error']
        columns.insert(i1_col,'I1')
        columns.insert(i2_col,'I2')
        columns.insert(o_col,'O')
        new_cols = ['I1','I2'] + [f'C{i}' for i in range(1,N_e-2)] + ['G','Jumps_eq','Jumps','Current','Error']

        # Without dictonary input make new
        if (dic==None and dic_nc==None):

            dic     = {}
            dic_nc  = {}

        # For variable numbers of nanoparticles
        for index,i in enumerate(N):
            
            if disordered:
                df  = pd.read_csv(folder+f"Np={i}_Nj=4_Ne={N_e}.csv")
            else:
                df  = pd.read_csv(folder+f"Nx={i}_Ny={i}_Nz=1_Ne={N_e}.csv")
                
            df          = df.round(4)
            df.columns  = columns
            df          = df[new_cols]
            df1         = df.copy()

            # Drop failed simulations
            df1         = df1[df1['Error'] != 0].reset_index(drop=True)

            # Drop simulations outsite accepted errors
            df1         = df1[(np.round(df1['Error']/df1['Current'].abs(),2) >= min_error) &
                              (np.round(df1['Error']/df1['Current'].abs(),2) < max_error)].reset_index(drop=True)

            # Sort and prepare for fitness calculation (drop voltages without full set of input combinations)
            df1         = df1.sort_values(by=['C1','G','I1','I2'], ignore_index=True)
            df1         = prepare_for_fitness_calculation(df=df1, min_current=0.0, N_c=(N_e-3),
                                                          off_state=off_state[index], on_state=on_state[index])
            
            # Copy all droped values to df2
            df2         = df[~df['G'].isin(df1['G'])]
            df2         = df2.sort_values(by=['C1','G','I1','I2'], ignore_index=True)

            # For bootstrapping
            if boot_steps != 0:

                # Sample new currents based on errors
                df_var1     = vary_currents_by_error(df=df1, M=boot_steps)
                df_var2     = vary_currents_by_error(df=df2, M=boot_steps)
                df_var2.loc[:,'Current'] = 0.0

                # Set values below minimum current to minimum current
                if min_currents != 0:

                    df_var1.loc[(df_var1['Current'].abs() < min_currents) & (df_var1['Current'] >= 0), 'Current'] = min_currents
                    df_var1.loc[(df_var1['Current'].abs() < min_currents) & (df_var1['Current'] < 0), 'Current'] = -min_currents
                    df_var2.loc[(df_var2['Current'].abs() < min_currents) & (df_var2['Current'] >= 0), 'Current'] = min_currents
                    df_var2.loc[(df_var2['Current'].abs() < min_currents) & (df_var2['Current'] < 0), 'Current'] = -min_currents

                # Append to dictonaries
                dic[i]      = df_var1
                dic_nc[i]   = df_var2

            else:

                # Append to dictonaries
                dic[i]      = df1
                dic_nc[i]   = df2

    # For variable numbers of electrodes
    elif (isinstance(N, int) and isinstance(N_e, list)):
        
        # Without dictonary input make new
        if (dic==None and dic_nc==None):

            dic     = {}
            dic_nc  = {}
        
        # For variable numbers of electrodes
        for index, i in enumerate(N_e):

            # DataFrame Columns
            columns = [f'C{i}' for i in range(1,i-2)] + ['G','Jumps_eq','Jumps','Current','Error']
            columns.insert(i1_col,'I1')
            columns.insert(i2_col,'I2')
            columns.insert(i-1,'O')
            new_cols = ['I1','I2'] + [f'C{i}' for i in range(1,i-2)] + ['G','Jumps_eq','Jumps','Current','Error']
            
            if disordered:
                df  = pd.read_csv(folder+f"Np={N}_Nj=4_Ne={i}.csv")
            else:
                df  = pd.read_csv(folder+f"Nx={N}_Ny={N}_Nz=1_Ne={i}.csv")
            df.columns  = columns
            df          = df[new_cols]
            df1         = df.copy()

            # Drop failed simulations
            df1         = df1[df1['Error'] != 0].reset_index(drop=True)
            
            # Drop simulations outsite accepted errors
            df1         = df1[(np.round(df1['Error']/df1['Current'].abs(),2) >= min_error) &
                              (np.round(df1['Error']/df1['Current'].abs(),2) < max_error)].reset_index(drop=True)
            
            # Sort and prepare for fitness calculation (drop voltages without full set of input combinations)
            df1         = df1.sort_values(by=['C1','G','I1','I2'], ignore_index=True)
            df1         = prepare_for_fitness_calculation(df=df1, min_current=0.0, N_c=(i-3),
                                                          off_state=off_state[index], on_state=on_state[index])
            
            # Copy all droped values to df2
            df2         = df[~df['G'].isin(df1['G'])]
            df2         = df2.sort_values(by=['C1','G','I1','I2'], ignore_index=True)

            # For bootstrapping
            if boot_steps != 0:
                
                # Sample new currents based on errors
                df_var1     = vary_currents_by_error(df=df1, M=boot_steps)
                df_var2     = vary_currents_by_error(df=df2, M=boot_steps)
                df_var2.loc[:,'Current'] = 0.0

                # Set values below minimum current to minimum current
                if min_currents != 0:
                    df_var1.loc[(df_var1['Current'].abs() < min_currents) & (df_var1['Current'] >= 0), 'Current'] = min_currents
                    df_var1.loc[(df_var1['Current'].abs() < min_currents) & (df_var1['Current'] < 0), 'Current'] = -min_currents
                    df_var2.loc[(df_var2['Current'].abs() < min_currents) & (df_var2['Current'] >= 0), 'Current'] = min_currents
                    df_var2.loc[(df_var2['Current'].abs() < min_currents) & (df_var2['Current'] < 0), 'Current'] = -min_currents

                # Append to dictonaries
                dic[i]      = df_var1
                dic_nc[i]   = df_var2

            else:

                # Append to dictonaries
                dic[i]      = df1
                dic_nc[i]   = df2

    else:

        # DataFrame Columns
        columns = [f'C{i}' for i in range(1,N_e-2)] + ['G','Jumps_eq','Jumps','Current','Error']
        columns.insert(i1_col,'I1')
        columns.insert(i2_col,'I2')
        columns.insert(o_col,'O')
        new_cols = ['I1','I2'] + [f'C{i}' for i in range(1,N_e-2)] + ['G','Jumps_eq','Jumps','Current','Error']

        # Load DataFrame
        if disordered:
            df  = pd.read_csv(folder+f"Np={N}_Nj=4_Ne={N_e}.csv")
        else:
            df  = pd.read_csv(folder+f"Nx={N}_Ny={N}_Nz=1_Ne={N_e}.csv")
        df          = df.round(4)
        df.columns  = columns
        df          = df[new_cols]
        df1         = df.copy()
        
        # Drop failed simulations
        df1         = df1[df1['Error'] != 0].reset_index(drop=True)

        # Drop simulations outsite accepted errors
        df1         = df1[(np.round(df1['Error']/df1['Current'].abs(),2) >= min_error) &
                            (np.round(df1['Error']/df1['Current'].abs(),2) < max_error)].reset_index(drop=True)
        
        # Sort and prepare for fitness calculation (drop voltages without full set of input combinations)
        df1         = df1.sort_values(by=['C1','G','I1','I2'], ignore_index=True)
        df1         = prepare_for_fitness_calculation(df=df1, min_current=0.0, N_c=(N_e-3),
                                                        off_state=off_state[0], on_state=on_state[0])
        
        # Copy all droped values to df2
        df2         = df[~df['G'].isin(df1['G'])]
        df2         = df2.sort_values(by=['C1','G','I1','I2'], ignore_index=True)

        # For bootstrapping
        if boot_steps != 0:

            # Sample new currents based on errors
            df_var1     = vary_currents_by_error(df=df1, M=boot_steps)
            df_var2     = vary_currents_by_error(df=df2, M=boot_steps)
            df_var2.loc[:,'Current'] = 0.0

            # Set values below minimum current to minimum current
            if min_currents != 0:

                df_var1.loc[(df_var1['Current'].abs() < min_currents) & (df_var1['Current'] >= 0), 'Current']   = min_currents
                df_var1.loc[(df_var1['Current'].abs() < min_currents) & (df_var1['Current'] < 0), 'Current']    = -min_currents
                df_var2.loc[(df_var2['Current'].abs() < min_currents) & (df_var2['Current'] >= 0), 'Current']   = min_currents
                df_var2.loc[(df_var2['Current'].abs() < min_currents) & (df_var2['Current'] < 0), 'Current']    = -min_currents

            return df_var1, df_var2

        else:

            return df1, df2

    return dic, dic_nc

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
# MISC
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def standard_scale(arr : np.array)->np.array:
    """Standard scale: y = ( y - mean(y) ) / std(y) a 1D array

    Parameters
    ----------
    arr : np.array
        Data

    Returns
    -------
    np.array
        Standard scaled data
    """
    return (arr - np.mean(arr))/np.std(arr)

def poincare_map_zero_corssing(arr : np.array)->np.array:
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

#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#
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

def autocorrelation(x : np.array, y : np.array, lags : int)->np.array:
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

    return [np.corrcoef(x, y)[0,1] if l==0 else np.corrcoef(x[:-l], y[l:])[0,1] for l in range(lags)]



# def prepare_for_fitness_calculation(df : pd.DataFrame, N_c : int, min_current=None, input1_col='I1', input2_col='I2',
#     gate_col='G', control_col='C', current_col='Current', off_state=0.0, on_state=0.01) -> pd.DataFrame:
#     """
#     Prepares a pandas Dataframe of electric currents for calculation of gate fitnesses\\
#     Number of Controls NC must be provided as argument\\
#     Allows to exclude any currents having an absolute value less than min_current if min_current != None\\
#     Allows to check df if for each voltage combination exactly all possible input states are present\\ 
#     Other Attributes define column names for various features
#     """

#     # Sort Dataframe by C1,C2,..G,I1,I2 and exclude Currents if min_current != None
#     sort_cols = [control_col + '{}'.format(i) for i in range(1, N_c + 1)]
#     sort_cols.extend([gate_col, input1_col, input2_col])
    
#     data    = df.copy()
    
#     if min_current != None:
#         # data.loc[data[current_col].abs() <= min_current, current_col] = min_current
#         data = data[data[current_col].abs() > min_current]
    
#     data    = data.sort_values(by=sort_cols)
#     data    = data.reset_index(drop=True)

#     # Move through each row and proof that for each electrode voltage combination all possible input states are present
#     i       = 0
#     rows    = np.floor(len(data))

#     while (i < rows):

#         try:

#             cond1 = ((data[input1_col][i]   == off_state) and (data[input2_col][i]   == off_state))
#             cond2 = ((data[input1_col][i+1] == off_state) and (data[input2_col][i+1] == on_state))
#             cond3 = ((data[input1_col][i+2] == on_state)  and (data[input2_col][i+2] == off_state))
#             cond4 = ((data[input1_col][i+3] == on_state)  and (data[input2_col][i+3] == on_state))
            
#             cond5 = (data[gate_col][i] == data[gate_col][i+1]) 
#             cond6 = (data[gate_col][i] == data[gate_col][i+2]) 
#             cond7 = (data[gate_col][i] == data[gate_col][i+3])

#             cond8 = (data[control_col + '1'][i] == data[control_col + '1'][i+1]) 
#             cond9 = (data[control_col + '1'][i] == data[control_col + '1'][i+2]) 
#             cond10= (data[control_col + '1'][i] == data[control_col + '1'][i+3])
        
#         except:

#             try:
#                 data = data.drop(i)
#             except:
#                 pass

#             try:
#                 data = data.drop(i+1)
#             except:
#                 pass

#             try:
#                 data = data.drop(i+2)
#             except:
#                 pass

#             data = data.reset_index(drop=True)
#             break

#         if not(cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond7 and cond8 and cond9 and cond10):

#             data    = data.drop(i)
#             data    = data.reset_index(drop=True)
#             rows    = rows - 1
#             continue

#         i = i + 4
    
#     return data

def get_on_off_rss(df00 : pd.DataFrame, df01 : pd.DataFrame, df10 : pd.DataFrame, df11 : pd.DataFrame, gate : str, all=False) -> pd.DataFrame:
    """
    Get off and on state for fitness calculation for gate provided as string\\
    Get residual sums of squares as sqrt(RSS/4)
    """

    df = pd.DataFrame()

    if gate == 'AND':

        df['off']   = (df00['Current'] + df01['Current'] + df10['Current'])/3
        df['on']    = df11['Current']
        df['res']   = np.sqrt(((df00['Current'] - df['off'])**2 + (df01['Current'] - df['off'])**2 + (df10['Current'] - df['off'])**2 + (df11['Current'] - df['on'])**2)/4)

    elif gate == 'OR':

        df['off']   = df00['Current']
        df['on']    = (df01['Current'] + df10['Current'] + df11['Current'])/3
        df['res']   = np.sqrt(((df00['Current'] - df['off'])**2 + (df01['Current'] - df['on'])**2 + (df10['Current'] - df['on'])**2 + (df11['Current'] - df['on'])**2)/4)

    elif gate == 'XOR':

        df['off']   = (df00['Current'] + df11['Current'])/2
        df['on']    = (df01['Current'] + df10['Current'])/2
        df['res']   = np.sqrt(((df00['Current'] - df['off'])**2 + (df01['Current'] - df['on'])**2 + (df10['Current'] - df['on'])**2 + (df11['Current'] - df['off'])**2)/4)

    elif gate == 'NAND':

        df['on']    = (df00['Current'] + df01['Current'] + df10['Current'])/3
        df['off']   = df11['Current']
        df['res']   = np.sqrt(((df00['Current'] - df['on'])**2 + (df01['Current'] - df['on'])**2 + (df10['Current'] - df['on'])**2 + (df11['Current'] - df['off'])**2)/4)

    elif gate == 'NOR':

        df['on']    = df00['Current']
        df['off']   = (df01['Current'] + df10['Current'] + df11['Current'])/3
        df['res']   = np.sqrt(((df00['Current'] - df['on'])**2 + (df01['Current'] - df['off'])**2 + (df10['Current'] - df['off'])**2 + (df11['Current'] - df['off'])**2)/4)

    elif gate == 'XNOR':

        df['on']    = (df00['Current'] + df11['Current'])/2
        df['off']   = (df01['Current'] + df10['Current'])/2
        df['res']   = np.sqrt(((df00['Current'] - df['on'])**2 + (df01['Current'] - df['off'])**2 + (df10['Current'] - df['off'])**2 + (df11['Current'] - df['on'])**2)/4)
    
    elif gate == 'P':

        df['on']    = (df10['Current'] + df11['Current'])/2
        df['off']   = (df00['Current'] + df01['Current'])/2
        df['res']   = np.sqrt(((df00['Current'] - df['off'])**2 + (df01['Current'] - df['off'])**2 + (df10['Current'] - df['on'])**2 + (df11['Current'] - df['on'])**2)/4)
    
    elif gate == 'notP':

        df['on']    = (df00['Current'] + df01['Current'])/2
        df['off']   = (df10['Current'] + df11['Current'])/2
        df['res']   = np.sqrt(((df00['Current'] - df['on'])**2 + (df01['Current'] - df['on'])**2 + (df10['Current'] - df['off'])**2 + (df11['Current'] - df['off'])**2)/4)

    elif gate == 'Q':

        df['on']    = (df01['Current'] + df11['Current'])/2
        df['off']   = (df00['Current'] + df10['Current'])/2
        df['res']   = np.sqrt(((df00['Current'] - df['off'])**2 + (df01['Current'] - df['on'])**2 + (df10['Current'] - df['off'])**2 + (df11['Current'] - df['on'])**2)/4)
    
    elif gate == 'notQ':

        df['on']    = (df00['Current'] + df10['Current'])/2
        df['off']   = (df01['Current'] + df11['Current'])/2
        df['res']   = np.sqrt(((df00['Current'] - df['on'])**2 + (df01['Current'] - df['off'])**2 + (df10['Current'] - df['on'])**2 + (df11['Current'] - df['off'])**2)/4)

    elif gate == 'PnotQ':

        df['on']    = df10['Current']
        df['off']   = (df00['Current'] + df01['Current'] + df11['Current'])/3
        df['res']   = np.sqrt(((df00['Current'] - df['off'])**2 + (df01['Current'] - df['off'])**2 + (df10['Current'] - df['on'])**2 + (df11['Current'] - df['off'])**2)/4)
    
    elif gate == 'notPQ':

        df['on']    = df01['Current']
        df['off']   = (df00['Current'] + df10['Current'] + df11['Current'])/3
        df['res']   = np.sqrt(((df00['Current'] - df['off'])**2 + (df01['Current'] - df['on'])**2 + (df10['Current'] - df['off'])**2 + (df11['Current'] - df['off'])**2)/4)
    
    elif gate == 'notPandQ':

        df['on']    = (df00['Current'] + df01['Current'] + df11['Current'])/3
        df['off']   = df10['Current']
        df['res']   = np.sqrt(((df00['Current'] - df['on'])**2 + (df01['Current'] - df['on'])**2 + (df10['Current'] - df['off'])**2 + (df11['Current'] - df['on'])**2)/4)
    
    elif gate == 'PandnotQ':

        df['on']    = (df00['Current'] + df10['Current'] + df11['Current'])/3
        df['off']   = df01['Current']
        df['res']   = np.sqrt(((df00['Current'] - df['on'])**2 + (df01['Current'] - df['off'])**2 + (df10['Current'] - df['on'])**2 + (df11['Current'] - df['on'])**2)/4)

    return df

def fitness(df : pd.DataFrame, N_controls : int, delta=0.01, min_current=None, off_state=0.0, on_state=0.01,
    gates=['AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR'], input1_column = 'I1', input2_column = 'I2', gate_column = 'G',
    controls_column = 'C', current_column = 'Current', error_column='Error') -> pd.DataFrame:
    """
    Calculate Fitness based on m/(sqrt(RSS/4) + delta*|c|)\\
    Input pandas Dataframe of electric currents and provide number of controls Nc\\
    Before starting the fitness calculation: Allows to exclude any currents having an absolute value less than min_current and check if all possible input states are present
    """

    if min_current != None:
        df = prepare_for_fitness_calculation(df=df, N_c=N_controls, min_current=min_current, input1_col=input1_column, input2_col=input2_column,
                                             gate_col=gate_column, control_col=controls_column, current_col=current_column, off_state=off_state, on_state=on_state)

    columns         = [gate_column] + [controls_column + '{}'.format(i) for i in range(1, N_controls+1)]
    
    df00            = df.copy()
    df00            = df00[(df00[input1_column] == off_state) & (df00[input2_column] == off_state)]
    df00            = df00.sort_values(by=columns)
    df00            = df00.reset_index(drop=True)

    df01            = df.copy()
    df01            = df01[(df01[input1_column] == off_state) & (df01[input2_column] == on_state)]
    df01            = df01.sort_values(by=columns)
    df01            = df01.reset_index(drop=True)
 
    df10            = df.copy()
    df10            = df10[(df10[input1_column] == on_state) & (df10[input2_column] == off_state)]
    df10            = df10.sort_values(by=columns)
    df10            = df10.reset_index(drop=True)

    df11            = df.copy()
    df11            = df11[(df11[input1_column] == on_state) & (df11[input2_column] == on_state)]
    df11            = df11.sort_values(by=columns)
    df11            = df11.reset_index(drop=True)
    
    # Setup fitness dataframe with columns based on current dataframe columns except inputs and current + error
    fitness         = pd.DataFrame(columns=list(df00.columns) + [g + ' Fitness' for g in gates])
    if error_column != None:
        fitness         = fitness.drop(columns=[current_column,error_column])
    else:
        fitness         = fitness.drop(columns=[current_column])

    # Input Parameter and init gate fitness as nan
    for col in list(df00.columns):

        if ((col != "Current") and (col != "Error")):
            
            fitness[col] = (df00[col] + df01[col] + df10[col] + df11[col])/4

    # Calculate Fitness for each Gate
    for g in gates:

        df_pre = get_on_off_rss(df00, df01, df10, df11, g)
        
        df_pre['m']                 = df_pre['on'] - df_pre['off']
        df_pre['denom']             = df_pre['res'] + delta*(df_pre['off'].abs())
        fitness[g + ' Fitness']     = df_pre['m']/df_pre['denom']
    
    fitness = fitness.reset_index(drop=True)

    return fitness

def vary_currents_by_error(df : pd.DataFrame, M : int, current_col='Current', error_col='Error') -> pd.DataFrame:
    """
    M times vary currents by error to get M sepearate datasets\\
    Append each of these datasets to final dataframe
    """

    dic_tmp = {}

    # Produce M df with variable currents
    for i in range(M):
    
        data        = df.copy()
        data        = data.reset_index(drop=True)
        currents    = data[current_col].values
        errors      = np.abs(data[error_col].values)
        
        currents_norm           = np.random.normal(loc=currents, scale=errors)
        data[current_col]       = currents_norm
        data                    = data.drop(columns=error_col)

        dic_tmp[i] = data

    # Concat dic to df and return
    df_norm = pd.DataFrame(columns=dic_tmp[0].columns)

    for i in range(M):

        df_norm = pd.concat([df_norm, dic_tmp[i]])

    df_norm = df_norm.reset_index(drop=True)

    return df_norm

def abundance(df : pd.DataFrame, gates=['AND Fitness', 'OR Fitness', 'XOR Fitness', 'NAND Fitness', 'NOR Fitness', 'XNOR Fitness'], bins=0, range=(-1,2)) -> pd.DataFrame:
    """
    Empirical Abundace based on pandas dataframe of fitness values for gates provided as list of strings
    """

    df_abundance = pd.DataFrame()
    
    if type(bins) == int: 
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

def display_network(np_network_sim : nanonets.simulation, fig=None, ax=None, blue_color='#348ABD', red_color='#A60628', save_to_path=False, 
                    style_context=['science','bright'], node_size=300, edge_width=1.0, font_size=12, title='', title_size='small',
                    arrows=False, provide_electrode_labels=None, np_numbers=False, height_scale=1, width_scale=1, margins=None):

    colors = np.repeat(blue_color, np_network_sim.N_particles+np_network_sim.N_electrodes)
    
    if np_numbers:
        node_labels = {i : i for i in np_network_sim.G.nodes}
    else:
        node_labels = {i : '' for i in np_network_sim.G.nodes}

    if provide_electrode_labels == None:
        colors[np_network_sim.N_particles:] = red_color
    else:
        colors[np_network_sim.N_particles:] = None
    
        for i, electrode_label in enumerate(provide_electrode_labels):
            node_labels[-1-i] = electrode_label

    with plt.style.context(style_context):

        if fig == None:
            fig = plt.figure()
            fig.set_figheight(fig.get_figheight()*height_scale)
            fig.set_figwidth(fig.get_figwidth()*width_scale)
        if ax == None:
            ax = fig.add_subplot()

        ax.axis('off')
        ax.set_title(title, size=title_size)

        nx.draw_networkx(G=np_network_sim.G, pos=np_network_sim.pos, ax=ax, node_color=colors, arrows=arrows,
                         node_size=node_size, font_size=font_size, width=edge_width, labels=node_labels, clip_on=False, margins=margins)

        if save_to_path != False:

            fig.savefig(save_to_path, bbox_inches='tight', transparent=True)

    return fig, ax

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

def display_network_currents(path : str, row, N_electrodes : int, charge_landscape=False, pos=None, fig=None, ax=None,
                             arrow_scale=2, arrowsize=12, node_size=300, blue_color='#348ABD', red_color='#A60628',
                             position_by_currents=False, display_path=None, edge_vmin=0, edge_vmax=1):

    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot()
        
    ax.axis('off')

    df          = pd.read_csv(path)

    if type(row) != int:
        values  = df.loc[row[0]:row[1],:].mean().values
    else:
        values  = df.loc[row,:].values
    junctions   = np.array([eval(val) for val in df.columns])

    values_new      = []
    junctions_new   = []

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

    values_new = arrow_scale*(values_new - np.min(values_new))/(np.max(values_new) - np.min(values_new))

    G = nx.DiGraph()
    G.add_nodes_from(np.arange(np.min(junctions)-N_electrodes, np.max(junctions)+1-N_electrodes))

    if charge_landscape:

        if type(row) != int:
            states  = pd.read_csv(path.replace("net_currents", "mean_state")).loc[row[0]:row[1],:].mean().values
        else:
            states  = pd.read_csv(path.replace("net_currents", "mean_state")).loc[row,:].values
        colors  = np.repeat(blue_color, len(G.nodes)-N_electrodes)
        colors[np.where(states < 0)] = red_color
        colors  = np.insert(colors, 0, np.repeat(blue_color, N_electrodes))
        states  = np.abs(states)
        states  = node_size*(states - np.min(states))/(np.max(states)-np.min(states))
        states  = np.insert(states, 0, np.repeat(node_size, N_electrodes))
    else:
        states  = np.repeat(node_size, len(G.nodes))
        colors  = np.repeat(blue_color, len(G.nodes))


    for val, junction in zip(values_new, junctions_new):

        G.add_edge(junction[0], junction[1], width=val)

    widths = [G[u][v]['width'] for u, v in G.edges]

    if pos == None:
        if position_by_currents:
            pos = nx.kamada_kawai_layout(G=G, weight='width')
        else:
            pos = nx.kamada_kawai_layout(G=G)
    else:
        keys        = [-i for i in range(1, N_electrodes+1)]
        key_vals    = [pos[i] for i in keys]
        new_keys    = keys[::-1]

        for key in keys:
            pos.pop(key)
        
        for i, key in enumerate(new_keys):
            pos[key] = key_vals[i]

    nx.draw(G=G, pos=pos, ax=ax, edge_color=widths, arrowsize=arrowsize, node_size=states, edge_cmap=plt.cm.Reds, node_color=colors, edge_vmin=edge_vmin, edge_vmax=edge_vmax)

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

def nonlinear_parameter(df : pd.DataFrame, input1_column = 'I1', input2_column = 'I2', current_column='Current', on_state=0.01, off_state=0)->pd.DataFrame:

    currents    = pd.DataFrame()

    df_1 = df[df[input1_column] == off_state]
    df_1 = df_1[df_1[input2_column] == off_state]
    df_1 = df_1.reset_index(drop=True)
    currents['I00'] = df_1[current_column]

    df_1 = df[df[input1_column] == off_state]
    df_1 = df_1[df_1[input2_column] == on_state]
    df_1 = df_1.reset_index(drop=True)
    currents['I01'] = df_1[current_column]

    df_1 = df[df[input1_column] == on_state]
    df_1 = df_1[df_1[input2_column] == off_state]
    df_1 = df_1.reset_index(drop=True)
    currents['I10'] = df_1[current_column]

    df_1 = df[df[input1_column] == on_state]
    df_1 = df_1[df_1[input2_column] == on_state]
    df_1 = df_1.reset_index(drop=True)
    currents['I11'] = df_1[current_column]

    currents    = currents.replace(0, np.nan)
    currents    = currents.dropna()
    currents    = currents.reset_index(drop=True)

    df_new = pd.DataFrame()

    df_new['Iv']    = (currents["I11"] + currents["I10"] + currents["I01"] + currents["I00"])/4
    df_new['Ml']    = (currents["I11"] + currents["I10"] - currents["I01"] - currents["I00"])/4
    df_new['Mr']    = (currents["I11"] - currents["I10"] + currents["I01"] - currents["I00"])/4
    df_new['X']     = (currents["I11"] - currents["I10"] - currents["I01"] + currents["I00"])/4

    return df_new

def expected_value(arr : np.array, order=1, bins=1000)->float:

    count, bins = np.histogram(a=arr, bins=bins)
    probs       = count / np.sum(count)
    mids        = 0.5*(bins[1:]+ bins[:-1])
    exp_val     = np.sum(probs * mids**order)

    return exp_val

def return_ndr(arr : np.array)->np.array:
    return (1 - np.tanh(np.mean(arr)/np.std(arr)))/2

def return_nls(df : pd.DataFrame, ml_col='Ml', mr_col='Mr', x_col='X', bins=1000)->np.array:
    return expected_value(df[x_col].values, order=2, bins=bins)/(expected_value(df[ml_col].values, order=2, bins=bins) + expected_value(df[mr_col].values, order=2, bins=bins))

def fft(signal, dt, n_padded=0, use_hann=True):

    if use_hann:

        bm          = hann(len(signal))
        signal_w    = signal*bm
    else:
        signal_w    = signal.copy()
    
    if n_padded != 0:
        signal_p    = np.pad(signal_w, (0, n_padded - len(signal_w)), 'constant')
    else:
        signal_p    = signal_w.copy()

    n_0         = int(signal.shape[-1]/2)
    signal_fft  = np.fft.fft(signal_p)

    if n_padded == 0:
        freq    = 2 * np.pi * np.fft.fftfreq(signal.shape[-1]) / dt

    else:
        freq    = 2 * np.pi * np.fft.fftfreq(n_padded) / dt

    return freq[:len(freq)//2]*1e-9, np.abs(signal_fft[:len(freq)//2])

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
        class_instance  = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter)
        class_instance.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                                        stat_size=stat_size, save=False, output_potential=True, init=True)
    else:
        class_instance  = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter)
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

def return_res(np_network_sim : nanonets.simulation, time : np.array, voltages : np.array, stat_size=20, I_to_U=1e-4, fit_after_n=0):

    currents = []

    for i in range(stat_size):
        np_network_sim.run_var_voltages(voltages=voltages, time_steps=time, target_electrode=(np_network_sim.N_electrodes-1), save_th=0.1, init=True, eq_steps=0)
        currents.append(np_network_sim.return_output_values()[:,2])
    
    I_mean  = np.mean(currents, axis=0)
    I_std   = np.std(currents, axis=0)/np.sqrt(stat_size)
    res     = np.sum((I_to_U*I_mean[fit_after_n:] - voltages[fit_after_n+1:,0])**2)

    return I_mean[fit_after_n:], I_std[fit_after_n:], res

def metropolis_optimization(np_network_sim : nanonets.simulation, time : np.array, voltages : np.array, n_runs : int,
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

def select_electrode_currents(np_network_sim : nanonets.simulation):

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
    np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, np_info=np_info,
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
    np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, np_info=np_info,
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