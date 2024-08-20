import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import networkx as nx
import nanonets
import scienceplots
from typing import Union, Tuple, List
from scipy.signal.windows import hann

blue_color  = '#348ABD'
red_color   = '#A60628'

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

def load_time_params(folder : str):

    txt_file    = open(folder+'params.txt', 'r')
    lines       = txt_file.readlines()

    params = []

    for line in lines:
        
        if ((line[0] != '#') and (line[0] != '\n')):

            params.append(line[:-1])

    N_processes         = eval(params[0])
    network_topology    = params[1]

    if network_topology == 'cubic':
        topology_parameter  = {
            "Nx"    :   eval(params[2][0]),
            "Ny"    :   eval(params[2][2]),
            "Nz"    :   eval(params[2][4]),
            "e_pos" :   eval(params[3])
        }
    else:
        topology_parameter  = {
            "Np"    :   eval(params[2][0]),
            "Nj"    :   eval(params[2][2]),
            "e_pos" :   eval(params[3])
        }

    eq_steps = eval(params[4])

    np_info = {
        "eps_r"         : eval(params[5]),
        "eps_s"         : eval(params[6]),
        "mean_radius"   : eval(params[7]),
        "std_radius"    : eval(params[8]),
        "np_distance"   : eval(params[9])
    }

    res_info = {
        "mean_R"    : eval(params[10]),
        "std_R"     : eval(params[11])    
    }

    T_val       = eval(params[12])
    save_th     = eval(params[13])

    return N_processes, network_topology, topology_parameter, eq_steps, np_info, res_info, T_val, save_th

def load_params(folder : str):

    txt_file    = open(folder+'params.txt', 'r')
    lines       = txt_file.readlines()

    params = []

    for line in lines:
        
        if ((line[0] != '#') and (line[0] != '\n')):

            params.append(line[:-1])

    N_processes         = eval(params[0])
    network_topology    = params[1]

    if network_topology == 'cubic':
        topology_parameter  = {
            "Nx"    :   eval(params[2][0]),
            "Ny"    :   eval(params[2][2]),
            "Nz"    :   eval(params[2][4]),
            "e_pos" :   eval(params[3])
        }
    else:
        topology_parameter  = {
            "Np"    :   eval(params[2][0]),
            "Nj"    :   eval(params[2][2]),
            "e_pos" :   eval(params[3])
        }

    sim_dic = {
        'error_th'  :   eval(params[4]),
        'max_jumps' :   eval(params[5]),
        'eq_steps'  :   eval(params[6])
    }

    np_info = {
        "eps_r"         : eval(params[7]),
        "eps_s"         : eval(params[8]),
        "mean_radius"   : eval(params[9]),
        "std_radius"    : eval(params[10]),
        "np_distance"   : eval(params[11])
    }

    res_info = {
        "mean_R"    : eval(params[12]),
        "std_R"     : eval(params[13])
    }

    T_val       = eval(params[14])
    save_th     = eval(params[15])

    return N_processes, network_topology, topology_parameter, sim_dic, np_info, res_info, T_val, save_th

def get_boolean_data(folder : str, N : Union[int, list], N_e : Union[int, list], boot_steps=0, i1_col=1, i2_col=3, o_col=7,
                    min_currents=0.0, min_error=0.0, max_error=np.inf, dic=None, dic_nc=None, off_state=[0.0], on_state=[0.01], disordered=False)->Tuple[dict,dict]:

    # For variable numbers of nanoparticles
    if (type(N) == list and type(N_e) == int):

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
    elif (type(N) == int and type(N_e) == list):
        
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

def prepare_for_fitness_calculation(df : pd.DataFrame, N_c : int, min_current=None, input1_col='I1', input2_col='I2',
    gate_col='G', control_col='C', current_col='Current', off_state=0.0, on_state=0.01) -> pd.DataFrame:
    """
    Prepares a pandas Dataframe of electric currents for calculation of gate fitnesses\\
    Number of Controls NC must be provided as argument\\
    Allows to exclude any currents having an absolute value less than min_current if min_current != None\\
    Allows to check df if for each voltage combination exactly all possible input states are present\\ 
    Other Attributes define column names for various features
    """

    # Sort Dataframe by C1,C2,..G,I1,I2 and exclude Currents if min_current != None
    sort_cols = [control_col + '{}'.format(i) for i in range(1, N_c + 1)]
    sort_cols.extend([gate_col, input1_col, input2_col])
    
    data    = df.copy()
    
    if min_current != None:
        # data.loc[data[current_col].abs() <= min_current, current_col] = min_current
        data = data[data[current_col].abs() > min_current]
    
    data    = data.sort_values(by=sort_cols)
    data    = data.reset_index(drop=True)

    # Move through each row and proof that for each electrode voltage combination all possible input states are present
    i       = 0
    rows    = np.floor(len(data))

    while (i < rows):

        try:

            cond1 = ((data[input1_col][i]   == off_state) and (data[input2_col][i]   == off_state))
            cond2 = ((data[input1_col][i+1] == off_state) and (data[input2_col][i+1] == on_state))
            cond3 = ((data[input1_col][i+2] == on_state)  and (data[input2_col][i+2] == off_state))
            cond4 = ((data[input1_col][i+3] == on_state)  and (data[input2_col][i+3] == on_state))
            
            cond5 = (data[gate_col][i] == data[gate_col][i+1]) 
            cond6 = (data[gate_col][i] == data[gate_col][i+2]) 
            cond7 = (data[gate_col][i] == data[gate_col][i+3])

            cond8 = (data[control_col + '1'][i] == data[control_col + '1'][i+1]) 
            cond9 = (data[control_col + '1'][i] == data[control_col + '1'][i+2]) 
            cond10= (data[control_col + '1'][i] == data[control_col + '1'][i+3])
        
        except:

            try:
                data = data.drop(i)
            except:
                pass

            try:
                data = data.drop(i+1)
            except:
                pass

            try:
                data = data.drop(i+2)
            except:
                pass

            data = data.reset_index(drop=True)
            break

        if not(cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond7 and cond8 and cond9 and cond10):

            data    = data.drop(i)
            data    = data.reset_index(drop=True)
            rows    = rows - 1
            continue

        i = i + 4
    
    return data

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
                             arrow_scale=2, arrowsize=12, node_size=300, blue_color='#348ABD', red_color='#A60628', position_by_currents=False):

    if fig == None:
        fig = plt.figure()
    if ax == None:
        ax = fig.add_subplot()
        
    ax.axis('off')

    df          = pd.read_csv(path)
    values      = df.loc[row,:].values
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

    nx.draw(G=G, pos=pos, ax=ax, edge_color=widths, arrowsize=arrowsize, node_size=states, edge_cmap=plt.cm.Reds, node_color=colors)

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