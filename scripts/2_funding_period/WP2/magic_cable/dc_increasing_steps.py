import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size):

    np_info2    = {
        'np_index'      : [81], 
        'mean_radius'   : 5e3,
        'std_radius'    : 0.0
    }
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=True, np_info2=np_info2)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    N_voltages          = 200000
    time_step           = 1e-9
    stat_size           = 50
    time_steps          = np.arange(N_voltages)*time_step
    N_p                 = 9
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
                            [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],
                            [0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','constant','constant','constant',
                            'constant','constant','constant','floating']
    }
    folder                  = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/magic_cable/dc_increasing_steps/"
    U_0                     = 0.1
    steps_per_step          = 40000
    U_i1                    = U_0*np.repeat(np.arange(6),steps_per_step)
    U_i2                    = U_0*np.repeat(np.arange(4,-1,-1),steps_per_step)
    U_i                     = np.hstack((U_i1,U_i2))
    N_voltages              = len(U_i)
    volt                    = np.zeros(shape=(len(U_i),9))
    volt[:,0]               = U_i
    time_step               = 1e-9
    stat_size               = 50
    time_steps              = np.arange(N_voltages)*time_step
    
    run_simulation(time_steps, volt, topology_parameter, folder, stat_size)