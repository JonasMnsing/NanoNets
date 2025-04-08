import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=False)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=293.0)

if __name__ == '__main__':

    # Global
    N_voltages  = 40000
    time_step   = 1e-10
    stat_size   = 200
    time_steps  = np.arange(N_voltages)*time_step
    N_p         = 9
    volt        = np.zeros(shape=(N_voltages,9))
    volt[:,0]   = 1.0
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
    if topology_parameter["electrode_type"][-1] == 'floating':
        # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/wo_magic_cable/dc_input_vs_network/"
        folder      = "/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/dc_input_vs_network/293/"
    else:
        # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/current/wo_magic_cable/dc_input_vs_network/"
        folder      = "/home/j/j_mens07/phd/data/2_funding_period/current/wo_magic_cable/dc_input_vs_network/293/"
        
    run_simulation(time_steps, volt, topology_parameter, folder, stat_size)