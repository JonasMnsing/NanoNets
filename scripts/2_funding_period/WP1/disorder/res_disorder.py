import numpy as np
import sys

# Add to path
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import nanonets_utils
import multiprocessing

# Simulation Function
def parallel_code(thread, voltages, folder, topology_parameter, res_info, sigma_R):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, res_info=res_info,
                                              seed=thread, add_to_path=f"_{sigma_R}_{thread}")
    sim_class.run_const_voltages(voltages=voltages, target_electrode=target_electrode, save_th=20)
    resistances  = sim_class.model.resistances
    np.savetxt(f"{folder}resistances_{sigma_R}_{thread}.csv", resistances)

if __name__ == '__main__':

    # N_p x N_p Values for Network Size 
    N_p = 9

    # Topology Parameter
    topology_parameter          = {
        "Nx"                :   N_p,
        "Ny"                :   N_p,
        "Nz"                :   1,
        "e_pos"             :   [[0,0,0], [int((N_p-1)/2),0,0], [N_p-1,0,0], 
                                [0,int((N_p-1)/2),0], [0,N_p-1,0], [N_p-1,int((N_p)/2),0],
                                [int((N_p)/2),(N_p-1),0], [N_p-1,N_p-1,0]],
        "electrode_type"    :   ['constant','constant','constant','constant','constant','constant','constant','floating']
    }

    if topology_parameter["electrode_type"][-1] == "constant":
        folder  = "/home/j/j_mens07/phd/data/1_funding_period/current/res_disorder/"
    else:
        folder  = "/home/j/j_mens07/phd/data/1_funding_period/potential/res_disorder/"

    # Number of voltages and CPU processes
    N_voltages  = 20000 #80640
    N_processes = 10 #36

    # Voltage values
    U_e         = 0.1
    voltages    = nanonets_utils.logic_gate_sample(U_e=U_e, input_pos=[1,3], N_samples=N_voltages, topology_parameter=topology_parameter)

    # Nanoparticle Information
    sigma_R = 5.0
    res_info = {
        "mean_R"    : 25.0,     # Average resistance
        "std_R"     : sigma_R,  # Standard deviation of resistances
        "dynamic"   : False     # Dynamic or constant resistances
    }
   
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, voltages, folder, topology_parameter, res_info, sigma_R))
        process.start()