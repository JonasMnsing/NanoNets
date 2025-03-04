# Libraries
import numpy as np
import sys
sys.path.append("src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils
import multiprocessing

def parallel_code(thread : int, N : int, V_arr : np.array, path : str, N_processes: int):

    topology_parameter = {
        "Nx"                : N,
        "Ny"                : N,
        "Nz"                : 1,
        "e_pos"             : [[0,0,0],[(N-1)//2,0,0],[N-1,0,0],
                               [0,(N-1)//2,0],[N-1,(N-1)//2,0],
                               [0,N-1,0],[(N-1)//2,N-1,0],[N-1,N-1,0]],
        "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','constant']
    }

    sim_dic =   {
        "error_th"        : 0.05,
        "max_jumps"       : 10000000,
        "eq_steps"        : 100000,
        "jumps_per_batch" : 5000,
        "kmc_counting"    : False,
        "min_batches"     : 5
    }

    np_info = {
        "eps_r"         : 2.6,
        "eps_s"         : 3.9,
        "mean_radius"   : 10.0,
        "std_radius"    : 0.0,
        "np_distance"   : 1.0
    }
    
    voltages            = nanonets_utils.distribute_array_across_processes(process=thread, data=V_arr, N_processes=N_processes)
    np_network_cubic    = nanonets.simulation(topology_parameter=topology_parameter, folder=path, np_info=np_info)
    np_network_cubic.run_const_voltages(voltages=voltages, target_electrode=7, save_th=10, sim_dic=sim_dic)

if __name__ == '__main__':

    N_min, N_max    = 3, 16
    V_min, V_max    = 0.0, 0.2
    N_data          = 2000
    N_processes     = 10
    folder          = "/home/j/j_mens07/phd/data/1_funding_period/current/magnitude_scaled/"
    input_pos       = [1,3]
    alphas          = [0.49, 0.56, 0.67, 0.73, 0.86, 0.92, 1., 1.07, 1.17, 1.16, 1.24, 1.28, 1.35, 1.41]

    for i, N in enumerate(range(N_min, N_max+1)):

        V_arr                   = np.zeros((N_data,9))
        V_arr[:,input_pos[0]]   = alphas[i]*np.linspace(V_min, V_max, N_data, endpoint=False)
        V_arr[:,input_pos[1]]   = alphas[i]*np.linspace(V_min, V_max, N_data, endpoint=False)

        procs = []
        for i in range(N_processes):
            process = multiprocessing.Process(target=parallel_code, args=(i, N, V_arr, folder, N_processes))
            process.start()
            procs.append(process)
        
        for p in procs:
            p.join()