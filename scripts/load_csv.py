import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import model
import multiprocessing

if __name__ == '__main__':

    N_processes         = 10
    voltages            = pd.read_csv("scripts/voltage_configs.csv", header=None)
    print(voltages)

    def parallel_code(thread, voltages):

        # Topology values
        topology_parameter          = {}
        topology_parameter["Nx"]    = 5
        topology_parameter["Ny"]    = 5
        topology_parameter["Nz"]    = 1
        topology_parameter["e_pos"] = [[0,0,0], [5-1,0,0], [0,5-1,0], [5-1,5-1,0]]

        # Simulation Values
        sim_dic                 = {}
        sim_dic['error_th']     = 0.05
        sim_dic['max_jumps']    = 10000000

        # Misc
        target_electrode    = len(topology_parameter["e_pos"]) - 1
        folder              = f"/mnt/c/Users/jonas/Desktop/phd/test_run/error/run2/{thread}_"

        # Run Simulation
        sim_class = model.simulation(voltages=voltages.values)
        sim_class.init_cubic(folder=folder, topology_parameter=topology_parameter)
        sim_class.run_const_voltages(target_electrode=target_electrode, T_val=0.28, sim_dic=sim_dic, save_th=1)

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,voltages))
        process.start()