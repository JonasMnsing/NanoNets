import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import nanonets
import multiprocessing

if __name__ == '__main__':

    N_processes         = 1
    
    for i in [1,1.2,1.4,1.6,1.8,2.0]:

        voltages            = pd.read_csv("7_7_and_values.csv", header=0, index_col=0)
        voltages            = pd.concat([voltages]*4, ignore_index=True).iloc[[0,10,20,30]].reset_index(drop=True).iloc[:,:8]
        voltages['I1']      = [0.0,0.0,0.01,0.01]
        voltages['I2']      = [0.0,0.01,0.0,0.01]
        voltages['O']       = 0
        voltages            = voltages[['C1','I1','C2','I2','C3','C4','C5','O','G']]
        voltages            = pd.concat([voltages]*10, ignore_index=True)
        voltages[['C1','I1','C2','I2','C3','C4','C5']] = voltages[['C1','I1','C2','I2','C3','C4','C5']]*i
        # print(pd.concat([voltages]*10, ignore_index=True))

        def parallel_code(thread, voltages):

            # Topology values
            topology_parameter          = {}
            N                           = 7
            topology_parameter["Nx"]    = N
            topology_parameter["Ny"]    = N
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N)/2),0], [int((N)/2),(N-1),0], [N-1,N-1,0]]

            # Simulation Values
            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 10000000

            # Misc
            target_electrode    = len(topology_parameter["e_pos"]) - 1
            folder              = "test_runs/"

            # Run Simulation
            sim_class = nanonets.simulation(voltages=voltages.values)
            sim_class.init_cubic(folder=folder, topology_parameter=topology_parameter)
            sim_class.run_const_voltages(target_electrode=target_electrode, T_val=0, sim_dic=sim_dic, save_th=1)

        parallel_code(0, voltages)
        # for i in range(N_processes):

        #     process = multiprocessing.Process(target=parallel_code, args=(i,voltages))
        #     process.start()