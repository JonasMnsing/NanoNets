import numpy as np
import pandas as pd
import multiprocessing

# Extend PATH Variable
import sys
sys.path.append("src/")
import nanonets

def parallel_code(thread, N_voltages):

    network_topology    = "cubic"
    topology_parameter  = {
    "Nx"    : 7,
    "Ny"    : 7,
    "Nz"    : 1,
    "e_pos" : [[0,0,0],[2,0,0],[0,2,0],[4,0,0],[0,4,0],[6,0,0],[0,6,0],[6,2,0],[2,6,0],[6,4,0],[4,6,0],[6,6,0]]
    }

    rs = np.random.RandomState(thread)

    sim_dic = {
        "error_th"    : 0.01,
        "max_jumps"   : 100000,
        "eq_steps"    : 10000
    }

    v_rand          = np.repeat(rs.uniform(low=-0.05, high=0.05, size=((int(N_voltages/4),9))),4,axis=0)
    v_gates         = np.repeat(rs.uniform(low=-0.1, high=0.1, size=int(N_voltages/4)),4)
    i1              = np.tile([0.0,0.0,0.01,0.01], int(N_voltages/4))
    i2              = np.tile([0.0,0.01,0.0,0.01], int(N_voltages/4))

    voltages            = pd.DataFrame(np.zeros((N_voltages,9+4))) 
    voltages.iloc[:,0]  = v_rand[:,0]
    voltages.iloc[:,1]  = i1
    voltages.iloc[:,2]  = i2
    voltages.iloc[:,-1] = v_gates
    voltages.iloc[:,3]  = v_rand[:,1]
    voltages.iloc[:,4]  = v_rand[:,2]
    voltages.iloc[:,5]  = v_rand[:,3]
    voltages.iloc[:,6]  = v_rand[:,4]                
    voltages.iloc[:,7]  = v_rand[:,5]
    voltages.iloc[:,8]  = v_rand[:,6]                           
    voltages.iloc[:,9]  = v_rand[:,7]
    voltages.iloc[:,10] = v_rand[:,8]

    np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter,
                                         folder="scripts/const_voltages/gate_sensitivity/data/")
    np_network_sim.run_const_voltages(voltages=voltages.values, target_electrode=11, save_th=10, sim_dic=sim_dic)

for i in range(10):

    process = multiprocessing.Process(target=parallel_code, args=(i, 2000))
    process.start()