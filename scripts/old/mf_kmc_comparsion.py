import sys
sys.path.append("/home/jonas/phd/NanoNets/src")
# sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tunneling
import time

voltages = pd.read_csv("scripts/voltage_configs.csv", sep=' ', header=None).values

N_threads       = 10
N_voltages      = voltages.shape[0]
index           = [i for i in range(N_voltages)]
rows            = [index[i::N_threads] for i in range(N_threads)]

def parallel_code(thread):
        
    thread_rows = rows[thread]

    for N in range(2, 11):

        topology_parameter = {
            "Nx"    :   N,
            "Ny"    :   N,
            "Nz"    :   1,
            "e_pos" :   [[0,0,0],[N-1,0,0],[0,N-1,0],[N-1,N-1,0]]
        }

        target_electrode    = len(topology_parameter["e_pos"]) - 1
        folder              = "../data/mf_kmc_comparsion/"

        t0  = time.time()
        tunneling.cubic_net_simulation(target_electrode, topology_parameter, voltages[thread_rows,:], folder, 9)
        t1  = time.time()
        t10 = t1-t0

        with open("times.csv","a") as f:
            np.savetxt(f, np.array([N, t10]).reshape(1,-1), delimiter=',')

Parallel(n_jobs=10)(delayed(parallel_code)(i) for i in range(10))
