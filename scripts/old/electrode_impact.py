import sys
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src")
import numpy as np
from joblib import Parallel, delayed
import model

topology_parameter      = {}
topology_parameter[0]   = {
    "Nx"    :   5,
    "Ny"    :   5,
    "Nz"    :   1,
    "e_pos" :   [[0,0,0],[1,0,0],[2,0,0],[3,0,0],[4,0,0],[0,1,0],[0,2,0],[0,3,0],[0,4,0],
                [4,1,0],[4,2,0],[4,3,0],[1,4,0],[2,4,0],[3,4,0],[4,4,0]]
}

topology_parameter[1]    = {
    "Nx"    :   9,
    "Ny"    :   9,
    "Nz"    :   1,
    "e_pos" :   [[0,0,0],[2,0,0],[4,0,0],[6,0,0],[8,0,0],[0,2,0],[0,4,0],[0,6,0],[0,8,0],
                [8,2,0],[8,4,0],[8,6,0],[2,8,0],[4,8,0],[6,8,0],[8,8,0]]
}

setup               = 0
target_electrode    = len(topology_parameter[setup]["e_pos"]) - 1
min_voltage         = -0.05
max_voltage         = 0.05
n_voltages          = 501
voltage_sweep       = np.linspace(min_voltage,max_voltage,n_voltages)
# voltages            = np.random.uniform(low=min_voltage, high=max_voltage, size=(n_voltages,len(topology_parameter[setup]["e_pos"])))
voltages            = np.zeros(shape=(n_voltages,len(topology_parameter[setup]["e_pos"])))
voltages            = np.hstack([voltages,np.atleast_2d(np.zeros(n_voltages)).T])
# folder              = f"../data/electrode_impact/setup{setup}/"
folder              = f"../data/electrode_impact/zeros/"

def parallel_code(i):
    
    voltages[:,i] = voltage_sweep 
    model.cubic_net_simulation(target_electrode, topology_parameter[setup], voltages, folder, 100, f"_E{i}")

Parallel(n_jobs=8)(delayed(parallel_code)(i) for i in range(len(topology_parameter[setup]["e_pos"])))
