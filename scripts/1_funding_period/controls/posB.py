import logging
import numpy as np
from pathlib import Path

from nanonets import Simulation
from nanonets.utils import logic_gate_sample, distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_PARTICLES = 9
N_CONTROLS  = [1,3,5,7,9,11,13]
V_CONTROL   = 0.1
V_INPUT     = 0.01
V_GATE      = 0.0
N_DATA      = 20000
N_PROCS     = 10
T_VAL       = 5
LOG_LEVEL   = logging.INFO
# PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/system_size_scaled/")
PATH        = Path("/scratch/j_mens07/data/1_funding_period/system_size/system_size_scaled/")
INPUT_POS   = [0,1]
SAVE_TH     = 10
# --------------------- 

def get_transfer_coeff(pos):
    topo = {"Nx":N_PARTICLES,"Ny":N_PARTICLES}
    topo["e_pos"] = pos
    topo["electrode_type"] = len(pos)*['constant']
    sim_class = Simulation(topology_parameter=topo, pack_optimizer=False)
    sim_class.build_conductance_matrix()
    sim_class.init_transfer_coeffs()
    return sim_class.get_transfer_coeffs()

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for N_c in N_CONTROLS:
        topo = {"Nx":N_PARTICLES,"Ny":N_PARTICLES}
        pos  = [[8,4],[4,8]]
        if N_c >= 1:
            pos.append([0,0])
        if N_c >= 3:
            pos.append([2,0])    
            pos.append([0,2])
        if N_c >= 5:
            pos.append([4,0])    
            pos.append([0,4])
        if N_c >= 7:
            pos.append([6,0])    
            pos.append([0,6])    
        if N_c >= 9:
            pos.append([8,0])    
            pos.append([0,8])    
        if N_c >= 11:
            pos.append([8,2])    
            pos.append([2,8])    
        if N_c >= 13:    
            pos.append([8,6])    
            pos.append([6,8])
        pos.append([8,8])
        N_e = len(pos)
        topo["e_pos"] = pos
        topo["electrode_type"] = N_e*['constant']
        
        volt = logic_gate_sample(V_CONTROL, INPUT_POS, N_DATA, topo, V_INPUT,
                                 V_GATE, sample_technique='uniform')
        t_coeff = get_transfer_coeff(pos)
        scale   = np.divide(t_coeff[1], t_coeff, where=t_coeff!=0)
        volt    *= np.hstack((scale,0.0))
        for p in range(N_PROCS):
            volt_p  = distribute_array_across_processes(p, volt, N_PROCS)
            args    = (volt_p,topo,PATH)
            kwargs  = {"net_kwargs":{"pack_optimizer":False},
                       "sim_kwargs":{"save_th":SAVE_TH}}
            tasks.append((args,kwargs))
    
    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()