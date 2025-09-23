import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets import Simulation
from nanonets.utils import logic_gate_sample, distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_PARTICLES     = 9
N_E             = 8
V_CONTROL       = [0.01,0.02,0.03,0.04,0.05]
V_INPUT         = 0.01
V_GATE          = 0.0
N_DATA          = 19200
N_PROCS         = 48
LOG_LEVEL       = logging.INFO
PATH            = Path("/scratch/j_mens07/data/1_funding_period/controls/range/")
INPUT_POS       = [1,3]
E_POS_REF       = 1
SAVE_TH         = 100
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)

    topo = {"Nx": N_PARTICLES, "Ny": N_PARTICLES,
                "e_pos" : [[0,0],
                           [int((N_PARTICLES-1)/2),0],
                           [N_PARTICLES-1,0],
                           [0,int((N_PARTICLES-1)/2)],
                           [0,N_PARTICLES-1],
                           [N_PARTICLES-1,int((N_PARTICLES)/2)],
                           [int((N_PARTICLES)/2),(N_PARTICLES-1)],
                           [N_PARTICLES-1,N_PARTICLES-1]],
                "electrode_type" : ['constant']*N_E}
    
    tasks = []
    for V in V_CONTROL:
        
        volt = logic_gate_sample(V, INPUT_POS, N_DATA, topo, V_INPUT,
                                 V_GATE, sample_technique='uniform')
        for p in range(N_PROCS):
            volt_p  = distribute_array_across_processes(p, volt, N_PROCS)
            args    = (volt_p,topo,PATH)
            kwargs  = {"net_kwargs":{"add_to_path":f"_{V}","pack_optimizer":False},
                       "sim_kwargs":{"save_th":SAVE_TH}}
            tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()