import numpy as np
import logging
from pathlib import Path

# Nanonets
from nanonets.utils import distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_PARTICLES     = 9
N_ELECTRODES    = 8
V_RANGE         = 0.1
V_GATE          = 0.0
V_INPUT         = 0.03
N_DATA          = 3600
N_PROCS         = 36
T_VAL           = 5
LOG_LEVEL       = logging.INFO
PATH            = Path("/scratch/j_mens07/data/1_funding_period/iv_curves/control_impact/")
INPUT_POS       = 1
CONTROL_POS     = [0,2,5]
SAVE_TH         = 10
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    topo = {"Nx": N_PARTICLES, "Ny": N_PARTICLES,
            "e_pos" : [[0,0], 
                        [int((N_PARTICLES-1)/2),0],
                        [N_PARTICLES-1,0], 
                        [0,int((N_PARTICLES-1)/2)],
                        [0,N_PARTICLES-1],
                        [N_PARTICLES-1,int((N_PARTICLES)/2)],
                        [int((N_PARTICLES)/2),(N_PARTICLES-1)],
                        [N_PARTICLES-1,N_PARTICLES-1]],
            "electrode_type" : ['constant']*N_ELECTRODES}
    
    for pos in CONTROL_POS:
        volt                = np.zeros(shape=(N_DATA,N_ELECTRODES+1))
        volt[:,INPUT_POS]   = V_INPUT
        volt[:,pos]         = np.linspace(-V_RANGE,V_RANGE,N_DATA)
        for p in range(N_PROCS):
            volt_p  = distribute_array_across_processes(p, volt.copy(), N_PROCS)
            args    = (volt_p,topo,PATH)
            kwargs  = {"net_kwargs":{"pack_optimizer":False,"add_to_path":f"_{pos}"},
                       "sim_kwargs":{"save_th":SAVE_TH}}
            tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()