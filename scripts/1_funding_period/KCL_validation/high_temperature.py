import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_PARTICLES     = [3,9,15]
N_E             = 8
V_RANGE         = 0.1
N_DATA          = 1000
N_PROCS         = 10
T_VAL           = 293
VOLTAGES        = np.zeros((N_DATA,N_E+1))
VOLTAGES[:,:-2] = np.random.uniform(-V_RANGE,V_RANGE,(N_DATA,N_E-1))
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/KCL_validation/")
SIM_DIC         = {
    "duration"        : True,
    "ac_time"         : 40e-9,
    "error_th"        : 0.05,
    "max_jumps"       : 10000000,
    "n_eq"            : 50,
    "n_per_batch"     : 20,
    "kmc_counting"    : False,
    "min_batches"     : 5}
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for n in N_PARTICLES:
        topo = {"Nx": n, "Ny": n,
                "e_pos" : [[0,0], 
                           [int((n-1)/2),0],
                           [n-1,0], 
                           [0,int((n-1)/2)],
                           [0,n-1],
                           [n-1,int((n)/2)],
                           [int((n)/2),(n-1)],
                           [n-1,n-1]],
                "electrode_type" : ['constant']*N_E}
        for p in range(N_PROCS):
            volt_p  = distribute_array_across_processes(p, VOLTAGES.copy(), N_PROCS)
            args    = (volt_p,topo,PATH)
            kwargs  = {"net_kwargs":{"pack_optimizer":False},
                       "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":5,"T_val":T_VAL}}
            tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()