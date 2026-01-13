import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation

# ─── Configuration ───
N_PROCS         = 1
V_INPUT_MAX     = 0.5
V_GATE_MAX      = 0.5
N_INPUTS        = 960
N_GATES         = 960
V_INPUT         = np.round(np.linspace(-V_INPUT_MAX, V_INPUT_MAX, N_INPUTS),4)
V_GATES         = [0.0] #np.round(np.linspace(-V_GATE_MAX, V_GATE_MAX, N_INPUTS),4)
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/nanonets/scripts/1_funding_period/iv_curves/set/data/")
topo            = {"Nx": 1,"Ny": 1, "electrode_type": ['constant','constant']}
sim_dic         = {"duration" : True, "ac_time" : 40e-9, "error_th" : 0.05, "max_jumps" : 10000000, "n_eq" : 50, "n_per_batch" : 20, "kmc_counting" : False, "min_batches" : 5}
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for V_g in V_GATES:
        volt        = np.zeros((N_INPUTS,3))
        volt[:,0]   = V_INPUT
        volt[:,-1]  = V_g
        args        = (volt,topo,PATH)
        kwargs      = {'net_kwargs': {'add_to_path' : f"_{V_g:.3f}_{sim_dic['duration']}","pack_optimizer":False},"sim_kwargs":{"sim_dic":sim_dic,"save_th":10}}
        tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()