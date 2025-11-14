import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import logic_gate_sample, distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_PROCS         = 32
V_INPUT_MAX     = 0.5
V_GATE_MAX      = 0.5
N_INPUTS        = 960
N_GATES         = 960
V_INPUT         = np.round(np.linspace(-V_INPUT_MAX, V_INPUT_MAX, N_INPUTS),4)
V_GATES         = np.round(np.linspace(-V_GATE_MAX, V_GATE_MAX, N_INPUTS),4)
LOG_LEVEL       = logging.INFO
PATH            = Path("/scratch/j_mens07/data/1_funding_period/iv_curves/set/")
topo            = {"Nx": 1,"Ny": 1, "electrode_type": ['constant','constant']}
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
        kwargs      = {'net_kwargs': {'add_to_path' : f"_{V_g:.3f}","pack_optimizer":False},"sim_kwargs":{"save_th":10}}
        tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()