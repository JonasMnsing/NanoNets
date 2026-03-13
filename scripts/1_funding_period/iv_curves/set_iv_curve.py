import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation, distribute_array_across_processes

# ─── Configuration ───
N_PROCS         = 32
V_INPUT_MAX     = 0.1
V_GATE_MAX      = 1.0
N_INPUTS        = 320
N_GATES         = 320
V_INPUT         = np.linspace(-V_INPUT_MAX, V_INPUT_MAX, N_INPUTS, endpoint=False)
V_GATES         = np.linspace(-V_GATE_MAX, V_GATE_MAX, N_INPUTS, endpoint=False)
LOG_LEVEL       = logging.INFO
PATH            = Path("/home/j/j_mens07/phd/data/1_funding_period/iv_curves/iv_curves_self_cap/")
topo            = {"Nx": 1,"Ny": 1, "electrode_type": ['constant','constant']}
SIM_DIC         = {
    "n_trajectories" : 100,
    "sim_time"       : 1e-6,
    "eq_time"        : 2e-6,
    "ac_time"        : 40e-9,
    "max_jumps"      : 4000,
    "max_eq_jumps"   : 6000
}

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for V_g in V_GATES:
        volt        = np.zeros((N_INPUTS,3))
        volt[:,0]   = V_INPUT
        volt[:,-1]  = V_g
        for p in range(N_PROCS):
            volt_p      = distribute_array_across_processes(p, volt.copy(), N_PROCS)
            args        = (volt_p,topo,PATH)
            kwargs      = {'net_kwargs': {"pack_optimizer":False},
                        "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":10}}
            tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()