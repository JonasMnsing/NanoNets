import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation, distribute_array_across_processes

# ─── Configuration ───
N_PROCS     = 10
LOG_LEVEL   = logging.INFO
PATH        = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/iv_curves/network/")
SIM_DIC     = {
    "n_trajectories" : 100,
    "sim_time"       : 1e-6,
    "eq_time"        : 2e-6,
    "ac_time"        : 40e-9,
    "max_jumps"      : 4000,
    "max_eq_jumps"   : 6000
}

# Network
N_E     = 8
L_VALS  = [3,5,7,9,11,13,15]

# Voltage
V_INPUT_MAX     = 0.1
N_INPUTS        = 300
VOLTAGE         = np.zeros(shape=(N_INPUTS,N_E+1))
VOLTAGE[:,0]    = np.linspace(-V_INPUT_MAX, V_INPUT_MAX, N_INPUTS)

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for L in L_VALS:
        TOPOLOGY = {"Nx": L,"Ny": L, "e_pos": [
            [(L-1)//2, 0],[0, 0],[L-1, 0],
            [0, (L-1)//2],[L-1, (L-1)//2],
            [0, L-1],[L-1, L-1],[(L-1)//2, L-1]],
            "electrode_type": ['constant']*N_E}
        for p in range(N_PROCS):
            volt        = VOLTAGE.copy()
            volt_p      = distribute_array_across_processes(p, volt.copy(), N_PROCS)
            args        = (volt_p,TOPOLOGY,PATH)
            kwargs      = {'net_kwargs': {"pack_optimizer":False,"add_to_path":"_zero"},
                            "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":10}}
            tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()