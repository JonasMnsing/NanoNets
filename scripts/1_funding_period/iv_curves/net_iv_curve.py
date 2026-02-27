import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation, distribute_array_across_processes

# ─── Configuration ───
N_PROCS     = 10
LOG_LEVEL   = logging.INFO
PATH        = Path("/home/j/j_mens07/phd/data/1_funding_period/iv_curves/iv_curves_self_cap/")
SIM_DIC     = {
    "n_trajectories" : 400,
    "sim_time"       : 1e-7,
    "eq_time"        : 2e-7,
    "ac_time"        : 40e-9,
    "max_jumps"      : 10000000
}
# SIM_DIC = {
#     "n_trajectories" : 400,         # Number of independent KMC runs per voltage
#     "sim_time"       : 3e-6,      # Production duration per trajectory [s]
#     "eq_time"        : 1.5e-6,      # Equilibration duration per trajectory [s]
#     "ac_time"        : 40e-9,       # Base parameter for Numba initialization
#     "max_jumps"      : 10000000     # Safety break
# }

# Network
N_E     = 8
L_VALS  = [3,5,7,9,11]

# Voltage
V_INPUT_MAX     = 0.1
V_GATE_MAX      = 0.5
N_INPUTS        = 320
V_GATES         = [0.0] #np.round(np.linspace(-V_GATE_MAX, V_GATE_MAX, N_INPUTS),4)
VOLTAGE         = np.zeros(shape=(N_INPUTS,N_E+1))
VOLTAGE[:,0]    = np.round(np.linspace(-V_INPUT_MAX, V_INPUT_MAX, N_INPUTS),4)

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
        for V_g in V_GATES:
            for p in range(N_PROCS):
                volt        = VOLTAGE.copy()
                volt_p      = distribute_array_across_processes(p, volt.copy(), N_PROCS)
                args        = (volt_p,TOPOLOGY,PATH)
                kwargs      = {'net_kwargs': {"pack_optimizer":False, "add_to_path":"_small"},
                                "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":10}}
                tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()