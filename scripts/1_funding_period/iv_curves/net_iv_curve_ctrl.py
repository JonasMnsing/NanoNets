import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation

# ─── Configuration ───
N_PROCS         = 10
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/iv_curves/ctrl_sweep/")
# PATH            = Path("/home/j/j_mens07/phd/data/1_funding_period/iv_curves/ctrl_sqeep/")
SIM_DIC = {
    "n_trajectories" : 400,         # Number of independent KMC runs per voltage
    "sim_time"       : 3e-6,      # Production duration per trajectory [s]
    "eq_time"        : 1.5e-6,      # Equilibration duration per trajectory [s]
    "ac_time"        : 40e-9,       # Base parameter for Numba initialization
    "max_jumps"      : 10000000     # Safety break
}

# Network
N_E     = 8
L_VALS  = [3,5,7,9,11,13,15]

# Voltage
V_INPUT_MAX     = 0.05
V_CTRL_VALS     = [0.0]#,0.005,0.01,0.015,0.02,0.025,0.03]
V_CTRL_POS      = [1,3,5]
N_INPUTS        = 250
VOLTAGE         = np.zeros(shape=(N_INPUTS,N_E+1))
VOLTAGE[:,0]    = np.round(np.linspace(0, V_INPUT_MAX, N_INPUTS),4)

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
        for pos in V_CTRL_POS:
            for i, V_ctrl in enumerate(V_CTRL_VALS):
                volt        = VOLTAGE.copy()
                volt[:,pos] = V_ctrl
                args        = (volt,TOPOLOGY,PATH)
                kwargs      = {'net_kwargs': {"pack_optimizer":False, "add_to_path":f"_{pos}_{i}_equal"},
                                "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":10}}
                if i != 0:
                    tasks.append((args,kwargs))
                else:
                    if pos == 1:
                        tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()