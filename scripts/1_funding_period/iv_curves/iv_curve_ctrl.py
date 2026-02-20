import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation

# ─── Configuration ───
N_PROCS         = 10
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/iv_curves/ctrl_sweep/")
SIM_DIC         = {
    "duration"        : True,
    "ac_time"         : 40e-9,
    "error_th"        : 0.05,
    "max_jumps"       : 10000000,
    "n_eq"            : 50,
    "n_per_batch"     : 20,
    "kmc_counting"    : False,
    "min_batches"     : 5}

# Network
L           = 9
N_E         = 8
TOPOLOGY    = {"Nx": L,"Ny": L, "e_pos": [
    [(L-1)//2, 0],[0, 0],[L-1, 0],
    [0, (L-1)//2],[L-1, (L-1)//2],
    [0, L-1],[L-1, L-1],[(L-1)//2, L-1]],
    "electrode_type": ['constant']*N_E}

# Voltage
V_INPUT_MAX     = 0.2
V_CTRL_VALS     = [0.0,0.025,0.05,0.075,0.1]
V_CTRL_POS      = [1,3,5]
N_INPUTS        = 250
VOLTAGE         = np.zeros(shape=(N_INPUTS,N_E+1))
VOLTAGE[:,0]    = np.round(np.linspace(0, V_INPUT_MAX, N_INPUTS),4)

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for pos in V_CTRL_POS:
        for i, V_ctrl in enumerate(V_CTRL_VALS):
            volt        = VOLTAGE.copy()
            volt[:,pos] = V_ctrl
            args        = (volt,TOPOLOGY,PATH)
            kwargs      = {'net_kwargs': {"pack_optimizer":False, "add_to_path":f"_{pos}_{i}"},
                           "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":10}}
            if i != 0:
                tasks.append((args,kwargs))
            else:
                if pos == 1:
                    tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()