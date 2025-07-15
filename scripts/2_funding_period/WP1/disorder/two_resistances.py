import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import sys
sys.path.append("src/")
import nanonets_utils
import nanonets

# ─── Configuration ────────────────────────────────────────────────────────────────
N_P           = 9
N_J           = 2*N_P*(N_P-1)
P_TYPE_2      = 0.3
N_TYPE_2      = int(P_TYPE_2*N_J)
N_SAMPLES     = 2000
N_PROCESSES   = 10
U_E           = 0.1
U_I           = 0.01
INPUT_POS     = [1, 3]
MEAN_R_VALS   = [30.0,40.0,50.0,75.0,100.0,150.0,200.0,300.0,400.0,500.0]
SIGMA_R       = 0.0
BASE_FOLDER   = Path("/home/j/j_mens07/phd/data/2_funding_period/")
LOG_LEVEL     = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def run_res_disorder(thread: int, voltages: np.ndarray, folder: Union[Path, str], topology: Dict[str, Any], R: float, N: int) -> None:
    
    N_e         = len(topology["e_pos"])
    target      = N_e - 1
    res_info2   = {"R" : R, "N" : N,}

    sim = nanonets.simulation(topology_parameter=topology, folder=str(folder) + "/", res_info2=res_info2, seed=thread,
                                high_C_output=True, add_to_path=f"_R{R}_thread{thread}")
    sim.run_const_voltages(voltages=voltages, target_electrode=target, save_th=20)

    # Save final resistances
    resistances = sim.return_resistances()
    out_csv = Path(folder) / f"resistances_R{R}_thread{thread}.csv"
    np.savetxt(out_csv, resistances)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")

    # Build topology for 8-electrode network (7 constant, 1 floating)
    topology = {
        "Nx": N_P,
        "Ny": N_P,
        "Nz": 1,
        "e_pos": [
            [0,0,0], [(N_P-1)//2,0,0], [N_P-1,0,0],
            [0,(N_P-1)//2,0], [0,N_P-1,0], [N_P-1,(N_P-1)//2,0],
            [(N_P-1)//2,N_P-1,0], [N_P-1,N_P-1,0]
        ],
        "electrode_type": ['constant']*8
    }

    # Output Folder
    base_out = BASE_FOLDER / "current" / "res_disorder" / "two_resistances"
    base_out.mkdir(parents=True, exist_ok=True)

    # Generate voltage samples for logic gate inputs
    voltages = nanonets_utils.logic_gate_sample(U_e=U_E, input_pos=INPUT_POS, N_samples=N_SAMPLES,
                                                topology_parameter=topology, U_i=U_I, U_g=0.0, sample_technique='uniform')
    
    for mean_R in MEAN_R_VALS:

        # Prepare tasks: one per Monte Carlo seed
        tasks = []
        for thread_id in range(N_PROCESSES):
            args = (thread_id, voltages, base_out, topology, mean_R, N_TYPE_2)
            tasks.append((args, {}))

        # Launch simulations in parallel
        nanonets_utils.batch_launch(run_res_disorder, tasks, max_procs=N_PROCESSES)
