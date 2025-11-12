from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 9
N_PULSES        = 500
DT              = 1e-11
N_VOLTAGES      = 50000
AMPLITUDE_LIST  = [0.01,0.02,0.03,0.04,0.05]
OUTPUT_DIR      = Path("/scratch/j_mens07/data/2_funding_period/dynamic/Noise/white_noise/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 32
TOPOLOGY        = {"Nx": N_NP,"Ny": N_NP,"e_pos": [
    [(N_NP-1)//2, 0], [0, 0], [N_NP-1, 0],
    [0, (N_NP-1)//2],[N_NP-1, (N_NP-1)//2],
    [0, N_NP-1],[N_NP-1, N_NP-1],[(N_NP-1)//2, N_NP-1]],
    "electrode_type": [['constant']*8]}
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(exist_ok=True)
    tasks = []
    for amp in AMPLITUDE_LIST:
        t               = np.arange(N_VOLTAGES)*DT
        voltages        = np.zeros((N_VOLTAGES,len(TOPOLOGY["electrode_type"])+1))
        voltages[0,0]   = amp
        
        for i in range(N_PULSES):
            # output directory per frequency
            args    = (t, voltages, TOPOLOGY, OUTPUT_DIR)
            kwargs  = {
                'net_kwargs': {'add_to_path' : f"_{amp:.3f}_{i}"},
                'sim_kwargs': {'stat_size':1, 'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
