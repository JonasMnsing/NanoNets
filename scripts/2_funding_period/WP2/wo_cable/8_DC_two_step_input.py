from pathlib import Path
import logging
import numpy as np
import sys
sys.path.append("src/")
import nanonets_utils

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL                = 5.0
STAT_SIZE            = 100
N_VOLTAGES           = 10000
TIME_STEP            = 1e-10
N_NP                 = 9
U_0                  = 0.1
STEPS_PER_STEP       = 1500
STEPS_BETWEEN_LIST   = [0,2,4,6,8,10,12,14,16,18,20,30,40,50,100,200,400,600,800,1000,2000,4000]
BASE_DIR             = Path("/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/dc_two_step_input")
LOG_LEVEL            = logging.INFO
CPU_CNT              = 10
# ────────────────────────────────────────────────────────────────────────────────


def main():
    # Configure logging
    logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s: %(message)s")

    # Common topology: 2-electrode (constant input + floating output)
    topo = {
        "Nx": N_NP,
        "Ny": N_NP,
        "Nz": 1,
        "e_pos": [[(N_NP-1)//2, 0, 0], [(N_NP-1)//2, N_NP-1, 0]],
        "electrode_type": ['constant', 'floating']
    }

    # Precompute time_steps array
    time_steps = np.arange(N_VOLTAGES) * TIME_STEP

    # Ensure base output directory exists
    out_base = BASE_DIR
    out_base.mkdir(parents=True, exist_ok=True)

    # Build tasks for each gap
    tasks = []
    for gap in STEPS_BETWEEN_LIST:
        # Construct two-step voltage waveform
        volt = np.zeros((N_VOLTAGES, len(topo['e_pos'])+1),float)
        volt[:STEPS_PER_STEP, 0] = U_0
        start2 = STEPS_PER_STEP + gap
        volt[start2:start2 + STEPS_PER_STEP, 0] = U_0

        # Prepare simulation arguments and optional sim_kwargs
        args    = (time_steps, volt, topo, out_base, STAT_SIZE, T_VAL)
        kwargs  = {
            'sim_kwargs': {'high_C_output'  :   False,
                            'add_to_path'   :   f"_{gap}"}
        }
        tasks.append((args, kwargs))

    # Launch batch simulations
    nanonets_utils.batch_launch(nanonets_utils.run_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
