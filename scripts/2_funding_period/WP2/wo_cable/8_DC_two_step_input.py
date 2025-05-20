import logging
import multiprocessing
from pathlib import Path

import numpy as np
from core import run_simulation, batch_launch, build_voltage_matrix

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL                = 5.0
STAT_SIZE            = 100
N_VOLTAGES           = 10000
TIME_STEP            = 1e-10
N_NP                 = 9
U_0                  = 0.1
STEPS_PER_STEP       = 1500
STEPS_BETWEEN_LIST   = [0,40,80,160,200,400,500,1000,2000,4000]
STEPS_BETWEEN_LIST   = [2, 4, 6, 8, 12, 14, 16, 18, 25, 30]
STEPS_BETWEEN_LIST   = [2, 4, 6, 8, 12, 14, 16, 18, 25, 30]
OUTPUT_DIR           = Path("/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/dc_two_step_input/")
LOG_LEVEL            = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

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

    tasks = []
    for gap in STEPS_BETWEEN_LIST:
        # build two-step waveform: step1, gap, step2
        volt = np.zeros((N_VOLTAGES, len(topo['e_pos'])))
        volt[:STEPS_PER_STEP, 0] = U_0
        start2 = STEPS_PER_STEP + gap
        volt[start2:start2 + STEPS_PER_STEP, 0] = U_0

        out_folder = OUTPUT_DIR / f"gap_{gap}"
        out_folder.mkdir(parents=True, exist_ok=True)

        tasks.append((
            time_steps,
            volt,
            topo,
            out_folder,
            STAT_SIZE,
            T_VAL
        ))

    cpu_cnt = multiprocessing.cpu_count()
    logging.info(f"Launching {len(tasks)} two-step simulations on up to {cpu_cnt} cores…")
    batch_launch(run_simulation, tasks, max_procs=cpu_cnt)
    logging.info("All two-step simulations complete.")


if __name__ == "__main__":
    main()
