import logging
from pathlib import Path
import numpy as np
import sys
sys.path.append("src/")
import nanonets_utils

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL           = 5.0
STAT_SIZE_BASE  = 10
N_PERIODS       = 40
N_NP            = 9
FREQ_LIST_MHZ   = [133.,105.,86.,68.,55.,44.,36.,28.,23.,18.,15.,12.,10.,8.,6.,5.,2.,1.,
                   0.5,0.25,0.12,0.06,0.03,0.01,0.008,0.006,0.004,0.002,0.001]
AMPLITUDE       = 0.1
OUTPUT_DIR      = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# OUTPUT_DIR      = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 10
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s: %(message)s")
    tasks   = []

    # Define two floating-output electrode networks: 2‐electrode and 8‐electrode
    topo_list = []
    # 2‐electrode: one constant + floating
    topo_list.append({"Nx": N_NP,"Ny": N_NP,"Nz": 1,
                      "e_pos": [[(N_NP-1)//2, 0, 0],[(N_NP-1)//2, N_NP-1, 0]],
                      "electrode_type": ['constant', 'floating']})
    # 8‐electrode: 7 constant + floating
    topo_list.append({"Nx": N_NP,"Ny": N_NP,"Nz": 1,
                    "e_pos": [
                        [(N_NP-1)//2, 0, 0],
                        [0, 0, 0],
                        [N_NP-1, 0, 0],
                        [0, (N_NP-1)//2, 0],
                        [N_NP-1, (N_NP-1)//2, 0],
                        [0, N_NP-1, 0],
                        [N_NP-1, N_NP-1, 0],
                        [(N_NP-1)//2, N_NP-1, 0]
                    ],
                    "electrode_type": ['constant']*7 + ['floating']})

    # single output base folder
    out_base = OUTPUT_DIR / "potential" / "wo_magic_cable" / "ac_input_vs_freq"
    out_base.mkdir(parents=True, exist_ok=True)

    # build tasks for each topology and frequency
    for topo in topo_list:
        n_elec = len(topo["e_pos"])
        for freq_mhz in FREQ_LIST_MHZ:
            f0_hz       = freq_mhz * 1e6
            dt          = 1 / (40 * f0_hz)
            T_sim       = N_PERIODS / f0_hz
            N_steps     = int(np.ceil(T_sim / dt))
            stat_size   = max(STAT_SIZE_BASE, int(np.round(300 * freq_mhz / 5.0)))

            time_steps, volt = nanonets_utils.sinusoidal_voltages(
                N_steps,
                topo,
                amplitudes=[AMPLITUDE]+(n_elec-1)*[0.0],
                frequencies=[f0_hz]+(n_elec-1)*[0.0],
                time_step=dt
            )

            # output directory per frequency
            out_base.mkdir(exist_ok=True)
            args    = (time_steps, volt, topo, out_base, stat_size, T_VAL)
            kwargs  = {
                'sim_kwargs': {'high_C_output'  :   False,
                               'add_to_path'    :   f"_{freq_mhz:.3f}"}
            }
            tasks.append((args, kwargs))

    nanonets_utils.batch_launch(nanonets_utils.run_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()