from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 9
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
N_SAMPLES       = 100
SAMPLE_P_PERIOD = 40
F0S             = [28.0]
AMPLITUDE       = 0.02
U_BOUNDS        = 0.05
OUTPUT_DIR      = Path("/scratch/j_mens07/data/2_funding_period/dynamic/AC/res_disorder_freq_manipulation/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 32
NET_INDEX       = [12,3,4]
N_J_TOTAL       = 2*N_NP*(N_NP-1)
RES_VALUES      = [100.0,1600.0]
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Setup base directory once
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topo    = {"Nx": N_NP,"Ny": N_NP,
               "e_pos": [[(N_NP-1)//2, 0],
                         [0, 0],
                         [N_NP-1, 0],
                         [0, (N_NP-1)//2],
                         [N_NP-1, (N_NP-1)//2],
                         [0, N_NP-1],
                         [N_NP-1, N_NP-1],
                         [(N_NP-1)//2, N_NP-1]],
                "electrode_type": ['constant']*8}

    # Control Voltage Sample
    control_sweep = np.linspace(-U_BOUNDS,U_BOUNDS,N_SAMPLES)

    tasks = []
    for F0 in F0S:
        # Generate base waveform
        f0_hz       = F0 * 1e6
        dt          = 1 / (SAMPLE_P_PERIOD * f0_hz)
        T_sim       = N_PERIODS / f0_hz
        N_steps     = int(np.ceil(T_sim / dt))
        stat_size   = max(STAT_SIZE_BASE, int(np.round(300*F0/5.0)))

        time_steps, base_volt = sinusoidal_voltages(
            N_steps, topo,
            amplitudes=[AMPLITUDE,0.,0.,0.,0.,0.,0.,0.],
            frequencies=[f0_hz,0.,0.,0.,0.,0.,0.,0.],
            time_step=dt
        )

        for R_val in RES_VALUES:
            for i, controls in enumerate(control_sweep):
                # make a fresh copy of the base waveform
                volt_i = base_volt.copy()
                # apply 5‐element control vector to the correct electrodes
                volt_i[:, 3] = controls

                for j in NET_INDEX:
                    res_info2 = {'N':N_J_TOTAL//3,'mean_R':R_val,'std_R':0.0}
                    args    = (time_steps, volt_i, topo, OUTPUT_DIR)
                    kwargs  = {
                        'net_kwargs': {'add_to_path' : f"_{i}_mean2_{R_val}_{j}_{F0:.3f}",
                                       "res_info2":res_info2, "seed":j},
                        'sim_kwargs': {'stat_size' : stat_size, 'save':True}
                    }
                    tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
