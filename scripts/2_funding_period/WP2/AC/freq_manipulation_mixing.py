from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 9
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
N_SAMPLES       = 1280
SAMPLE_P_PERIOD = 40
F0              = 1.0
F1              = 3.0
AMPLITUDE       = 0.02
U_BOUNDS        = 0.05
OUTPUT_DIR      = Path("/scratch/j_mens07/data/2_funding_period/dynamic/AC/freq_manipulation_mixing/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 64
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
    control_sample = np.random.uniform(-U_BOUNDS,U_BOUNDS,(N_SAMPLES,5))
    
    # Generate base waveform
    f0_hz       = F0 * 1e6
    f1_hz       = F1 * 1e6
    dt          = 1 / (SAMPLE_P_PERIOD * f1_hz)
    T_sim       = N_PERIODS / f1_hz
    N_steps     = int(np.ceil(T_sim / dt))
    stat_size   = max(STAT_SIZE_BASE, int(np.round(300*F1/5.0)))


    time_steps, base_volt = sinusoidal_voltages(
        N_steps, topo,
        amplitudes=[0.,AMPLITUDE,AMPLITUDE,0.,0.,0.,0.,0.],
        frequencies=[0.,f0_hz,f1_hz,0.,0.,0.,0.,0.],
        time_step=dt
    )
    
    tasks = []
    for i, controls in enumerate(control_sample):
        # make a fresh copy of the base waveform
        volt_i = base_volt.copy()
        # apply 5‐element control vector to the correct electrodes
        volt_i[:, [0,3,4,5,6]] = controls

        args    = (time_steps, volt_i, topo, OUTPUT_DIR)
        kwargs  = {
            'net_kwargs': {'add_to_path' : f"_{i}"},
            'sim_kwargs': {'stat_size' : stat_size, 'save':True}
        }
        tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()