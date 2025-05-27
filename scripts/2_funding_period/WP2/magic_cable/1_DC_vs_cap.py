from pathlib import Path
import logging, numpy as np
import sys
sys.path.append("src/")
import nanonets_utils

# ─── Configuration ────────────────────────────────────────────────────────────────
U_0         = 0.1
T_VAL       = 5.0
STAT_SIZE   = 50
N_VOLT      = 200_000
TIME_STEP   = 1e-10
N_P         = 9
FOLDER      = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# FOLDER   = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
CAP_VALS    = [1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
CPU_CNT     = len(CAP_VALS)
LOG_LEVEL   = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def main():
    """Launch DC input vs capacitance sweeps for 2- and 8-electrode setups."""
    # Setup logging
    logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s: %(message)s")

    # Precompute time array and constant voltage matrix
    time_steps = np.arange(N_VOLT) * TIME_STEP

    # Base output path
    out_base = FOLDER / "potential" / "magic_cable" / "dc_input_vs_cap"
    out_base.mkdir(parents=True, exist_ok=True)

    tasks = []

    # Two-electrode (constant input + floating output)
    topo2 = {
        "Nx": N_P,
        "Ny": N_P,
        "Nz": 1,
        "e_pos": [[(N_P-1)//2,0,0], [(N_P-1)//2, N_P-1,0]],
        "electrode_type": ['constant', 'floating']
    }

    volt2       = np.zeros((N_VOLT, len(topo2['e_pos'])+1), float)
    volt2[:, 0] = U_0

    for cap in CAP_VALS:
        # Prepare args and kwargs
        args = (time_steps, volt2, topo2, out_base, STAT_SIZE, T_VAL)
        kwargs = {'sim_kwargs': {'high_C_output': True,
                                 'np_info2': {'np_index': [81],
                                              'mean_radius': cap,
                                              'std_radius': 0.0},
                                 'add_to_path': f"_{cap}"}}
        tasks.append((args, kwargs))

    # Eight-electrode (constant at 7, floating last)
    topo8 = {
        "Nx": N_P,
        "Ny": N_P,
        "Nz": 1,
        "e_pos": [[(N_P-1)//2,0,0],[0,0,0],[N_P-1,0,0],
                   [0,(N_P-1)//2,0],[N_P-1,(N_P-1)//2,0],
                   [0,N_P-1,0],[N_P-1,N_P-1,0],[(N_P-1)//2,N_P-1,0]],
        "electrode_type": ['constant']*7 + ['floating']
    }
    volt8       = np.zeros((N_VOLT, len(topo8['e_pos'])+1), float)
    volt8[:, 0] = U_0

    for cap in CAP_VALS:
        args    = (time_steps, volt8, topo8, out_base, STAT_SIZE, T_VAL)
        kwargs  = {'sim_kwargs': {'high_C_output': True,
                                 'np_info2': {'np_index': [81],
                                              'mean_radius': cap,
                                              'std_radius': 0.0},
                                 'add_to_path': f"_8cap{cap}"}}
        tasks.append((args, kwargs))

    # Launch all simulations in parallel
    nanonets_utils.batch_launch(nanonets_utils.run_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()