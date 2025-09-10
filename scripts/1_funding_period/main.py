import logging
from pathlib import Path
from nanonets.utils import logic_gate_sample, distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_PARTICLES     = 9
N_ELECTRODES    = 8
V_CONTROL       = 0.002
V_INPUT         = 0.004
V_GATE          = 0.0
N_DATA          = 5000
N_PROCS         = 10
T_VAL           = 5
LOG_LEVEL       = logging.INFO
PATH            = Path("/home/j/j_mens07/phd/NanoNets/scripts/1_funding_period/")
INPUT_POS       = [1,3]
SAVE_TH         = 10
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    topo = {"Nx": N_PARTICLES, "Ny": N_PARTICLES,
            "e_pos" : [[0,0], [int((N_PARTICLES-1)/2),0], [N_PARTICLES-1,0],
                       [0,int((N_PARTICLES-1)/2)], [0,N_PARTICLES-1],
                       [N_PARTICLES-1,int((N_PARTICLES)/2)],
                       [int((N_PARTICLES)/2),(N_PARTICLES-1)], [N_PARTICLES-1,N_PARTICLES-1]],
            "electrode_type" : ['constant']*N_ELECTRODES}
    volt = logic_gate_sample(V_CONTROL, INPUT_POS, N_DATA, topo, V_INPUT,
                                V_GATE, sample_technique='uniform')
    for p in range(N_PROCS):
        volt_p  = distribute_array_across_processes(p, volt, N_PROCS)
        args    = (volt_p,topo,PATH)
        kwargs  = {"net_kwargs":{"pack_optimizer":False},
                    "sim_kwargs":{"save_th":SAVE_TH}}
        tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()