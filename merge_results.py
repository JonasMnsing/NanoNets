import sys
sys.path.append("src/")
import nanonets_utils

N_stat          = 100
N_th            = 10
Nx, Ny, Nz, Ne  = 7, 7, 1, 2

# for r in range(1,11):
    # for R in [50,100,200,400,800,1600,3200,6400,12800]:
for R in [25,30,35,40,45]:#100,200,400,800,1600,3200,6400,12800]:

    # folder  = f'/home/jonas/phd/data/1I_1O_R/R_{R}/r{r}'
    folder  = f'/home/jonas/phd/NanoNets/scripts/2_funding_period/WP2/step_input/1I_1O_R_dis/data/blocked/R_{R}'
    
    nanonets_utils.store_average_time_results(folder, Nx, Ny, Nz, Ne, N_stat, N_th)
    nanonets_utils.store_average_time_currents(folder, Nx, Ny, Nz, Ne, N_stat, N_th)
    nanonets_utils.store_average_time_states(folder, Nx, Ny, Nz, Ne, N_stat, N_th)

    # print(f"R_{R} / r_{r} check")