# first line: 1
@memory.cache
def get_all_data(freq_list):

    dts, N_steps, time = {}, {}, {}
    y_t_u, y_d_u, y_t_u_e = {}, {}, {}

    slice_start = N_SKIP * SAMPLE_P_PERIOD

    for freq in freq_list:
        # --- 1. Constants & Timing ---
        dt = 1 / (40 * freq * 1e6)
        n_s = int(np.ceil((N_PERIODS / (freq * 1e6)) / dt))

        dts[freq] = dt
        N_steps[freq] = n_s
        time[freq] = dt * np.arange(n_s)

        # --- 2. Load Main Data (Selective Column Loading) ---
        # Over SSHFS, loading only the columns you need is MUCH faster
        path_u = f"{PATH_U}Nx=9_Ny=9_Ne=8_{freq:.3f}_0.0195.parquet"
        df_u = pd.read_parquet(path_u, columns=['Observable', 'Error'])
        
        y_t_u[freq] = df_u['Observable'].values[slice_start:].copy()
        y_t_u_e[freq] = df_u['Error'].values[slice_start:].copy()
        
        # --- 3. Load Mean State (Large File) ---
        path_mean = f"{PATH_U}mean_state_Nx=9_Ny=9_Ne=8_{freq:.3f}_0.0195.parquet"
        
        # Optimization: If the parquet is row-group partitioned, 
        # reading iloc[slice:] still pulls the whole file. 
        # If the file is massive, consider reading only the values needed.
        df_mean = pd.read_parquet(path_mean)
        mean_slice = df_mean.iloc[slice_start:, 8:].values
        
        y_d_u[freq] = get_displacement_currents(mean_slice, C_US, dt)
    
    print(f"Done loading {freq} MHz...")

    return dts, N_steps, time, y_t_u, y_d_u, y_t_u_e
