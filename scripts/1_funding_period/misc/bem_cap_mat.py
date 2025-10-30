import bempp.api
import numpy as np
import gmsh
import sys

# --- 1. Define Physical and Grid Parameters ---

R = 10e-9  # Sphere radius (10 nm) in meters
GAP = 1e-9   # Gap between spheres (1 nm) in meters
N_SIDE = 5   # 5x5 grid
N_SPHERES = N_SIDE * N_SIDE

# Center-to-center distance for nearest neighbors
D_CENTER = 2 * R + GAP  # 21 nm

# Mesh size. We need to resolve the small gap.
# A smaller value gives a more accurate result but is much slower.
# 0.5nm (5e-10) is a reasonable starting point.
MESH_SIZE = 0.5e-9  # 0.5 nm in meters

def create_geometry_and_mesh():
    """
    Uses gmsh to create and mesh the 5x5 sphere array.
    Each sphere surface is tagged with a physical ID from 1 to 25.
    """
    print(f"Initializing gmsh to create {N_SPHERES} spheres...")
    gmsh.initialize()
    gmsh.model.add("5x5_sphere_array")

    # Center the array at (0, 0, 0)
    x_offset = (N_SIDE - 1) * D_CENTER / 2.0
    y_offset = (N_SIDE - 1) * D_CENTER / 2.0
    
    sphere_tags = []
    
    try:
        for i in range(N_SIDE):  # x-direction
            for j in range(N_SIDE):  # y-direction
                
                # Calculate sphere index (0 to 24)
                k = i * N_SIDE + j
                
                # gmsh physical tags must be 1-based
                domain_tag = k + 1 
                
                x = i * D_CENTER - x_offset
                y = j * D_CENTER - y_offset
                z = 0
                
                # Create sphere volume
                sphere_vol_tag = gmsh.model.occ.addSphere(x, y, z, R)
                sphere_tags.append(sphere_vol_tag)

        # Synchronize model before getting boundaries
        gmsh.model.occ.synchronize()

        # Create physical groups for the SURFACES
        # This is how bempp will identify each conductor
        for i in range(N_SPHERES):
            k = i
            domain_tag = k + 1 # 1-based tag
            
            # Get the surface of the volume
            # getBoundary returns [(dim, tag), ...]
            surfaces = gmsh.model.getBoundary([(3, sphere_tags[i])], combined=False)
            if not surfaces:
                raise RuntimeError(f"Could not find boundary for sphere {k}")
                
            surface_tag = surfaces[0][1]
            
            # Add a physical group for this surface
            gmsh.model.addPhysicalGroup(2, [surface_tag], domain_tag)
            gmsh.model.setPhysicalName(2, domain_tag, f"sphere_{domain_tag}")

        # Set mesh size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", MESH_SIZE)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", MESH_SIZE)
        
        # Generate 2D surface mesh
        print("Starting mesh generation... (This may take a moment)")
        gmsh.model.mesh.generate(2)
        
        # Import directly into bempp
        print("Importing mesh into BEMPP...")
        grid = bempp.api.import_grid_from_gmsh(gmsh.model)
        
    except Exception as e:
        print(f"An error occurred during gmsh operation: {e}")
        gmsh.finalize()
        sys.exit(1)
        
    gmsh.finalize()
    
    print(f"Mesh generated with {grid.number_of_elements} elements.")
    return grid

def solve_for_column(l, dp0_space, p1_space, slp_op):
    """
    Solves the BEM system for the l-th column of the C-matrix.
    Sets V_l = 1 and V_k = 0 for all k != l.
    Returns a vector of total charges [Q_0, Q_1, ..., Q_24].
    """
    
    # gmsh tags are 1-based
    l_tag = l + 1 
    
    # 1. Define the Dirichlet boundary condition (Potential V)
    # We use a Python function that bempp can evaluate.
    # It checks the domain_index (our 1-based tag) for each element.
    @bempp.api.real_callable
    def dirichlet_fun(x, y, z, n, domain_index, res):
        res[0] = 1.0 if domain_index == l_tag else 0.0

    # Create a GridFunction object from our Python function
    dirichlet_grid_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_fun)

    # 2. Assemble the BEM system
    # We are solving the weak form: <test, SLP * sigma> = <test, V>
    # A = slp_op.weak_form() (Already precomputed)
    # b = dirichlet_grid_fun.projections(dp0_space)
    
    print(f"  Assembling RHS for column {l} (V_{l}=1)...")
    rhs = dirichlet_grid_fun.projections(dp0_space)

    # 3. Solve the linear system
    # We use GMRES to find the charge density 'sigma'
    print(f"  Solving BEM system for column {l}...")
    from bempp.api.linalg import gmres
    # Set use_strong_form=True for faster assembly.
    # This solves A * x = V directly, where A is the discretized SLP.
    # This requires an identity operator to map 'sigma' to the RHS.
    # Let's stick to the weak form, it's more standard.
    
    # Re-check: For V = SLP(sigma), the standard formulation is:
    # A = slp_op.weak_form()
    # b = dirichlet_grid_fun.projections(dp0_space)
    # x, info = gmres(A, b)
    # sigma = bempp.api.GridFunction(dp0_space, coefficients=x)
    # This is correct.
    
    # Ah, but we can use the strong form if we solve V = SLP(sigma)
    # with a different solver.
    # Let's use the simple, pre-assembled operator.
    # We need to solve slp_op * sigma = dirichlet_grid_fun
    
    # Create the BEM problem
    identity = bempp.api.operators.boundary.sparse.identity(
        dp0_space, p1_space, dp0_space)
        
    lhs_op = slp_op
    rhs_vec = identity * dirichlet_grid_fun
    
    # Using 'use_strong_form=True' is often more stable and direct
    # for this type of problem.
    x, info = gmres(slp_op, dirichlet_grid_fun, use_strong_form=True, 
                    restart=500, tol=1E-5)
    
    if info != 0:
        print(f"  WARNING: GMRES did not converge for column {l} (info={info}).")
        
    # The solution x is the coefficients of the charge density sigma
    sigma = bempp.api.GridFunction(dp0_space, coefficients=x)

    # 4. Calculate total charge on *each* sphere
    # Q_k = integral(sigma) over surface of sphere k
    
    charges = np.zeros(N_SPHERES)
    
    grid = dp0_space.grid
    domain_indices = grid.domain_indices
    integration_elements = grid.integration_elements
    coefficients = sigma.coefficients

    for k in range(N_SPHERES):
        k_tag = k + 1  # 1-based tag
        
        # Find all mesh elements belonging to sphere k
        mask = (domain_indices == k_tag)
        
        if np.any(mask):
            # Q_k = sum(sigma_i * area_i) for all elements i on sphere k
            charge_k = np.sum(coefficients[mask] * integration_elements[mask])
            charges[k] = charge_k
        else:
            print(f"  WARNING: No elements found for sphere {k} (tag {k_tag}).")

    return charges

def main():
    """
    Main execution function.
    """
    # 1. Create the mesh
    grid = create_geometry_and_mesh()

    # 2. Set up BEM function spaces
    # dp0_space: Piecewise constant space for charge density (sigma)
    # p1_space:  Piecewise linear space for potential (V)
    dp0_space = bempp.api.function_space(grid, "DP", 0)
    p1_space = bempp.api.function_space(grid, "P", 1)
    
    print(f"BEM spaces created. N_DOFs (charge): {dp0_space.global_dof_count}, N_DOFs (potential): {p1_space.global_dof_count}")

    # 3. Assemble the main BEM operator (Single Layer Potential)
    # This is the (V = SLP * sigma) operator.
    # This is computationally expensive and is done only once.
    # We use the 'p1_space' for the range, as V is continuous.
    print("Assembling BEM operator (SLP)... (This is the slowest part)")
    
    # We need the operator mapping sigma (dp0) to V (p1).
    # The operator itself: domain=dp0, range=p1, dual=dp0
    # bempp.api.operators.boundary.laplace.single_layer(
    #   domain_space, range_space, dual_to_range_space)
    slp_op = bempp.api.operators.boundary.laplace.single_layer(
        dp0_space, p1_space, dp0_space)
    print("Operator assembly complete.")
    
    # Initialize the capacitance matrix
    # This is C_vacuum
    C_vacuum = np.zeros((N_SPHERES, N_SPHERES))

    # 4. Loop over all 25 spheres to find each column
    for l in range(N_SPHERES):
        print(f"--- Processing Column {l}/{N_SPHERES-1} ---")
        
        # Solve for V_l = 1
        charge_vector_l = solve_for_column(l, dp0_space, p1_space, slp_op)
        
        # This vector is the l-th column of the C-matrix
        # C_kl = Q_k when V_l = 1
        C_vacuum[:, l] = charge_vector_l
        
        print(f"  Diagonal element C({l},{l}) = {C_vacuum[l,l]:.4e} F")
        if l > 0:
            print(f"  Off-diag element C({l},{l-1}) = {C_vacuum[l,l-1]:.4e} F")

    # 5. Final Result
    print("\n--- Calculation Complete ---")
    
    # The calculated matrix is C_vacuum (capacitance in vacuum, eps_r = 1)
    # The physical capacitance C = C_vacuum * (epsilon / epsilon_0)
    
    print("Capacitance Matrix (in Farads) for a vacuum (epsilon_r = 1):")
    np.set_printoptions(precision=3, suppress=True)
    print(C_vacuum)
    
    # Save the matrix to a file
    np.save("capacitance_matrix_vacuum.npy", C_vacuum)
    np.savetxt("capacitance_matrix_vacuum.txt", C_vacuum, fmt="%.5e")

    print("\n-------------------------------------------------------------")
    print("IMPORTANT:")
    print("The matrix above (and in the .txt/.npy files) is C_vacuum.")
    print("To get your final physical capacitance matrix, you MUST multiply")
    print("this entire matrix by the RELATIVE PERMITTIVITY (epsilon_r)")
    print("of your insulating material.")
    print("\n  C_physical = C_vacuum * (epsilon_r)")
    print("  where epsilon_r = (your_epsilon) / (8.854e-12 F/m)")
    print("-------------------------------------------------------------")
    
    # Example: Print capacitance of an isolated sphere for comparison
    eps_0 = bempp.api.global_parameters.physical_constants.epsilon_0
    C_isolated = 4 * np.pi * eps_0 * R
    print(f"\nFor comparison, C for one isolated sphere: {C_isolated:.4e} F")
    print(f"Calculated C_00 (corner):                 {C_vacuum[0,0]:.4e} F")
    print(f"Calculated C_12,12 (center):              {C_vacuum[12,12]:.4e} F")
    print(f"Note: Diagonal C_kk values are > C_isolated due to")
    print("      shielding from the other 24 grounded spheres.")


if __name__ == "__main__":
    main()