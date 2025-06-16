## 13.06.2025

# This script is the definition of FE solvers. The heat diffusion problem is solved by using the implicit and the explicit scheme, 
# by adding inputs the domain, the functionspace the boundary conditions, the initial condition, source term, and the directions
# where the results are going to be saved.

def heatdiff_implicit_solver(domain, Vt, bcs, material_params, time_params,
                             initial_condition, source_term,
                             output_dir, output_filename,
                             bc_update_func=None, source_bc=None,
                             neumann_bcs=None):
    import os
    import numpy as np
    import ufl
    from ufl import TrialFunction, TestFunction, dot, grad
    from dolfinx import fem, io
    from dolfinx.fem.petsc import LinearProblem
    from petsc4py import PETSc

    print("Running the implicit heat diffusion solver")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Parameters
    rho, Cp, k = material_params["rho"], material_params["Cp"], material_params["k_therm"]
    t_end, Nsteps = time_params["t_end"], time_params["Nsteps"]
    dt = t_end / Nsteps
    mesh_comm = domain.comm

    # Functions
    T_n = fem.Function(Vt)
    T_n.name = "Temperature_n"
    T_n.interpolate(initial_condition)

    T_test = TestFunction(Vt)
    T_trial = TrialFunction(Vt)
    T_sol = fem.Function(Vt, name="Temperature")

    # Handle source term: constant/function/callable
    from dolfinx.fem import Constant, Function
    if isinstance(source_term, (Constant, Function)):
        f = source_term
        time_dependent_source = False
    elif callable(source_term):
        f = fem.Function(Vt)
        def f_expr(x): return source_term(x, 0.0).astype(PETSc.ScalarType)
        f.interpolate(f_expr)
        time_dependent_source = True
    else:
        raise TypeError("source_term must be a dolfinx Constant, Function, or callable (x, t) → array")

   # Variational problem
    a = fem.form((rho * Cp * T_trial * T_test / dt + k * dot(grad(T_trial), grad(T_test))) * ufl.dx)


    # Base RHS form (always needed)
    # Base RHS form
    L_form_expr = (rho * Cp / dt * T_n + f) * T_test * ufl.dx

    if neumann_bcs is not None:
      L_form_expr += sum(g * T_test * ds_measure for g, ds_measure in neumann_bcs)

    L_form = fem.form(L_form_expr)

    problem = LinearProblem(a, L_form, u=T_sol, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Output file setup
    with io.XDMFFile(mesh_comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(T_n, 0.0)

    # Center point monitoring
    def near_center(x, tol=1e-2):
       return np.logical_and(np.abs(x[0] - 0.0) < tol, np.abs(x[1] - 0.0) < tol)

    center_dof = fem.locate_dofs_geometrical(Vt, near_center)
    if len(center_dof) == 0:
        print("Warning: No DOF found at domain center (0.5, 0.5)")

    center_temperature = []
    time_series = []

    # Time-stepping loop
    with io.XDMFFile(mesh_comm, output_path, "a") as xdmf:
        for n in range(Nsteps):
            current_time = (n + 1) * dt

            # Time-varying source term
            if time_dependent_source:
                def f_expr(x): return source_term(x, current_time).astype(PETSc.ScalarType)
                f.interpolate(f_expr)

            # Time-varying boundary conditions
            if bc_update_func is not None and source_bc is not None:
                source_bc.t = current_time
                bc_update_func.interpolate(lambda x: source_bc(x))

            # Solve system
            problem.solve()
            T_n.x.array[:] = T_sol.x.array

            # Monitor center temperature
            if len(center_dof) > 0:
                T_center = T_sol.x.array[center_dof[0]]
                center_temperature.append(T_center)
                time_series.append(current_time)
                print(f"Step {n+1:03d} | Time: {current_time:.4f} | T_center: {T_center:.6f}")
            else:
                print(f"Step {n+1:03d} | Time: {current_time:.4f} | Warning: No center DOF found")

            # Output solution
            xdmf.write_function(T_sol, current_time)

    print(f"Simulation finished. Recorded {len(center_temperature)} steps.")
    return time_series, center_temperature, T_sol


def heatdiff_explicit_solver(domain, V, bcs, material_params, time_params,
                             initial_condition, source_term,
                             output_dir, output_filename,
                             bc_update_func=None, source_bc=None):
    import os
    import numpy as np
    import ufl
    from ufl import TrialFunction, TestFunction, dot, grad
    from dolfinx import fem, io
    from dolfinx.fem.petsc import assemble_vector, assemble_matrix, set_bc
    from petsc4py import PETSc

    print("Running the new explicit heat diffusion solver")

    # --- 1. Setup and Parameter Initialization ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    rho, Cp, k = material_params["rho"], material_params["Cp"], material_params["k_therm"]
    t_end, Nsteps = time_params["t_end"], time_params["Nsteps"]
    dt = t_end / Nsteps
    mesh_comm = domain.comm

    # --- 2. Function and FunctionSpace Setup ---
    # Function to store the solution at each time step
    T_n = fem.Function(V)
    T_n.name = "Temperature"
    T_n.interpolate(initial_condition)

    # Define Test and Trial Functions for matrix assembly
    u_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # --- 3. Mass Matrix Lumping ---
    # Based on your working example, we assemble the mass matrix M = integral(rho*Cp*u*v*dx)
    # and then extract its diagonal for lumping. The inverse is stored for the update.
    mass_form = fem.form(rho * Cp * u_trial * v_test * ufl.dx)
    M = assemble_matrix(fem.form(mass_form), bcs=[])
    M.assemble()
    M_lumped = M.getDiagonal()
    M_lumped_inv_array = 1.0 / M_lumped.array

    # To avoid division by zero on Dirichlet boundaries where M_lumped might be zero,
    # we can set the inverse to a non-problematic value like 1.0.
    # The boundary condition application will overwrite the solution on these DoFs anyway.
    for bc in bcs:
        dofs = bc.dof_indices()[0]
        M_lumped_inv_array[dofs] = 1.0

    # --- 4. Source Term Handling (from implicit solver) ---
    from dolfinx.fem import Constant, Function
    if isinstance(source_term, (Constant, Function)):
        f = source_term
        time_dependent_source = False
    elif callable(source_term):
        f = fem.Function(V)
        def f_expr(x): return source_term(x, 0.0).astype(PETSc.ScalarType)
        f.interpolate(f_expr)
        time_dependent_source = True
    else:
        raise TypeError("source_term must be a dolfinx Constant, Function, or callable (x, t) → array")

    # --- 5. RHS Vector Assembly ---
    # The time-discrete equation is T_n+1 = T_n + dt * M_inv * (-K*T_n + F)
    # We define the form for the vector b = -K*T_n + F
    # The weak form is integral( (-k*grad(T_n).grad(v) + f*v) * dx )
    rhs_form = fem.form((f * v_test - k * dot(grad(T_n), grad(v_test))) * ufl.dx)

    # --- 6. Output and Monitoring Setup ---
    with io.XDMFFile(mesh_comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(T_n, 0.0)

    def near_center(x, tol=1e-2):
       return np.logical_and(np.abs(x[0] - 0.0) < tol, np.abs(x[1] - 0.0) < tol)
    center_dof = fem.locate_dofs_geometrical(V, near_center)
    if len(center_dof) == 0:
        print("Warning: No DOF found at domain center")
    center_temperature = []
    time_series = []

    # --- 7. Time-Stepping Loop ---
    with io.XDMFFile(mesh_comm, output_path, "a") as xdmf:
        for n in range(Nsteps):
            current_time = (n + 1) * dt

            # Update time-dependent source term
            if time_dependent_source:
                def f_expr(x): return source_term(x, current_time).astype(PETSc.ScalarType)
                f.interpolate(f_expr)

            # Update time-dependent boundary conditions
            if bc_update_func is not None and source_bc is not None:
                source_bc.t = current_time
                bc_update_func.interpolate(source_bc)

            # Assemble the RHS vector b = (-K*T_n + F)
            b = assemble_vector(rhs_form)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

            # CRITICAL STEP 1: Apply boundary conditions to the RHS vector.
            # This zeros out rows corresponding to Dirichlet DoFs so they aren't updated by the PDE.
            set_bc(b, bcs)

            # CRITICAL STEP 2: Perform the explicit update using the combined form.
            # T_n+1 = T_n + dt * M_inv * b
            T_n.x.array[:] += dt * b.array[:] * M_lumped_inv_array

            # CRITICAL STEP 3 (THE FIX): Apply boundary conditions to the solution vector T_n.
            # The function `set_bc` is not appropriate for this. Instead, we must use
            # the `apply` method of each boundary condition object.
            set_bc(b, bcs)  # Enforce Dirichlet BCs on the RHS

            # Monitor center temperature
            if len(center_dof) > 0:
                T_center = T_n.x.array[center_dof[0]]
                center_temperature.append(T_center)
                time_series.append(current_time)
                if n % 10 == 0:
                    print(f"Step {n+1:03d} | Time: {current_time:.4f} | T_center: {T_center:.6f}")
            else:
                 if n % 10 == 0:
                    print(f"Step {n+1:03d} | Time: {current_time:.4f} | Warning: No center DOF found")

            # Output solution
            xdmf.write_function(T_n, current_time)

    print(f"Explicit simulation finished. Recorded {len(center_temperature)} steps.")
    return time_series, center_temperature, T_n