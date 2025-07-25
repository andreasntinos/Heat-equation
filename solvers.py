## 13.06.2025
# This script is the definition of FE heat diffusion solvers. The heat diffusion problem is solved by using the implicit and the explicit scheme, 
# by adding inputs the domain, the functionspace the boundary conditions, the initial condition, source term, and the directions
# where the results are going to be saved.

import os 
import numpy as np # linear algebra and numerical operations
import ufl # finite element library for defining variational forms
from ufl import TrialFunction, TestFunction, dot, grad
from dolfinx import fem, io # finite element library for defining function spaces, boundary conditions, and solving problems
from dolfinx.fem.petsc import LinearProblem 
from petsc4py import PETSc # PETSc library for solving linear algebra problems
from ufl import dx
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, set_bc # used for the variational formulation in the explicit scheme
from dolfinx.fem import Constant, Function, locate_dofs_geometrical, set_bc 
import time

#======================================
#           IMPLICIT SCHEME           #
#======================================

def heatdiff_implicit_solver(domain, Vt, bcs, material_params, time_params,
                             initial_condition, source_term,
                             output_dir, output_filename,
                             bc_update_func=None, source_bc=None,
                             neumann_bcs=None):
    

    # Create the output directory if it does not already exist
    os.makedirs(output_dir, exist_ok=True) # create output directory if it doesn't exist
    output_path = os.path.join(output_dir, output_filename) # the path where the xdmf files are saved

    # Extract material parameters from the dictionary
    rho, Cp, k = material_params["rho"], material_params["Cp"], material_params["k_therm"]
    # Extract time parameters from the dictionary
    t_end, Nsteps = time_params["t_end"], time_params["Nsteps"]
    # Calculate the time step size
    dt = t_end / Nsteps # time step size
    print(f"Time step used:{dt:.4f} seconds")
    # Get the MPI communicator associated with the mesh
    mesh_comm = domain.comm # communicator for the mesh

    # Functions for the finite element problem
    T_n = fem.Function(Vt) # function to store the solution
    T_n.name = "Temperature_n" # Function to store the previous time step solution
    T_n.interpolate(initial_condition)

    T_test = TestFunction(Vt)  # Define the test function (v)
    T_trial = TrialFunction(Vt)  # Define the trial function (T)
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
    a = ((rho * Cp * T_trial * T_test / dt + k * dot(grad(T_trial), grad(T_test))) * ufl.dx)
    
    # Define the base right-hand side (RHS) of the variational form (L(T_test))
    # This represents the known terms from the previous time step and the source
    L_form_expr = (rho * Cp / dt * T_n + f) * T_test * ufl.dx

    neumann_funcs = []

    # Handle Neumann boundary conditions if provided
    if neumann_bcs is not None:
       for g, ds_measure in neumann_bcs:
           if isinstance(g, (fem.Constant, fem.Function)):
              L_form_expr += g * T_test * ds_measure
              neumann_funcs.append((None, g))  # static
           elif callable(g):
              g_fun = fem.Function(Vt)
              def g_expr(x, g_=g): return g_(x, 0.0).astype(PETSc.ScalarType)
              g_fun.interpolate(g_expr)
              L_form_expr += g_fun * T_test * ds_measure
              neumann_funcs.append((g, g_fun))  # time-dependent
           else:
            raise TypeError("Neumann BC 'g' must be Constant, Function, or callable.")

    L_form = (L_form_expr)

    # Create a DOLFINx LinearProblem object
    # This sets up the system Ax = b, where A is from 'a', b is from 'L_form', x is 'T_sol', and 'bcs' are applied
    problem = LinearProblem(a, L_form, u=T_sol, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Output file setup
    with io.XDMFFile(mesh_comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(T_n, 0.0)

    # Define a probe point to monitor temperature during the simulation
    def probe_point(x, tol=1e-8):
        return (
           (np.abs(x[0] - 0.004) < tol) &
           (np.abs(x[1] - 0.0025) < tol) 
           
    )

    def probe_point(x, tol=1e-8):
        return (
           (np.abs(x[0] - 0.0) < tol) &
           (np.abs(x[1] - 0.0) < tol) 
           
    )

    # Locate the degrees of freedom (DoFs) that are geometrically close to the probe point
    top_dof = fem.locate_dofs_geometrical(Vt, probe_point)
    if len(top_dof) == 0:
        print("Warning: No DOF found at domain center (0.5, 0.5)")

    top_temperatures = []
    time_series = []

    start_time = time.time()

    # Time-stepping loop
    with io.XDMFFile(mesh_comm, output_path, "a") as xdmf:
        for n in range(Nsteps):
            current_time = (n + 1) * dt   # Calculate the current simulation time

             # Update time-varying internal source term if applicable
            if time_dependent_source:
                def f_expr(x): return source_term(x, current_time).astype(PETSc.ScalarType)
                f.interpolate(f_expr)

            # Update Neumann BCs
            # Update Neumann BCs each step
            for g_callable, g_fun in neumann_funcs:
               if g_callable is not None:
                def g_expr(x, g_=g_callable): 
                    return g_(x, current_time).astype(PETSc.ScalarType)
                g_fun.interpolate(g_expr)

            # Time-varying boundary conditions
            if bc_update_func is not None and source_bc is not None:
                source_bc.t = current_time
                bc_update_func.interpolate(lambda x: source_bc(x))

            # Solve system
            problem.solve()
            T_n.x.array[:] = T_sol.x.array

            # Monitor top temperature
            if len(top_dof) > 0: #
                T_top = T_sol.x.array[top_dof[0]]
                top_temperatures.append(T_top)
                time_series.append(current_time)
                print(f"Step {n+1:03d} | Time: {current_time:.4f} | T_top: {T_top:.6f}")
            else:
                print(f"Step {n+1:03d} | Time: {current_time:.4f} | Warning: No center DOF found")

            # Output solution
            xdmf.write_function(T_sol, current_time)

    print(f"Simulation finished. Recorded {len(top_temperatures)} steps.")
    elapsed_time = time.time() - start_time   
    print(f"\n Simulation finished in {elapsed_time:.2f} seconds.")
    return time_series, top_temperatures, T_sol


#======================================
#           EXPLICIT SCHEME           #
#======================================

def heatdiff_explicit_solver(domain, Vt, bcs, material_params, time_params,
                             initial_condition, source_term,
                             output_dir, output_filename,
                             bc_update_func=None, source_bc=None,
                             neumann_bcs=None):
    # Parameters
    rho, Cp, k = material_params["rho"], material_params["Cp"], material_params["k_therm"]
    t_end, Nsteps = time_params["t_end"], time_params["Nsteps"]
    dt = t_end / Nsteps
    mesh_comm = domain.comm
    print(f"Time step used: {dt:.7f} seconds")

    # Initial condition
    T_n = Function(Vt)
    T_n.name = "Temperature"
    T_n.interpolate(initial_condition)

    T_test = TestFunction(Vt)

    # Mass lumping
    one = fem.Function(Vt)
    one.interpolate(lambda x: np.ones_like(x[0]))
    mass_form_lump = fem.form(rho * Cp * one * T_test * dx)
    M_lump_vec = assemble_vector(mass_form_lump)
    M_lump_vec.assemble()
    M_lumped_inv_array = 1.0 / M_lump_vec.array

    # Time-dependent source setup
    from collections import deque
    time_dependent_source = False
    if isinstance(source_term, (Constant, Function)):
        f = source_term
    elif callable(source_term):
        f = Function(Vt)
        time_dependent_source = True
        def f_expr(x): return source_term(x, 0.0).astype(PETSc.ScalarType)
        f.interpolate(f_expr)
    else:
        raise TypeError("source_term must be Constant, Function, or callable")

    # Handle Neumann BCs
    neumann_funcs = []
    L_form_expr = -k * dot(grad(T_n), grad(T_test)) * dx + f * T_test * dx

    if neumann_bcs is not None:
        for g, ds_measure in neumann_bcs:
            if isinstance(g, (Constant, Function)):
                L_form_expr += g * T_test * ds_measure
                neumann_funcs.append((None, g))  # static
            elif callable(g):
                g_fun = Function(Vt)
                def g_expr(x, g_=g): return g_(x, 0.0).astype(PETSc.ScalarType)
                g_fun.interpolate(g_expr)
                L_form_expr += g_fun * T_test * ds_measure
                neumann_funcs.append((g, g_fun))  # dynamic
            else:
                raise TypeError("Neumann BC 'g' must be Constant, Function, or callable")

    rhs_form = fem.form(L_form_expr)

    # Output setup
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with io.XDMFFile(mesh_comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(T_n, 0.0)

    # Probe point (only for rosenthal)
    #def probe_point(x, tol=1e-8):
    #    return (
    #       (np.abs(x[0] - 0.04) < tol) &
    #       (np.abs(x[1] - 0.01) < tol) &
    #       (np.abs(x[2] - 0.002) < tol)
    #)

    # Probe point (for the Dogbone)
    #def probe_point(x, tol=1e-8):
    #    return (
    #       (np.abs(x[0] - 0.004) < tol) &
    #       (np.abs(x[1] - 0.0025) < tol) &
    #       (np.abs(x[2] - 0.0) < tol)
    #)

    # Probe point
    def probe_point(x, tol=1e-8):
        return (
           (np.abs(x[0] - 0.0) < tol) &
           (np.abs(x[1] - 0.0) < tol) &
           (np.abs(x[2] - 0.0) < tol)
    )

    top_dof = locate_dofs_geometrical(Vt, probe_point)

    top_temp = []
    time_series = []

    start_time = time.time()
    T_max = -np.inf  # start with something lower than any real temperature


    # Time stepping
    with io.XDMFFile(mesh_comm, output_path, "a") as xdmf:
        for n in range(Nsteps):
            t = (n + 1) * dt

            if time_dependent_source:
                def f_expr(x): return source_term(x, t).astype(PETSc.ScalarType)
                f.interpolate(f_expr)

            for g_callable, g_fun in neumann_funcs:
                if g_callable is not None:
                    def g_expr(x, g_=g_callable): return g_(x, t).astype(PETSc.ScalarType)
                    g_fun.interpolate(g_expr)

            if bc_update_func is not None and source_bc is not None:
                source_bc.t = t
                bc_update_func.interpolate(lambda x: source_bc(x))

            b = assemble_vector(rhs_form)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            start, end = Vt.dofmap.index_map.local_range
            owned = np.arange(start, end, dtype=np.int32)

            T_n.x.array[owned] += dt * b.array[owned] * M_lumped_inv_array
            T_n.x.scatter_forward()

            for bc in bcs:
                bc.set(T_n.x.array, None)

            current_max = np.max(T_n.x.array) # max of all DOFs this step

            if current_max > T_max:
                T_max = current_max

            if len(top_dof) > 0:
                T_val = T_n.x.array[top_dof[0]]
                top_temp.append(T_val)
                time_series.append(t)
                if n % 100 == 0 or n == Nsteps - 1:
                    print(f"Step {n+1:05d} | Time: {t:.4f} | T_center: {T_val:.6f}")
            if n % 10 == 0:
                xdmf.write_function(T_n, t)

    print(f"\n Explicit simulation complete. {len(top_temp)} steps recorded.")
    elapsed_time = time.time() - start_time
    print(f" Maximum temperature reached: {T_max:.2f} K")
    print(f" Total CPU time: {elapsed_time:.2f} seconds")

    return time_series, top_temp, T_n


#======================================
#            THETA-SCHEME             #
#======================================
def heatdiff_theta_solver(domain, Vt, bcs, material_params, time_params,
                          initial_condition, source_term,
                          output_dir, output_filename,
                          theta=1.0,
                          bc_update_func=None, source_bc=None,
                          neumann_bcs=None):
    
    # Assert that theta is a valid value between 0 and 1
    assert 0.0 <= theta <= 1.0, f"θ must be between 0 and 1. Got {theta}"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Construct the full path for the XDMF output file
    output_path = os.path.join(output_dir, output_filename)

    # Extract material properties
    rho, Cp, k = material_params["rho"], material_params["Cp"], material_params["k_therm"]

    # Extract time stepping parameters
    t_end, Nsteps = time_params["t_end"], time_params["Nsteps"]
    dt = t_end / Nsteps
    # Get the MPI communicator for parallel operations
    mesh_comm = domain.comm

    # Define functions for temperature at previous (T_n) and current (T_sol) time steps
    T_n = fem.Function(Vt)
    T_n.name = "Temperature"
    T_n.interpolate(initial_condition)

    T_sol = fem.Function(Vt, name="Temperature")
    T_test = TestFunction(Vt)
    T_trial = TrialFunction(Vt)

    # Setup source
    if isinstance(source_term, (fem.Constant, fem.Function)):
        f = source_term
        time_dependent = False
    elif callable(source_term):
        f = fem.Function(Vt)
        f.interpolate(lambda x: source_term(x, 0.0).astype(PETSc.ScalarType))
        time_dependent = True
    else:
        raise TypeError("source_term must be Constant, Function, or callable")

    # Neumann BCs
    # Setup for Neumann Boundary Conditions (heat flux)
    # The theta scheme also requires Neumann fluxes at both t_n and t_{n+1}
    L_neumann = 0
    neumann_funcs = []
    if neumann_bcs:
        for g, ds in neumann_bcs:
            if isinstance(g, (fem.Constant, fem.Function)):
                L_neumann += g * T_test * ds
                neumann_funcs.append((None, g))
            elif callable(g):
                g_fun = fem.Function(Vt)
                g_fun.interpolate(lambda x: g(x, 0.0).astype(PETSc.ScalarType))
                L_neumann += g_fun * T_test * ds
                neumann_funcs.append((g, g_fun))
            else:
                raise TypeError("Invalid Neumann BC")

    with io.XDMFFile(mesh_comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(T_n, 0.0)

   # Probe point (only for rosenthal)
    #def probe(x, tol=1e-8):
    #    return (
    #       (np.abs(x[0] - 0.04) < tol) &
    #       (np.abs(x[1] - 0.01) < tol) &
    #       (np.abs(x[2] - 0.002) < tol)
    #)

    # Probe point for the simple diffusion 
    #def probe(x, tol=1e-8):
    #    return (
    #       (np.abs(x[0] - 0.0) < tol) &
    #       (np.abs(x[1] - 0.0) < tol) 
    #    )

    # Probe point for the dogbone
    def probe(x, tol=1e-8):
        return (
           (np.abs(x[0] - 0.004) < tol) &
           (np.abs(x[1] - 0.0025) < tol) &
           (np.abs(x[2] - 0.0) < tol)
    )

    top_dof = fem.locate_dofs_geometrical(Vt, probe)
    top_temp, time_series = [], []
    start_time = time.time() # Start timer for simulation duration

    # Precompute mass lumping for the explicit part of the scheme (when theta = 0.0)
    # This block is only entered if theta is exactly 0.0
    if theta == 0.0:
        one = fem.Function(Vt) # Create a function of ones
        one.interpolate(lambda x: np.ones_like(x[0])) # Interpolate 1.0 at all DoFs
        mass_form_lump = fem.form(rho * Cp * one * T_test * dx) # Formulate lumped mass matrix diagonal
        M_lump_vec = assemble_vector(mass_form_lump) # Assemble to a vector
        M_lump_vec.assemble() # Finalize assembly
        M_lumped_inv_array = 1.0 / M_lump_vec.array # Calculate inverse for direct multiplication

    # Time-stepping loop
    for n in range(Nsteps):
        t = (n + 1) * dt
        current_time = (n + 1) * dt # Calculate the current time (t_{n+1})
        
       # Update time-dependent source terms for current and previous time steps
        if time_dependent:
            f.interpolate(lambda x: source_term(x, t).astype(PETSc.ScalarType))
        for g_func, g_fun in neumann_funcs:
            if g_func:
                g_fun.interpolate(lambda x, g_=g_func: g_(x, t).astype(PETSc.ScalarType))

        if bc_update_func and source_bc:
            source_bc.t = t
            bc_update_func.interpolate(lambda x: source_bc(x))

        if theta == 1.0:
            a = (rho * Cp * T_trial * T_test / dt + k * dot(grad(T_trial), grad(T_test))) * ufl.dx
            L = (rho * Cp * T_n / dt + f) * T_test * ufl.dx + L_neumann
            problem = LinearProblem(a, L, u=T_sol, bcs=bcs,
                                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem.solve()
            T_n.x.array[:] = T_sol.x.array
        else:
            rhs = fem.form((-k * dot(grad(T_n), grad(T_test)) + f * T_test) * ufl.dx + L_neumann)
            b = assemble_vector(rhs)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b, bcs)

            start, end = Vt.dofmap.index_map.local_range
            owned = np.arange(start, end, dtype=np.int32)

            T_n.x.array[owned] += dt * b.array[owned] * M_lumped_inv_array
            T_n.x.scatter_forward()
            for bc in bcs:
                bc.set(T_n.x.array, None)
            T_sol.x.array[:] = T_n.x.array

        # Monitor temperature at the probe point
        if len(top_dof) > 0: # Check if a probe DOF was successfully located
            top_val = T_sol.x.array[top_dof[0]] # Get temperature value at the probe DOF
            top_temp.append(top_val) # Store temperature
            time_series.append(current_time) # Store current time
            
            # Print progress only every 100 steps or at the first/last step
            if n == 0 or (n + 1) % 100 == 0 or n == Nsteps - 1:
                print(f"Step {n+1:04d} | t = {current_time:.3f} | T_probe = {top_val:.6f}")
        else:
            # Print warning if no probe DOF found
            if n == 0 or (n + 1) % 100 == 0 or n == Nsteps - 1:
                print(f"Step {n+1:04d} | t = {current_time:.3f} | Warning: No probe DOF found for monitoring.")

        # Write solution to XDMF file (e.g., every 10 steps to reduce file size, or every step)
        if (n + 1) % 10 == 0 or n == Nsteps - 1: # Example: write every 10 steps and at the very last step
            with io.XDMFFile(mesh_comm, output_path, "a") as xdmf:
                xdmf.write_function(T_sol, current_time)

    elapsed_time = time.time() - start_time # Calculate total simulation time
    print(f"\n θ-solver complete. {len(top_temp)} steps recorded. Total CPU time: {elapsed_time:.2f} seconds.")
    # Return the time series, probe point temperatures, and the final temperature field
    return time_series, top_temp, T_sol