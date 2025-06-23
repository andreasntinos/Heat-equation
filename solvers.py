## 13.06.2025

# This script is the definition of FE solvers. The heat diffusion problem is solved by using the implicit and the explicit scheme, 
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
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, set_bc
from dolfinx.fem import Constant, Function, locate_dofs_geometrical, set_bc

def heatdiff_implicit_solver(domain, Vt, bcs, material_params, time_params,
                             initial_condition, source_term,
                             output_dir, output_filename,
                             bc_update_func=None, source_bc=None,
                             neumann_bcs=None):

    os.makedirs(output_dir, exist_ok=True) # create output directory if it doesn't exist
    output_path = os.path.join(output_dir, output_filename) # the path where the xdmf files are saved

    # Parameters
    rho, Cp, k = material_params["rho"], material_params["Cp"], material_params["k_therm"]
    t_end, Nsteps = time_params["t_end"], time_params["Nsteps"]
    dt = t_end / Nsteps # time step size
    print(f"Time step used:{dt:.4f} seconds")
    mesh_comm = domain.comm # communicator for the mesh

    # Functions
    T_n = fem.Function(Vt) # function to store the solution
    T_n.name = "Temperature_n" # Function to store the previous time step solution
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
    a = ((rho * Cp * T_trial * T_test / dt + k * dot(grad(T_trial), grad(T_test))) * ufl.dx)
    
    # Base RHS form
    L_form_expr = (rho * Cp / dt * T_n + f) * T_test * ufl.dx

    neumann_funcs = []

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

    problem = LinearProblem(a, L_form, u=T_sol, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    # Output file setup
    with io.XDMFFile(mesh_comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(T_n, 0.0)

    # Top point monitoring
    def probe_point(x, tol=1e-8):
       #return np.logical_and(np.abs(x[0] - 0.004) < tol, np.abs(x[1] - 0.0025) < tol) #for the dogbone
       return np.logical_and(np.abs(x[0] - 0.0) < tol, np.abs(x[1] - 0.0) < tol) #for the lasertest
 
    top_dof = fem.locate_dofs_geometrical(Vt, probe_point)
    if len(top_dof) == 0:
        print("Warning: No DOF found at domain center (0.5, 0.5)")

    top_temperatures = []
    time_series = []

    # Time-stepping loop
    with io.XDMFFile(mesh_comm, output_path, "a") as xdmf:
        for n in range(Nsteps):
            current_time = (n + 1) * dt

            # Time-varying source term
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
    return time_series, top_temperatures, T_sol


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
    print(f"Time step used: {dt:.4f} seconds")

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

    # Probe point
    def probe_point(x, tol=1e-8):
    # return np.logical_and(np.abs(x[0] - 0.004) < tol, np.abs(x[1] - 0.0025) < tol)  # for the dogbone
      return np.logical_and(np.abs(x[0] - 0.0) < tol, np.abs(x[1] - 0.004) < tol)          # for the lasertest

    top_dof = locate_dofs_geometrical(Vt, probe_point)

    top_temp = []
    time_series = []

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

            if len(top_dof) > 0:
                T_val = T_n.x.array[top_dof[0]]
                top_temp.append(T_val)
                time_series.append(t)
                if n % 100 == 0 or n == Nsteps - 1:
                    print(f"Step {n+1:05d} | Time: {t:.4f} | T_center: {T_val:.6f}")

            xdmf.write_function(T_n, t)

    print(f"\n✅ Explicit simulation complete. {len(top_temp)} steps recorded.")
    return time_series, top_temp, T_n
