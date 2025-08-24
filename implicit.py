# 23.07.2024

#This code runs laser scanning simulations with the addition of evaporation, latent heat of fusion and radiation. The problem is solved with an implicit scheme and a Newton-Rapshon solver

#=====================================
#             LIBRARIES             
#=====================================

# Standard Python Libraries
import os        # Allows interaction with the operating system
import math      # Provides access to mathematical functions
import time      # Provides time-related functions, we need it to track the cimulation time

# Computing Libraries
import numpy as np                          # The fundamental package for numerical computations
import matplotlib.pyplot as plt             # 2D plotting library for data visualization

# MPI and Parallel Computing
from mpi4py import MPI                      #Enables parallel computing 

# PETSc and FEniCSx (for Finite Element Methods)
from petsc4py import PETSc                  # Interface to PETSc, used for scalable linear algebra and solvers
from dolfinx import fem, io, nls  # FEM tools for defining/solving PDEs, file I/O for meshes and results, and nonlinear solver support

from dolfinx.nls.petsc import NewtonSolver  # Newton nonlinear solver using PETSc 

# UFL (Unified Form Language)
import ufl                                  # Language for expressing variational forms in FEniCSx
from ufl import (
    grad, dot, TestFunction, TrialFunction, dx, ds, exp, Measure, derivative
)                                           # Key symbolic operations for defining variational problems

# Visualization
import matplotlib.pyplot as plt            # Library to plot the results

#=====================================
#                 MESH             
#=====================================

mesh_path = "/home/ntinos/Documents/FEnics/heat equation/checkpoints/Dogbone3D.xdmf"          # path to the mesh file
output_dir, output_filename = "results_nonlinear_evaporation", "nonlinear_final.xdmf"         # output folder and xdmf name
os.makedirs(output_dir, exist_ok=True)                                                        # Create the output directory if it doesn't already exist
output_path = os.path.join(output_dir, output_filename)

# Read mesh and associated facet tags from the XDMF file
with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
    domain = xdmf.read_mesh(name="dogbone_mesh")                                              # Load mesh
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)         # Connects the facets with the cells
    facet_tags = xdmf.read_meshtags(domain, name="facet_tags")                                # read booundary markers, facets where the boundary conditions will be applied

Vt = fem.functionspace(domain,("Lagrange", 1))                                                # Create a first-order Lagrange function space on the domain
fdim = domain.topology.dim - 1                                                                # Determine the topological dimension of facets (in 3D geometry fdim=2)
left_dofs = fem.locate_dofs_topological(Vt, fdim, facet_tags.indices[facet_tags.values == 2]) # left_dofs and right_dofs are where the dirichlet coudnary conditions are applied
right_dofs = fem.locate_dofs_topological(Vt, fdim, facet_tags.indices[facet_tags.values == 3])# ID 2 and 3 are assigned in the Dogbone3D_geometry.py

#----- laser surface------
ds_surface = Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=5)          # Defines a surface measure for integration over the laser-exposed boundary (facet tag 5)

#=====================================
#           DICTIONARIES             
#=====================================

Stainless_Steel = {
    "Density": 7850.0,                   # Density (kg/m^3)
    "Specific_heat": 500.0,              # Specific heat capacity (J/kg.K)
    "thermal_conductivity": 15.0,        # Thermal conductivity (W/m.K)
    "T_boil": 3090,                      # Boiling temperature (K)
    "T_solidus": 1700.0,                 # Solidus temperature (K)
    "T_liquidus": 1800.0,                # Liquidus temperature (K)
    "latent_heat_fusion": 267.7e3,       # Latent heat of fusion
    "latent_heat_evaporation": 7.41e6,   # latent heat of evaporation
    "Emissivity": 0.35,                  # for the radiaion
    "Stefan_Boltzmann": 5.670e-8          
}

Environment_conditions = {

    "Atmospheric_Pressure": 101325.0,    # Pa
    "Gas_constant": 150.774,             # J/(kg·K)
}

unit = 0.001
gauge_radius = 5 * unit / 2

laser_parameters = {
    "Absorptivity": 0.30,  # Absorptivity of the material (dimensionless)
    "Power": 200.0,  # Laser power (W)
    "Radius": 60e-6,  # Laser spot radius (m)
    "Scan_speed": 0.8,  # Laser scanning speed (m/s)
    "y0": gauge_radius  # Initial y-coordinate for the laser
}

#=====================================
#           TIME STEP           
#=====================================

t_end, Nsteps = 0.45, 60000   
dt = t_end/Nsteps  

#=====================================
#         CLASSES DEFINITIONS             
#=====================================

x_gauge_half = 8 * unit / 2 # this is the position where the laser starts scanning, for more details see Dogbone2D_geometry.py

# Class representing a moving laser heat source, this way only the laser parameters need to change and you have a different laser.
class MovingLaser:
    def __init__(self, params):
        self.A = params["Absorptivity"]
        self.P = params["Power"]
        self.R = params["Radius"]
        self.v = params["Scan_speed"]
        self.y0 = params["y0"]
        self.t = 0.0
        self.peak = 2 * self.A * self.P / (2*np.pi * self.R**2)

    def __call__(self, x, t):
        x0 = -x_gauge_half + self.v * t
        r2 = (x[0] - x0)**2 + (x[1] - self.y0)**2 + x[2]**2
        return self.peak * np.exp(-2 * r2 / self.R**2)

laser_func = fem.Function(Vt)                                                         # Create a FEM function on the function space Vt to store laser source values (a container)
laser_source_func = MovingLaser(laser_parameters)                                     # set the laser parameters form the dictionary

# Interpolate the laser heat source at time t = 0.0 onto all nodes of the FEM space, this sets the initial spatial distribution of the laser source as a FEM function.
laser_func.interpolate(lambda x: laser_source_func(x, 0.0).astype(PETSc.ScalarType))

# =====================================
#          VARIATIONAL FORM           #
#======================================

#initialization of Tn
T_room = 293      # Ambient room temperature (K)
# Define the initial temperature condition as a constant field equal to room temperature
def initial_condition(x):
    return np.full(x.shape[1], T_room, dtype=PETSc.ScalarType)

# Creates FEM function T_n to hold the temperature at the current time step
T_n = fem.Function(Vt); T_n.name = "Temperature_n"; T_n.interpolate(initial_condition)

T_test = TestFunction(Vt)   # Define trial and test functions for the variational problem
T_trial  = TrialFunction(Vt)
T_sol = fem.Function(Vt, name = "Temperature") # Create a FEM function to store the temperature solution

#--------boundary conditions---------
bcs = [fem.dirichletbc(PETSc.ScalarType(T_room), d, Vt) for d in [left_dofs, right_dofs]]

#---material & physical properties---
rho, Specific_heat, k = Stainless_Steel["Density"], Stainless_Steel["Specific_heat"], Stainless_Steel["thermal_conductivity"]
T_solidus, T_liquidus = Stainless_Steel["T_solidus"], Stainless_Steel["T_liquidus"]
latent_heat_fusion = Stainless_Steel["latent_heat_fusion"]

#--------evaporation term---------
Pa, Rv = Environment_conditions["Atmospheric_Pressure"], Environment_conditions["Gas_constant"]
deltaH_lv, T_boil = Stainless_Steel["latent_heat_evaporation"], Stainless_Steel["T_boil"]

prefactor = 0.82 * Pa*deltaH_lv/ ufl.sqrt(2 * np.pi * Rv * T_sol) # W/m^2
# Define evaporation heat loss as a temperature-dependent function (nonlinear)
# This term will act as a surface heat sink in the variational formulation
Q_evap = prefactor * exp ((deltaH_lv/ (T_boil*Rv))* (1-(T_boil / T_sol)))
# radiation term
sigma = Stainless_Steel["Stefan_Boltzmann"]
epsilon = Stainless_Steel["Emissivity"]
Q_rad = epsilon * sigma * (T_sol**4 - T_room**4)
                          
X_melt_expr = ufl.conditional(
    ufl.lt(T_n, T_solidus), PETSc.ScalarType(0.0),
    ufl.conditional(
        ufl.gt(T_n, T_liquidus), PETSc.ScalarType(1.0),
        (T_n - T_solidus) / (T_liquidus - T_solidus)
    )
)

dXdt = ufl.conditional(
    ufl.And(ufl.ge(T_n, T_solidus), ufl.le(T_n, T_liquidus)),
    1.0 / (T_liquidus - T_solidus) * (T_sol - T_n) / dt,
    PETSc.ScalarType(0.0)
)

# --- Latent heat source term (analytical) ---
Q_latent = rho * latent_heat_fusion * dXdt

Residual = (rho * Specific_heat* (T_sol-T_n)/dt) * T_test *dx \
           + k *  dot (grad(T_sol), grad(T_test))*dx \
           - laser_func * T_test * ds\
           #+ Q_evap * T_test * ds\
           #+ Q_rad * T_test * ds\
           #+ Q_latent * T_test * dx

# --- Jacobian of the Residual ---
Jacobian = derivative (Residual, T_sol, T_trial) # that means that the jacobian is computed in the direction of the trial function (check J.Bleyer hyperelasticity)

#--- Solver setup -----
problem = fem.petsc.NonlinearProblem(Residual, T_sol, bcs)
solver = nls.petsc.NewtonSolver(domain.comm, problem)    # Set a Newton solver for the nonlinear problem
# Set linear solver type for the inner Krylov iteration (used in each Newton step)
solver.krylov_solver.setType("gmres")     # GMRES(solver) for general nonlinear systems
solver.krylov_solver.getPC().setType("ilu")  # select the preconditioner (ilu works well for the problem)
solver.krylov_solver.getPC().setReusePreconditioner(True)
solver.krylov_solver.setTolerances(rtol=1e-6, atol=1e-8, max_it=1000)

solver.atol = 1e-5
solver.rtol = 1e-5
solver.convergence_criterion = "incremental"

# --- Output Init ---
with io.XDMFFile(domain.comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(T_n, 0.0)

# --- Probe point monitor setup: Define a small function to locate a specific probe point in the domain
def probe(x, tol=1e-8): 
    return (np.abs(x[0] - 0.004) < tol) & (np.abs(x[1] - 0.0025) < tol)

top_dof = fem.locate_dofs_geometrical(Vt, probe)                       # Locate degrees of freedom (DoFs) at the probe point
if len(top_dof) == 0: print("Warning: No DOF found at probe point.")   # Warn if no DoFs are found at the probe location (could be due to mesh resolution)

top_temps, time_series = [], []                                        # lists to store probe temperature and time during time
print("\n--- Starting Non-Linear Heat Diffusion Simulation (Laser + Evaporation) ---")
start = time.time()

# Open the XDMF file in append mode to store solution snapshots (e.g., for ParaView)
# Open the XDMF file in append mode to store solution snapshots (e.g., for ParaView)
with io.XDMFFile(domain.comm, output_path, "a") as xdmf:
    
    # Time-stepping loop
    
    for n in range(10000):
        t = (n + 1) * dt  # Compute current simulation time

        # Interpolate the time-dependent laser source at current time tNsteps
        laser_func.interpolate(
            lambda x: laser_source_func(x, t).astype(PETSc.ScalarType)
        )

        # Initialize the solution at current time step using the previous solution
        T_sol.x.array[:] = T_n.x.array

        # Solve the nonlinear PDE system and return number of iterations and success flag
        its, converged = solver.solve(T_sol)

        # Check if the nonlinear solver successfully converged
        if not converged:
            raise RuntimeError(f"Solver failed at t = {t:.4f}s after {its} iterations.")

        # --------------------------------------------------------------------------

        # Update the previous solution with the current one for the next time step
        T_n.x.array[:] = T_sol.x.array

        # Write the current temperature field to the XDMF output file with time tag
        
        if (n+1)% 5000 == 0 or n == Nsteps-1:
             xdmf.write_function(T_sol, t)

        # If a probe degree of freedom (DOF) is defined, extract and store the temperature at that point
        if top_dof.size:
            T_top = T_sol.x.array[top_dof[0]]  # Temperature at probe point
            top_temps.append(T_top)            # Store probe temperature
            time_series.append(t)              # Store corresponding time
            print(
                f"Step {n+1:03d} | Time: {t:.4f}s | Probe Temp: {T_top:.2f} K | Newton iterations: {its}"
            )
            print(f"Max T: {T_sol.x.array.max():.2f} K")
            
        else:
            # If no probe DOF is found, just print status
            print(
                f"Step {n+1:03d} | Time: {t:.4f}s | No probe DOF found | Newton iterations: {its}"
            )

# Final timing message
print(f"Simulation complete at t = {time.time()-start:.2f} seconds")


# --- Save probe temperature history ---
#output_txt = os.path.join(output_dir, "nonlin_Scenario4_P200_v0.8.txt")
output_txt = os.path.join(output_dir, "laser.txt")

with open(output_txt, "w") as f:
    f.write("# Time (s)\tTemperature (K)\n")
    for t, T in zip(time_series, top_temps):
        f.write(f"{t:.6f}\t{T:.2f}\n")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(time_series, top_temps, label="Temperature at Probe Point")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Temperature Evolution at Probe Point (Laser + Evaporation)")
plt.grid(True)
plt.legend()
plt.show()

