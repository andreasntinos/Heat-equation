# 22/ 07/2025
# this code runs the laser ascanning simulation on the cylindrical specimen, by calling the heat.diffusion solver and importing the mesh
# as an xdmf file (the mesh is genreated in the Dogbone3D_geometry.py)
# the result is tracking the temperature of a specific point (the end of the gage area) and this result is exported as a txt file that is then imported in the post-process codes
# careful, to run the post porcess files, you need to move them manually to the txt folder

#===========================================
#                 LIBRARIES
#===========================================
from mpi4py import MPI  # Import MPI for parallel processing
import numpy as np  # Import NumPy for numerical operations
import math  # Import math for mathematical functions
import os  # Import os for operating system interactions
from petsc4py import PETSc  # Import PETSc for high-performance numerical computing
from dolfinx import fem, io  # Import DOLFINx components for finite element methods and I/O
import ufl  # Import UFL for Unified Form Language
from ufl import TestFunction, TrialFunction, grad, dot  # Import specific UFL functions


# --- Load Mesh and Facet Tags ---
mesh_path = "/home/ntinos/Documents/FEnics/heat equation/checkpoints/Dogbone3D.xdmf"

# Load the mesh and facet tags from an XDMF file
with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
    domain = xdmf.read_mesh(name="dogbone_mesh") # Read the mesh named "dogbone_mesh"
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    facet_tags = xdmf.read_meshtags(domain, name="facet_tags") # Read facet tags

# --- Geometry and Tags ---
gdim, fdim = 3, 2
unit = 0.001
gauge_length = 8 * unit
gauge_diameter = 5 * unit
fillet_radius = 13.5 * unit
gauge_radius = gauge_diameter / 2.0
x_gauge_half = gauge_length / 2.0
delta_half_width = (13.5 * unit / 2.0) - gauge_radius
dx_fillet = math.sqrt(delta_half_width * (2 * fillet_radius - delta_half_width))
x_fillet_end = x_gauge_half + dx_fillet

# --- FE Space and Initial Condition ---
Vt = fem.functionspace(domain, ("Lagrange", 1))  # Define a Lagrange finite element space of degree 1
T_room = 293.0   # Room temperature in Kelvin

# Define the initial temperature condition as room temperature everywhere
def initial_condition(x):
    return np.full(x.shape[1], T_room, dtype=PETSc.ScalarType)

# --- Dirichlet BCs ---
left_side_tag, right_side_tag = 2, 3  # Get the indices of facets corresponding to the left and right side tags
left_facets = facet_tags.indices[facet_tags.values == left_side_tag]
right_facets = facet_tags.indices[facet_tags.values == right_side_tag]
left_dofs = fem.locate_dofs_topological(Vt, fdim, left_facets)
right_dofs = fem.locate_dofs_topological(Vt, fdim, right_facets)
bcs = [
    fem.dirichletbc(PETSc.ScalarType(T_room), left_dofs, Vt),
    fem.dirichletbc(PETSc.ScalarType(T_room), right_dofs, Vt)
]

# --- Material and Time Parameters ---
material_params = {
    "rho": 7850.0,  # Density (kg/m^3)
    "Cp": 500.0,  # Specific heat capacity (J/kg.K)
    "k_therm": 15.0  # Thermal conductivity (W/m.K)
}
time_params = {
    "t_end": 0.25,  # End time of the simulation (s)
    "Nsteps": 250000,  # Number of time steps
    "h_fine": 0.00002  # Characteristic mesh size for CFL condition (we add this for printing the CFL condition)
}

# --- Moving Laser Source (Callable) ---
laser_params = {
    "Absorptivity": 0.30,  # Absorptivity of the material (dimensionless)
    "Power": 45.0,  # Laser power (W)
    "Radius": 60e-6,  # Laser spot radius (m)
    "Scan_speed": 0.15,  # Laser scanning speed (m/s)
    "y0": gauge_radius  # Initial y-coordinate for the laser
}

alpha = material_params["k_therm"] / (material_params["rho"] * material_params["Cp"])
print(f"Thermal diffusivity α = {alpha:.3e} m²/s")

CFL = time_params["h_fine"]**2 / (2*alpha)
t = time_params["t_end"] / time_params["Nsteps"]
print(f"The CFL condition demands dt<{CFL:.6e} s")
if t > CFL:
    raise RuntimeError(f" The CFL condition is not respected! dt={t:.7e} > dt_CFL={CFL:.7e}")

# Define the MovingLaser class
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

laser = MovingLaser(laser_params)

# --- Neumann Boundary (Laser) ---
# Define the measure for the Neumann boundary condition (laser heat source) on subdomain_id 5
ds_laser = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=5)
neumann_conditions = [(laser, ds_laser)]
source_term = fem.Constant(domain, PETSc.ScalarType(0.0))

# --- Solver Call ---
from solvers import heatdiff_implicit_solver, heatdiff_explicit_solver

output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
output_filename = "Heat3D_dogbone.xdmf"

time_series, center_temp, T_final = heatdiff_explicit_solver(
    domain=domain,
    Vt=Vt,
    bcs=bcs,
    material_params=material_params,
    time_params=time_params,
    initial_condition=initial_condition,
    source_term=source_term,  # internal source = 0
    output_dir=output_dir,
    output_filename=output_filename,
    neumann_bcs=neumann_conditions
)

import matplotlib.pyplot as plt

#np.savetxt("lin_h0.0025.txt", np.column_stack((time_series, center_temp)), fmt="%.6e", header="time, center_temp", comments="# ")
#np.savetxt("SpeedP70_v1.0.txt", np.column_stack((time_series, center_temp)), fmt="%.6e", header="time, center_temp", comments="# ")
#np.savetxt("Speedv0.1_P70.txt", np.column_stack((time_series, center_temp)), fmt="%.6e", header="time, center_temp", comments="# ")
np.savetxt("lin_Scenario3_P45_v0.15.txt", np.column_stack((time_series, center_temp)), fmt="%.6e", header="time, center_temp", comments="#") 

# Plot the temperature at the laser center over time if running on the root MPI process
if MPI.COMM_WORLD.rank == 0:
    plt.figure(figsize=(8, 5))
    plt.plot(time_series, center_temp, label="T at laser center", lw=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature at Laser Center Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"T_vs_time_{int(laser_params['Radius']*1e6)}um.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Saved plot to: {plot_path}")
