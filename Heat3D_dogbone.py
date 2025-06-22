#===========================================
#                 LIBRARIES
#===========================================
from mpi4py import MPI
import numpy as np
import math
import os
from petsc4py import PETSc
from dolfinx import fem, io
import ufl
from ufl import TestFunction, TrialFunction, grad, dot

# --- Load Mesh and Facet Tags ---
mesh_path = "/home/ntinos/Documents/FEnics/heat equation/checkpoints/Dogbone3D.xdmf"

with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
    domain = xdmf.read_mesh(name="dogbone_mesh")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    facet_tags = xdmf.read_meshtags(domain, name="facet_tags")

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
Vt = fem.functionspace(domain, ("Lagrange", 1))
T_room = 293.0

def initial_condition(x):
    return np.full(x.shape[1], T_room, dtype=PETSc.ScalarType)

# --- Dirichlet BCs ---
left_side_tag, right_side_tag = 2, 3
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
    "rho": 7850.0,
    "Cp": 500.0,
    "k_therm": 15.0
}
time_params = {
    "t_end": 0.16,
    "Nsteps": 300
}

# --- Moving Laser Source (Callable) ---
laser_params = {
    "Absorptivity": 0.2,
    "Power": 70.0,
    "Radius": 130e-6,
    "Scan_speed": 0.1,
    "y0": gauge_radius
}

class MovingLaser:
    def __init__(self, params):
        self.A = params["Absorptivity"]
        self.P = params["Power"]
        self.R = params["Radius"]
        self.v = params["Scan_speed"]
        self.y0 = params["y0"]
        self.t = 0.0
        self.peak = 2 * self.A * self.P / (np.pi * self.R**2)

    def __call__(self, x, t):
        x0 = -x_gauge_half + self.v * t
        r2 = (x[0] - x0)**2 + (x[1] - self.y0)**2 + x[2]**2
        return self.peak * np.exp(-2 * r2 / self.R**2)

laser = MovingLaser(laser_params)

# --- Neumann Boundary (Laser) ---
ds_laser = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=5)
neumann_conditions = [(laser, ds_laser)]
source_term = fem.Constant(domain, PETSc.ScalarType(0.0))

# --- Solver Call ---
from solvers import heatdiff_implicit_solver

output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
output_filename = "Heat3D_dogbone.xdmf"

time_series, center_temp, T_final = heatdiff_implicit_solver(
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


np.savetxt("mesh_size2_3D.txt", np.column_stack((time_series, center_temp)), fmt="%.6e", header="time, center_temp", comments="# ")

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
