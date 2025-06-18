from dolfinx import mesh, fem, io
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
import os
import matplotlib.pyplot as plt

# Output path
out_file = "lasertest.xdmf"
output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Geometry
gdim, fdim = 2, 1
Nx, Ny = 100, 100
length_half = 0.004
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-length_half, -length_half]), np.array([length_half, length_half])],  # 2cm x 2cm domain
    [Nx, Ny],
    mesh.CellType.triangle,
)
Vt = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet BCs on left, right, bottom (correct domain values!)
def dirichlet_boundary(x):
    return (
        np.isclose(x[0], -length_half) |
        np.isclose(x[0],  length_half) |
        np.isclose(x[1], -length_half) 
    )

bc_facets = mesh.locate_entities_boundary(domain, fdim, dirichlet_boundary)
bc_dofs = fem.locate_dofs_topological(Vt, fdim, bc_facets)
T_room = 293.0  # Room temperature in Kelvin
bcs = [fem.dirichletbc(PETSc.ScalarType(T_room), bc_dofs, Vt)]

# Material + time params (steel-like)
material_params = {"rho": 7850.0,    # density in kg/m³
                   "Cp": 500.0,      # specific heat capacity in J/kg·K
                   "k_therm": 15.0   # thermal conductivity in W/m·K
                   }

alpha = material_params["k_therm"] / (material_params["rho"] * material_params["Cp"])  # thermal diffusivity
CFL_term  = (length_half /Nx)**2 / alpha
print(f"CFL condition demands: dt<{CFL_term:.4f}")

time_params = {"t_end": 0.08,  # total time in seconds, the total time is given by the laser 
                "Nsteps": 1000}  # number of time steps

#instead of the implcicit solver, the CFL condition is checked here in order to ensure stability
if time_params["t_end"] / time_params["Nsteps"] > CFL_term: 
    print(f"Warning: time step too large for stability: dt={time_params['t_end'] / time_params['Nsteps']:.4f} > CFL={CFL_term:.4f}")
    raise ValueError(f"Time step too large for stability: dt={time_params['t_end'] / time_params['Nsteps']:.4f} > CFL={CFL_term:.4f}")
  
# Initial condition: zero everywhere
def initial_condition(x): return np.full(x.shape[1], T_room)

# Source term (none)
source_term = fem.Constant(domain, PETSc.ScalarType(0.0))

# Tag top boundary for Neumann BC (y = 0.033) 
from dolfinx.mesh import meshtags, locate_entities
top_facets = locate_entities(domain, fdim, lambda x: np.abs(x[1] - length_half) < 1e-8)
values = np.full(len(top_facets), 1, dtype=np.int32)
facet_tags = meshtags(domain, fdim, top_facets, values)

# Neumann BC: constant heat flux on top
# g = fem.Constant(domain, PETSc.ScalarType(1e5))  # W/m² # in case you want to test a cnstant heat flux
# neumann_conditions = [(g, ds_top)]

laser_params = {
    "A": 0.15,           # absorptivity
    "P": 70.0,         # power [W]
    "R": 130e-6,        # beam radius [m]
    "v": 0.1,          # scan speed [m/s]
    "y0": length_half          # vertical position [m]
}

class MovingLaser:
    def __init__(self, params):
        self.A = params["A"]
        self.P = params["P"]
        self.R = params["R"]
        self.v = params["v"]
        self.y0 = params["y0"]
        self.t = 0.0
        self.peak = 2 * self.A * self.P / (np.pi * self.R**2)

    def __call__(self, x):
        x0 = -length_half + self.v * self.t
        r2 = (x[0] - x0)**2 + (x[1] - self.y0)**2
        return self.peak * np.exp(-2 * r2 / self.R**2)


laser = MovingLaser(laser_params)
scanned_distance = laser.v * time_params["t_end"]  # total distance scanned by the laser

if scanned_distance > 2*length_half:
    print(f"Warning: Scanned distance {scanned_distance:.4f} m exceeds domain size (0.02 m).")

ds_top = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=1)

def g_laser(x, t):
    laser.t = t
    return laser(x).astype(PETSc.ScalarType)

neumann_conditions = [(g_laser, ds_top)]

# Call solver
from solvers import heatdiff_implicit_solver

time_series, center_temp, T_final = heatdiff_implicit_solver(
    domain=domain,
    Vt=Vt,
    bcs=bcs,
    material_params=material_params,
    time_params=time_params,
    initial_condition=initial_condition,
    source_term=source_term,
    output_dir=output_dir,
    output_filename=out_file,
    neumann_bcs=neumann_conditions
)

# ===================================================
#              VALIDATION WITH GREEN'S FUNCTION
# ===================================================

import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.plot(time_series, center_temp, label="FEniCS solution for the top center point")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
plt.title("Temperature at the center of the top boundary over time")
plt.grid()
plt.legend()
plt.savefig(os.path.join(script_dir, "temperature_center_top_boundary.png"))
plt.show()  
