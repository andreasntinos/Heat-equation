from dolfinx import mesh, fem, io
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
import os

# Output path
out_file = "lasertest.xdmf"
output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Geometry
gdim, fdim = 2, 1
Nx, Ny = 100, 100
length_half = 0.01
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
        np.isclose(x[0], -0.01) |
        np.isclose(x[0],  0.01) |
        np.isclose(x[1], -0.01)
    )

bc_facets = mesh.locate_entities_boundary(domain, fdim, dirichlet_boundary)
bc_dofs = fem.locate_dofs_topological(Vt, fdim, bc_facets)
bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, Vt)]

# Material + time params (steel-like)
material_params = {"rho": 7850.0, "Cp": 500.0, "k_therm": 15.0}
alpha = material_params["k_therm"] / (material_params["rho"] * material_params["Cp"])  # thermal diffusivity
CFL_term  = (length_half /Nx)**2 / alpha
print(f"CFL condition demands: dt<{CFL_term:.4f}")
time_params = {"t_end": 5.0, "Nsteps": 3000}  

#instead of the implcicit solver, the CFL condition is checked here in order to ensure stability
if time_params["t_end"] / time_params["Nsteps"] > CFL_term: 
    print(f"Warning: time step too large for stability: dt={time_params['t_end'] / time_params['Nsteps']:.4f} > CFL={CFL_term:.4f}")
    raise ValueError(f"Time step too large for stability: dt={time_params['t_end'] / time_params['Nsteps']:.4f} > CFL={CFL_term:.4f}")
  
# Initial condition: zero everywhere
def initial_condition(x): return np.full(x.shape[1], 0.0)

# Source term (none)
source_term = fem.Constant(domain, PETSc.ScalarType(0.0))

# Tag top boundary for Neumann BC (y = 0.01)
from dolfinx.mesh import meshtags, locate_entities
top_facets = locate_entities(domain, fdim, lambda x: np.abs(x[1] - 0.01) < 1e-8)
values = np.full(len(top_facets), 1, dtype=np.int32)
facet_tags = meshtags(domain, fdim, top_facets, values)

# Neumann BC: constant heat flux on top
# g = fem.Constant(domain, PETSc.ScalarType(1e5))  # W/m² # in case you want to test a cnstant heat flux
# neumann_conditions = [(g, ds_top)]

class MovingLaser:
    def __init__(self, A=0.5, P=100.0, R=0.0015, v=0.01, y0=0.01):
        self.peak = 2 * A * P / (np.pi * R**2)
        self.R = R
        self.v = v
        self.y0 = y0
        self.t = 0.0

    def __call__(self, x):
        x0 = -0.01 + self.v * self.t
        r2 = (x[0] - x0)**2 + (x[1] - self.y0)**2
        return self.peak * np.exp(-2 * r2 / self.R**2)

laser = MovingLaser()
scanned_distance = laser.v * time_params["t_end"]  # total distance scanned by the laser

if scanned_distance > 2*length_half:
    print(f"Warning: Scanned distance {scanned_distance:.4f} m exceeds domain size (0.02 m).")

ds_top = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=1)

laser = MovingLaser()

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
