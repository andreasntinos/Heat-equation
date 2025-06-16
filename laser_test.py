from dolfinx import mesh, fem, io
from mpi4py import MPI
import numpy as np
import ufl
from petsc4py import PETSc
import os

#============================================
#       MESH IMPORT & GEOMETRY VALUES
#============================================

# Output path
out_file = "lasertest.xdmf" 

output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints" # the path where the xdmf files are saved (changes accordingly)
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)
gdim, fdim = 2, 1

domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2.0, -1.0]), np.array([2.0, 1.0])],
                                 [50, 50], mesh.CellType.triangle)

Vt = fem.functionspace(domain, ("Lagrange", 1))  # Scalar function space with Lagrange elements of order 1


# Dirichlet boundaries: left, right, bottom
def dirichlet_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

bc_facets = mesh.locate_entities_boundary(domain, fdim, dirichlet_boundary)
bc_dofs = fem.locate_dofs_topological(Vt, fdim, bc_facets)
bcs = [fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, Vt)]


# Material + time params
material_params = {"rho": 1.0, "Cp": 1.0, "k_therm": 1.0}
time_params = {"t_end": 1.0, "Nsteps": 10}

# Initial condition
def initial_condition(x): return np.full(x.shape[1], 0.0)

# Source term (none)
source_term = fem.Constant(domain, PETSc.ScalarType(0.0))

# Facet tagging
from dolfinx.mesh import meshtags, locate_entities
top_facets = locate_entities(domain, fdim, lambda x: np.isclose(x[1], 1.0))
values = np.full(len(top_facets), 1, dtype=np.int32)
facet_tags = meshtags(domain, fdim, top_facets, values)

#============================================
#             LASER EXPRESSION
#============================================

laser_params = {
    "A": 0.5,            # Absorptivity (0–1)
    "P": 60.0,           # Laser power [W]
    "R": 0.001,          # Beam radius [m]
    "v": 0.01,           # Scan speed [m/s]
    "y0": 0.0,           # Vertical (y) position of laser
    "x_start": -0.005,   # Start x
    "x_end": 0.005       # End x
}

class MovingLaser:
    def __init__(self, params):
        self.A = params["A"]
        self.P = params["P"]
        self.R = params["R"]
        self.v = params["v"]
        self.y0 = params["y0"]
        self.x_start = params["x_start"]
        self.t = 0.0
        self.peak = (2 * self.A * self.P) / (np.pi * self.R**2)

    def __call__(self, x):
        x0 = self.x_start + self.v * self.t
        r2 = (x[0] - x0)**2 + (x[1] - self.y0)**2
        return self.peak * np.exp(-2 * r2 / self.R**2)

laser_model = MovingLaser(laser_params)
g_laser = fem.Function(Vt)


# Neumann BC on top (facet ID = 1)
g = fem.Constant(domain, PETSc.ScalarType(100.0))
ds_top = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=1)
neumann_conditions = [(g, ds_top)]

# Solver
from solvers import heatdiff_implicit_solver  # Replace with correct import

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
