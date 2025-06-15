import numpy as np
import matplotlib.pyplot as plt

# === Load data ===
#my_data = np.loadtxt("heat_diffusion.txt")
#doken_data = np.loadtxt("dokken_output.txt")

#t1, T1 = my_data[:, 0], my_data[:, 1]
#t2, T2 = my_data[:, 0], doken_data[:, 1]

# === Plot ===
#plt.figure(figsize=(8, 5))

#plt.plot(t1, T1, label="My Code", marker='o')
#plt.plot(t2, T2, label="Doken_Code")
#plt.xlabel("Time [s]")
#plt.ylabel("Temperature")
#plt.title("Validation Comparison")
#plt.grid(True)
#plt.legend()
#plt.show()

#===================================================
#     VALIDATION WITH THE ANALYTICAL SOLUTION
#===================================================

# In order to validate the solver with the analytical solution, someone needs to select an analytical expression of u that satistfies the PDE
# and then substitute it on order to find the source term f and the initial and boundary condition of this thing by extrapolating the solution 
# at every point

from petsc4py import PETSc
from mpi4py import MPI
import ufl
import numpy as np
import matplotlib.pyplot as plt
from dolfinx import mesh, fem
from solvers import heatdiff_implicit_solver, heatdiff_explicit_solver  # Import your reusable solver!
from dolfinx.fem.petsc import set_bc


# Simulation parameters


# Mesh and function space
nx, ny = 100, 100
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Exact solution class
class ExactSolution:
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t
    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t

class ManufacturedSource:
    def __init__(self, alpha, beta):
        self.value = beta - 2 - 2 * alpha
    def __call__(self, x, t):
        return np.full(x.shape[1], self.value)
    
alpha = 3
beta = 1.2
t = 0
T = 2.0
num_steps = 7000
dt = (T - t) / num_steps

domain = mesh.create_unit_square(MPI.COMM_WORLD, 20, 20, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

u_exact = ExactSolution(alpha, beta, t)
initial_condition = u_exact

source_term = ManufacturedSource(alpha, beta)

material_params = {"rho": 1.0, "Cp": 1.0, "k_therm": 1.0}
time_params = {"t_end": T, "Nsteps": num_steps}

T_D = fem.Function(V)
T_D.interpolate(u_exact)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(T_D, fem.locate_dofs_topological(V, fdim, boundary_facets))
bcs = [bc]

# Solver call (returns final solution as T_sol)
time_series, center_temperature, T_sol = heatdiff_implicit_solver(
    domain, V, bcs,
    material_params,
    time_params,
    initial_condition,
    source_term,
    output_dir="./results",
    output_filename="validation.xdmf",
    bc_update_func=T_D,     # This is a dolfinx Function
    source_bc=u_exact       # This is the callable exact class
)

# Update analytical (exact) solution to final time
u_exact.t = T
T_D.interpolate(u_exact)  # T_D holds the exact solution at final time

#  T_sol is already the full numerical solution at t = T
T_num = T_sol

# L2 error
error_L2 = np.sqrt(domain.comm.allreduce(
    fem.assemble_scalar(fem.form((T_num - T_D)**2 * ufl.dx)), op=MPI.SUM))

# Max error
error_max = domain.comm.allreduce(
    np.max(np.abs(T_num.x.array - T_D.x.array)), op=MPI.MAX)

if domain.comm.rank == 0:
    print(f"L2-error: {error_L2:.2e}")
    print(f"Max error: {error_max:.2e}")

center_exact = []
for t in time_series:
    u_exact.t = t
    val = u_exact(np.array([[0.5], [0.5]]))  # center
    center_exact.append(val[0])


# Plot the temperature 
import matplotlib.pyplot as plt

# Plot center temperature comparison
if domain.comm.rank == 0:
    plt.figure(figsize=(8, 5))
    plt.plot(time_series, center_temperature, marker='o', label="Numerical (center)")
    plt.plot(time_series, center_exact, marker='x', label="Exact (center)")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature at center (x=0.5, y=0.5)")
    plt.title("Center Temperature Comparison (Numerical vs. Analytical)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
