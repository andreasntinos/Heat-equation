## 13.06.2025
## this code serves as a validation of the solver with the analytical solution of the heat diffusion equation but also 
#  verified though the corresponding code of Doken  "https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html"


#===================================================
#                     LIBRARIES
#===================================================
import numpy as np  # Import NumPy for numerical operations, especially for loading data
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from petsc4py import PETSc  # Import PETSc, though not directly used in this plotting script
from mpi4py import MPI  # Import MPI, though not directly used in this plotting script
import ufl  # Import UFL, though not directly used in this plotting script
from dolfinx import mesh, fem  # Import DOLFINx mesh and fem components, though not directly used in this plotting script
from solvers import heatdiff_implicit_solver, heatdiff_explicit_solver, heatdiff_theta_solver  # Import your reusable solver functions
from dolfinx.fem.petsc import set_bc  # Import set_bc, though not directly used in this plotting script
import os  # Import os for interacting with the operating system (e.g., changing directories)

data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/txt"
# Change working directory
os.chdir(data_folder)

# The following code plots the results of a heat diffusion simulation, comparing
# results from the implicit solver, explicit solver, and a reference "Dokken" code.

# Load the simulation results from text files
my_data = np.loadtxt("heat_diffusion_implicit.txt")
my_data2 = np.loadtxt("heat_diffusion_explicit.txt")
doken_data = np.loadtxt("dokken_output.txt")

# Extract time and temperature data from the loaded arrays
t1, T1 = my_data[:, 0], my_data[:, 1]
t2, T2 = my_data[:, 0], doken_data[:, 1]
t3, T3 = my_data2[:, 0], my_data2[:, 1]

# === Plot ===
# Create a plot to visualize the temperature evolution over time

# Plot the results from your explicit solver (theta=0 refers to explicit Euler)
plt.figure(figsize=(5, 4))

# Plot your custom code's data
plt.plot(
    t1, T3, label="Implicit solver - theta=1", color='red',
    marker='o', markersize=6,
    markevery=50  # Only mark every 50th point
)

# Plot reference or comparison data
plt.plot(
    t2, T2, label="Dokken Code", color='black',
    linestyle='-', linewidth=2.5
)

plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("Temperature", fontsize=12)
plt.title("Simple Heat Diffusion Validation - T at (0, 0)", fontsize=14)
# Simpler grid customization
plt.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()

#===================================================
#     VALIDATION WITH THE ANALYTICAL SOLUTION
#===================================================

# In order to validate the solver with the analytical solution, someone needs to select an analytical expression of u that satistfies the PDE
# and then substitute it on order to find the source term f and the initial and boundary condition of this thing by extrapolating the solution 
# at every point. The solution is a manufactured one (we choose a polynomial for simplicity adn then substitute it to the original PDE and find the source, but also we find
# the initial and boundary conditions by settng the variables of time and space accordingly)

# ----------------------------
# Mesh and function space
# ----------------------------
nx, ny = 5, 5  # 
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# ----------------------------
# Exact solution
# ----------------------------
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

# Parameters
alpha = 3.0
beta = 1.2
t0 = 0.0
T = 2.0 # the final time 
num_steps = 2000
dt = (T - t0) / num_steps # the time step

u_exact = ExactSolution(alpha, beta, t0)
source_term = ManufacturedSource(alpha, beta)

material_params = {"rho": 1.0, "Cp": 1.0, "k_therm": 1.0}
time_params = {"t_end": T, "Nsteps": num_steps}

# ----------------------------
# Initial condition and BCs
# ----------------------------

T_D = fem.Function(V)
T_D.interpolate(u_exact)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
bc = fem.dirichletbc(T_D, fem.locate_dofs_topological(V, fdim, boundary_facets))
bcs = [bc]

print(f"CFL condition estimate: {(1/nx)**2:.2e}")

# ----------------------------
# Call the solver
# ----------------------------

time_series, center_temperature, T_sol = heatdiff_implicit_solver(
    domain, V, bcs,
    material_params,
    time_params,
    u_exact,         # initial condition
    source_term,     # source term
    output_dir="./results",
    output_filename="validation.xdmf",
    bc_update_func=T_D,     # Function for BCs during solve
    source_bc=u_exact       # Exact solution class to update BCs
)

#========================================================
#              POST PROCESS THE RESULTS                 #
#========================================================

# The convergence is examined thorugh the L2 and the maximum nodal error


# the L2 error cannot give accurate results with interpolation order 1, so we project the solution to P2 finite elements and caluclate the difference between the 
# numerical and exact solution there

# Update exact solution to final time
u_exact.t = T  # final time

# ----------------------------
# Create a higher-order FE space (P2)
# ----------------------------
# Create P2 space for L2 residual
# ----------------------------
V2 = fem.functionspace(domain, ("Lagrange", 2))

# Exact solution in P2
u_ex = fem.Function(V2)
u_ex.interpolate(u_exact)

# ----------------------------
# Project P1 numerical solution to P2
# ----------------------------
u = ufl.TrialFunction(V2)
v = ufl.TestFunction(V2)

a_proj = u * v * ufl.dx
L_proj = T_sol * v * ufl.dx

T_sol_P2 = fem.Function(V2)
proj_problem = fem.petsc.LinearProblem(a_proj, L_proj, u=T_sol_P2)
proj_problem.solve()

# ----------------------------
# L2 error: P2 vs P2
# ----------------------------
error_form = fem.form((T_sol_P2 - u_ex)**2 * ufl.dx)
local_error = fem.assemble_scalar(error_form)
error_L2 = np.sqrt(domain.comm.allreduce(local_error, op=MPI.SUM))

# ----------------------------
# Max nodal error: compare P1 vs P1 (same nodes)
# ----------------------------
T_exact_P1 = fem.Function(V)
T_exact_P1.interpolate(u_exact)

error_max = domain.comm.allreduce(
    np.max(np.abs(T_sol.x.array - T_exact_P1.x.array)),
    op=MPI.MAX
)

if domain.comm.rank == 0:
    print(f"\n--- Results ---")
    print(f"L2 error : {error_L2:.2e}")
    print(f"Max nodal error: {error_max:.2e}")


# Correct arrays (lengths match; fixed 2.56e-5)
h = np.array([0.1, 0.05, 0.02, 0.014, 0.0125, 0.0083, 0.0066, 0.005])
L2_implicit = np.array([7.08e-3, 1.77e-3, 2.85e-4, 1.46e-4, 7.20e-5, 5.01e-5, 3.23e-5, 2.56e-5])

plt.figure(figsize=(6, 5))
plt.loglog(h, L2_implicit, 'o-', label="L2 implicit", linewidth=2, markersize=6)

# Reference line for 2nd order, scaled to the first point
slope_ref = (L2_implicit[0] / h[0]**2) * h**2
plt.loglog(h, slope_ref, 'k--', label='O($h^2$) reference')

plt.xlabel('Mesh size $h$', fontsize=12)
plt.ylabel('L2 error', fontsize=12)
plt.title('Mesh Convergence Plot', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--')
plt.tight_layout()
plt.show()


#=======================================
#           MESH CONVERGENCE           #
#=======================================

# this is the plot of the mesh convergence to verify that the error follows the theoretical slope: 

# Mesh sizes (h)
h = np.array([0.1, 0.05, 0.02, 0.014, 0.0083, 0.0066])

# L2 errors for implicit and explicit solvers
L2_implicit = np.array([7.08e-3, 1.77e-3, 2.85e-4, 1.46e-4,  5.01e-5, 3.23e-5])
L2_explicit = np.array([6.95e-3, 1.70e-3, 2.80e-4, 1.43e-4,  4.95e-5, 3.20e-5])  

# Log–Log plot
plt.figure(figsize=(5, 4))
plt.loglog(h, L2_implicit, 'o-', label="Implicit Solver", linewidth=2, markersize=6)
plt.loglog(h, L2_explicit, 's--', label="Explicit Solver", linewidth=2, markersize=6)

# Reference line for 2nd order convergence
slope_ref = h**2 * (L2_implicit[0] / h[0]**2)
plt.loglog(h, slope_ref, 'k-.', label=r'O($h^2$) reference')

# Labels and styling
plt.xlabel('Mesh size $h$', fontsize=12)
plt.ylabel(r'$L^2$ error', fontsize=12)
plt.title('Mesh Convergence study', fontsize=14)
plt.legend(fontsize=10)
# Softer, grey grid
#plt.grid(True, which='both', linestyle='--', color='grey', alpha=0.5)

plt.tight_layout()
plt.show()