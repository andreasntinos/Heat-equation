# 01.07.2025
# This code is the simple laser testing for the code with the addition of the laser scanning under Neumann bc

from dolfinx import mesh, fem, io  # Import DOLFINx components for mesh, finite element methods, and I/O
from mpi4py import MPI             # Import MPI for parallel processing
import numpy as np                 # Import NumPy for numerical operations
import ufl                         # Import UFL for Unified Form Language
from petsc4py import PETSc         # Import PETSc for high-performance numerical computing
import os                          # Import os for operating system interactions (e.g., creating directories)
import matplotlib.pyplot as plt    # Import Matplotlib for plotting

# Output path
out_file = "lasertest_3D3.xdmf"
output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints"
os.makedirs(output_dir, exist_ok=True)

# Geometry parameters- the initial domai was 0.045 but it is reduces here only for the sake of mesh convergence analysis
gdim, fdim = 3, 2                # Geometric dimension (3D) and facet dimension (2D)
Lx, Ly, Lz = 0.015, 0.01, 0.001  # Dimensions of the rectangular domain (Length_x, Length_y, Length_z) in meters
Nx, Ny, Nz = 30, 30, 10          # Number of cells in each direction for the mesh resolution

# Create a rectangular box mesh
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
    [Nx, Ny, Nz],
    cell_type=mesh.CellType.tetrahedron,
)
Vt = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet Boundary Conditions (BCs) on all sides except the top (z = Lz)
# Define a function to identify boundary facets where Dirichlet BCs will be applied
def dirichlet_boundary(x):
    return (
        np.isclose(x[0], 0.0) | np.isclose(x[0], Lx) 
    )

# Locate facets on the specified boundaries
bc_facets = mesh.locate_entities_boundary(domain, fdim, dirichlet_boundary)
# Locate degrees of freedom (DoFs) associated with these facets
bc_dofs = fem.locate_dofs_topological(Vt, fdim, bc_facets)
T_room = 293.0  # Room temperature in Kelvin
bcs = [fem.dirichletbc(PETSc.ScalarType(T_room), bc_dofs, Vt)]

# Material + time params
material_params = {"rho": 7850.0, "Cp": 500.0, "k_therm": 15.0}
alpha = material_params["k_therm"] / (material_params["rho"] * material_params["Cp"])

# Calculate CFL (Courant-Friedrichs-Lewy) limits for stability of explicit schemes
# h^2 / (2 * alpha) where h is the characteristic mesh size in each direction
CFL_x = (Lx / Nx)**2 / alpha
CFL_y = (Ly / Ny)**2 / alpha
CFL_z = (Lz / Nz)**2 / alpha

# Print the most restrictive CFL condition
CFL_min = min(CFL_x, CFL_y, CFL_z)
print(f"CFL condition demands: dt < {CFL_min:.4e} s")

# Time step
time_params = {"t_end": 0.15, "Nsteps": 2000}
dt = time_params["t_end"] / time_params["Nsteps"]

# Check stability
if dt > CFL_min:
    raise ValueError(f"Time step too large for stability: dt = {dt:.4e} > CFL limit = {CFL_min:.4e}")

# Initial condition
def initial_condition(x): return np.full(x.shape[1], T_room)

# Source term
source_term = fem.Constant(domain, PETSc.ScalarType(0.0))

# Tag top surface (z = Lz) for Neumann BC (laser heat source)
from dolfinx.mesh import meshtags, locate_entities
# Locate facets on the top surface (where z is close to Lz)
top_facets = locate_entities(domain, fdim, lambda x: np.isclose(x[2], Lz))
# Create a numpy array of values (tag=1) for these facets
values = np.full(len(top_facets), 1, dtype=np.int32)
# Create a MeshTags object to store the facet tags
facet_tags = meshtags(domain, fdim, top_facets, values)
# Define a UFL measure for integration over the top surface with the specified tag
ds_top = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags, subdomain_id=1)
# Laser parameters for the moving heat source

laser_params = {
    "A": 0.15,     # absorptivity
    "P": 150.0,    # power (Watt)
    "R": 120e-6,   # beam radius
    "v": 0.1,      # laser speed
    "y0": Ly / 2,  # the point where the laser point in the t direction
    "z0": Lz       # the point where laser points int the z direction
}

class MovingLaser3D:
    def __init__(self, params):
        self.A = params["A"]     # Absorptivity
        self.P = params["P"]     # Power
        self.R = params["R"]     # Radius
        self.v = params["v"]     # Scan speed
        self.y0 = params["y0"]   # Y-offset of the scan line
        self.z0 = params["z0"]   # Z-offset of the scan line (should be Lz for top surface)
        self.t =  0.0            # Current time (will be updated during simulation)
        # Calculate the peak intensity of the Gaussian heat source
        self.peak = 2 * self.A * self.P / (np.pi * self.R**2)

    def __call__(self, x):
        x0 = self.v * self.t
        r2 = (x[0] - x0)**2 + (x[1] - self.y0)**2 + (x[2] - self.z0)**2
        return self.peak * np.exp(-2 * r2 / self.R**2)

laser = MovingLaser3D(laser_params) # Create an instance of the MovingLaser3D

# Check if the laser scans beyond the domain length
scanned_distance = laser.v * time_params["t_end"]
if scanned_distance > Lx:
    print(f"Warning: Scanned distance {scanned_distance:.4f} m exceeds domain length (Lx = {Lx} m).")

# Wrapper function for the laser heat source, to be used by DOLFINx
def g_laser(x, t):
    laser.t = t
    return laser(x).astype(PETSc.ScalarType)

# Define Neumann boundary conditions (list of (function, measure) tuples)
neumann_conditions = [(g_laser, ds_top)]

# Call solver
from solvers import heatdiff_explicit_solver, heatdiff_theta_solver

# Run the heat diffusion simulation using the explicit solver
time_series, center_temp, T_final = heatdiff_theta_solver(
    domain=domain,
    Vt=Vt,
    bcs=bcs,
    material_params=material_params,
    time_params=time_params,
    initial_condition=initial_condition,
    source_term=source_term,
    output_dir=output_dir,
    output_filename=out_file,
    theta = 0.0,
    neumann_bcs=neumann_conditions
)

plt.figure(figsize=(10, 6))
plt.title("Temperature at the Center Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (K)")
# Plot the temperature at the center over time
plt.plot(time_series, center_temp, label="Center Temperature")
plt.show()

# Save the temperature at the center over time to a text file
output_txt = os.path.join(output_dir, "top_center_temperature_120.txt")
np.savetxt(output_txt, np.column_stack((time_series, center_temp)),
           header="Time [s]    Temperature [K]", fmt="%.6e")
print(f"Saved temperature time series to {output_txt}")

