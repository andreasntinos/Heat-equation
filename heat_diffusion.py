##### HEAT CONDUCTION EQUATION ##### 
# 12.06.2025

# This is the first and simpliest case of Heat Conduction, where the diffusion equation is being solved in a 2D rectangle with 
# i. zero temperature boundary conditions are applied in the left and right facet and a gaussian initial condition applied on the domain
# The solver for the heat conduction problem is built without taking into consideration Neumann boundary conditions or external heat source. 
# Under these conditions, the problem is linear and  is solved using the LinearSolver of dolfinx. The results are validated though the corresponding code of Doken 
# "https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html"

#===================================================
#                     LIBRARIES
#===================================================

import numpy as np
import ufl # symbolic algerba library for the solution of PDEs- the variational formulation of the problems happens through UFL
from mpi4py import MPI  # library for paraller computing, allow processes to communicate and coordinate

from dolfinx import mesh, fem, io #dolfinx is the library that i) will produce the mesh of our geometry, and 
                                                             # ii) will provide the solver for the linear problem 

from dolfinx.fem.petsc import LinearProblem 
from petsc4py import PETSc
from ufl import TestFunction, TrialFunction, Measure, dot, grad, sin
from petsc4py.PETSc import ScalarType

from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
import matplotlib.pyplot as plt # library for the post porcessing and plotting of the temperature profiles
import os # python module to interact with the operating system

#===================================================
#                 GEOMETRY AND MESH
#===================================================

mesh_comm = MPI.COMM_WORLD # the parallel communicator
length, height = 2.0, 2.0  # the dimensions of the geometry
Nx, Ny = 50, 50            # the number of finite elements used in this problem

domain = mesh.create_rectangle(mesh_comm, [np.array([-2.0, -2.0]), np.array([2.0, 2.0])],
                               [Nx, Ny],
                               mesh.CellType.triangle)

Vt = fem.functionspace(domain, ("Lagrange", 1)) # a scalar type function space where Lagrange FE are defined, 
                                                # with polynomial shape function of 1st order

out_file1 = "heat_diffusion_imlpicit.xdmf" # we create an xdmf file where the mesh of the geometry will be saved 
                                 # later, in the time scheme, the solution of T at each time step will be saved in this xdmf file 

out_file2 = "heat_diffusion_explicit.xdmf" 

script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints" # the path where the xdmf files are saved (changes accordingly)
os.makedirs(output_dir, exist_ok=True)

out_file = os.path.join(output_dir, "heat_diffusion.xdmf")

with io.XDMFFile(domain.comm, out_file, "w") as xdmf:
    xdmf.write_mesh(domain)

## INITIAL CONDITION 
# use the exponential function of the Gaussian distribution

def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

## BOUNDARY CONDITIONS 
# detect the boundary facets of the geometry
def top(x):
    return np.isclose(x[1], 2.0) # np.isclose(a,b) is a command in numpy that returns true when a and b are equal

def bottom(x):
    return np.isclose(x[1], -2.0)

def right(x):
    return np.isclose(x[0], 2.0)

def left(x):
    return np.isclose(x[0], -2.0)

# now detect the dofs of the nodes that touch the boundaries we set above
top_dofs = fem.locate_dofs_geometrical(Vt, top)
bottom_dofs = fem.locate_dofs_geometrical(Vt, bottom) # Not used in current BCs, but good to have
right_dofs = fem.locate_dofs_geometrical(Vt, right)
left_dofs = fem.locate_dofs_geometrical(Vt, left)

bcs = [fem.dirichletbc(ScalarType(0.0), left_dofs, Vt),
       fem.dirichletbc(ScalarType(0.0), right_dofs, Vt),
       fem.dirichletbc(ScalarType(0.0), top_dofs, Vt),
       fem.dirichletbc(ScalarType(0.0), bottom_dofs, Vt)
       ]

# Material parameters: for the case of validation we set these quantities equal to zero 
material_params = {"rho": 1.0, "Cp": 1.0, "k_therm": 1.0}
time_params = {"t_end": 1.0, "Nsteps": 2000}
# Initial condition
T_n = fem.Function(Vt)
T_n.name = "Temperature"
T_n.interpolate(initial_condition) # set the initial condition to follow the Gaussian distribution

f_const = fem.Constant(domain, PETSc.ScalarType(0.0))

source_term = f_const

#===================================================
#               VARIATIONAL FORMULATION
#===================================================

from solvers import heatdiff_implicit_solver, heatdiff_explicit_solver

time, temp1, _ = heatdiff_implicit_solver(
   domain, Vt, bcs,
    material_params,
   time_params,
    initial_condition,
    source_term,
    output_dir,
    out_file1
)

#np.savetxt("heat_diffusion.txt", np.column_stack((time, temp1)))


time2, temp2, _ = heatdiff_explicit_solver(
    domain, Vt, bcs, 
    material_params, 
    time_params,
    initial_condition,
    source_term,
    output_dir,
    out_file2
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(time, temp1, label="Implicit", marker='o', linewidth=2)
plt.plot(time2, temp2, label="Explicit", marker='x', linestyle='--', linewidth=2)
plt.xlabel("Time [s]")
plt.ylabel("Temperature at center")
plt.title("Heat Diffusion: Implicit vs Explicit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


diff = np.abs(np.array(temp2) - np.array(temp1))
print(f"Max difference at center: {np.max(diff):.2e}")
