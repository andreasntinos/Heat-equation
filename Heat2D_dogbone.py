##### HEAT CONDUCTION EQUATION ON DOGBONE #####
# 04.06.2025

#### The following code solves the heat diffusion equation on a 2D dogbone geometry. Thermal conductivity, heat capacity,
#### denisty and all the parameters of laser scanning are assumed to be constant with temperature, as also as the heat source term Q.
#### the laser source is added on the top surface of the geometry through Neumann boundary condition
#### This code is divided into 2 parts: i. the geometry generation through gmsh library, and the heat diffusion problem through FEniCS

#===========================================
#                 LIBRARIES
#===========================================
import numpy as np # library for mathematical computing in python (arrays, matrices, etc)
import ufl # symbolic algerba library for the solution of PDEs- the variational formulation of the problems is built through UFL

from dolfinx import geometry, fem, io #dolfinx is the library that provides the solver for our problem
from dolfinx.fem import locate_dofs_topological
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem 
from dolfinx.nls.petsc import NewtonSolver
from ufl import TestFunction, TrialFunction, Measure, dot, grad, SpatialCoordinate
from petsc4py.PETSc import ScalarType #need that to define the dirichlet boundary conditions
from dolfinx.geometry import BoundingBoxTree, compute_colliding_cells, compute_collisions_points
import matplotlib.pyplot as plt # library for the post processing and plotting of the temperature profiles
import math
import os 
from solvers import heatdiff_implicit_solver


#===========================================
#       MESH IMPORT & GEOMETRY VALUES
#===========================================
mesh_path = "/home/ntinos/Documents/FEnics/heat equation/checkpoints/Dogbone2D.xdmf"

# Corrected code
with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
    # Load the mesh using its default name "Grid"
    # Corrected code
    domain = xdmf.read_mesh(name="dogbone_mesh")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    
    # Load the tags using their default names
    cell_tags = xdmf.read_meshtags(domain, name="cell_tags")
    facet_tags = xdmf.read_meshtags(domain, name="facet_tags")


## GEOMETRY DIMENSIONS
gdim, fdim = 2, 1

unit = 0.001
gauge_length, gauge_diameter, end_diameter, end_length_straight = 8 * unit, 5 * unit, 13.5 * unit, 16 * unit
fillet_radius = 13.5 * unit

# Calculated dimensions needed for the laser path
gauge_radius, end_radius = gauge_diameter / 2.0, end_diameter / 2.0
delta_half_width = end_radius - gauge_radius
dx_fillet = math.sqrt(delta_half_width * (2 * fillet_radius - delta_half_width))

x_gauge_half = gauge_length / 2.0
x_fillet_end = x_gauge_half + dx_fillet

Vt = fem.functionspace(domain, ("Lagrange", 1)) # a scalar type function space where Lagrange FE are defined, with polynomial shape function of 1st order
T_sol = fem.Function(Vt, name="Temperature") # container in the FE space for the solution
T_room = 293 # all the temperatures are gonna be in Kelvin, and the outter temperature is the room temperture
left_side_tag, right_side_tag = 2, 3 # from the gmsh code, the IDs for the left and right facets (where Dirichlet bc are applied)
                                     # thesee fields MUST get the same ids with the left and right faces of the gmsh geometry

left_facets = facet_tags.indices[facet_tags.values == left_side_tag]
left_dofs = fem.locate_dofs_topological(Vt, fdim, left_facets)

# Correctly define dofs for the RIGHT side
right_facets = facet_tags.indices[facet_tags.values == right_side_tag]
right_dofs = fem.locate_dofs_topological(Vt, fdim, right_facets)

bcs = [fem.dirichletbc(ScalarType(T_room), left_dofs, Vt),
       fem.dirichletbc(ScalarType(T_room), right_dofs, Vt)]

material_params = {"rho": 8000.0, "Cp": 500.0, "k_therm": 15.0}

unit = 0.001
gauge_length, gauge_diameter, fillet_radius = 8 * unit, 5 * unit, 13.5 * unit
gauge_radius = gauge_diameter / 2.0
x_gauge_half = gauge_length / 2.0
delta_half_width = (13.5 * unit / 2.0) - gauge_radius
dx_fillet = math.sqrt(delta_half_width * (2 * fillet_radius - delta_half_width))
x_fillet_end = x_gauge_half + dx_fillet

t_end = 3.0
time_params = {"t_end": t_end, "Nsteps": 50}

def initial_condition(x):
    return np.full(x.shape[1], T_room, dtype=ScalarType)

# ===========================================
#         UFL LASER EXPRESSION
# ===========================================
A, P_laser, w0 = 0.5, 60.0, 100e-6

time_c = fem.Constant(domain, ScalarType(0.0))
x_coords = ufl.SpatialCoordinate(domain)
x_start_scan = -x_fillet_end
x0_t = x_start_scan + scan_v * time_c
y_center_fillet = gauge_radius + fillet_radius
y_fillet_r = y_center_fillet - ufl.sqrt(fillet_radius**2 - (x0_t - x_gauge_half)**2)
y_fillet_l = y_center_fillet - ufl.sqrt(fillet_radius**2 - (x0_t + x_gauge_half)**2)
y0_t = ufl.conditional(abs(x0_t) <= x_gauge_half, gauge_radius,
                       ufl.conditional(x0_t > x_gauge_half, y_fillet_r, y_fillet_l))
peak_flux = (2 * A * P_laser) / (np.pi * w0**2)
q_s = peak_flux * ufl.exp(-2 * ((x_coords[0] - x0_t)**2 + (x_coords[1] - y0_t)**2) / w0**2)

# ===========================================
#         CALLING THE SOLVER
# ===========================================



# Assuming the top surface has ID=4 from your meshing script.
top_surface_id = 4
ds = ufl.ds(domain=domain, subdomain_data=facet_tags)
ds_top = ds(top_surface_id)


# Create the list for the 'neumann_bcs' argument.

out_file = "Heat2D_dogbone.xdmf" 

script_dir = os.path.dirname(os.path.abspath(__file__))

output_dir = "/home/ntinos/Documents/FEnics/heat equation/checkpoints" # the path where the xdmf files are saved (changes accordingly)

# Define the heat flux 'g'. This could be constant or spatially varying
g = fem.Constant(domain, PETSc.ScalarType(1000.0))  # e.g., 1000 W/m^2

# Identify the top surface by facet tag ID (assuming ID 4 is the top)
top_surface_id = 4
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
ds_top = ds(top_surface_id)

# Package the Neumann condition as a list of (flux, measure) tuples
neumann_conditions = [(g, ds_top)]

# Call the solver
time_series, center_temp, T_final = heatdiff_implicit_solver(
    domain=domain,
    Vt=Vt,
    bcs=bcs,
    material_params=material_params,
    time_params=time_params,
    initial_condition=initial_condition,
    source_term=fem.Constant(domain, PETSc.ScalarType(0.0)),  # No body source
    output_dir=output_dir,
    output_filename=out_file,
    neumann_bcs=None  
)
