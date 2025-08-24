# This code works as a first verification of the heat diffusion solver we built. The full tutorial is documented in the following :
#  https://jsdokken.com/dolfinx-tutorial/chapter2/diffusion_code.html

import matplotlib as mpl
import pyvista
import ufl
import numpy as np

from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc

# Temporal parameters
t = 0.0
T = 1.0
num_steps = 2000
dt = T / num_steps

# Mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])],
                               [nx, ny], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

# Initial condition
def initial_condition(x, a=5):
    return np.exp(-a * (x[0]**2 + x[1]**2))

u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Solution variable
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

# Variational problem
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
rho = fem.Constant(domain, PETSc.ScalarType(1.0))
Cp = fem.Constant(domain, PETSc.ScalarType(1.0))
k_therm = fem.Constant(domain, PETSc.ScalarType(1.0))

a = rho * Cp * u * v * ufl.dx + k_therm * dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = rho * Cp * (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

# ParaView output: safer with WITH blocks!
with io.XDMFFile(domain.comm, "dokken.diffusion.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh, t)

    # PyVista GIF
    pyvista.start_xvfb()
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
    plotter = pyvista.Plotter()
    plotter.open_gif("u_time.gif", fps=10)

    grid.point_data["uh"] = uh.x.array
    warped = grid.warp_by_scalar("uh", factor=1)

    viridis = mpl.colormaps.get_cmap("viridis").resampled(25)
    sargs = dict(title_font_size=25, label_font_size=20, fmt="%.2e", color="black",
                 position_x=0.1, position_y=0.8, width=0.8, height=0.1)

    renderer = plotter.add_mesh(warped, show_edges=True, lighting=False,
                                cmap=viridis, scalar_bar_args=sargs,
                                clim=[0, max(uh.x.array)])

    # Probe DOF
    center_dof = fem.locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    center_temperature = []
    time_series = []

    for i in range(num_steps):
        t += dt

        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        if len(center_dof) > 0:
            center_temperature.append(uh.x.array[center_dof[0]])
            time_series.append(t)

        u_n.x.array[:] = uh.x.array

        xdmf.write_function(uh, t)

        new_warped = grid.warp_by_scalar("uh", factor=1)
        warped.points[:, :] = new_warped.points
        warped.point_data["uh"][:] = uh.x.array
        plotter.write_frame()

    plotter.close()

# Save probe data
np.savetxt("dokken_output.txt", np.column_stack((time_series, center_temperature)))
