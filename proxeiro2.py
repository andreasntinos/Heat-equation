# 23.07.2024 - 3D transient heat solver with moving laser + Neumann BC (explicit)

#===============================
#          LIBRARIES           #
#===============================

from mpi4py import MPI
import numpy as np
import math
import os
import time
from petsc4py import PETSc

from dolfinx import fem, io
from ufl import TestFunction, grad, dot, dx, ds, sqrt, exp, conditional, lt, gt
from dolfinx.fem.petsc import assemble_vector, set_bc
from dolfinx.fem import Constant, Function, locate_dofs_topological


#===============================
#          LOAD MESH           #
#===============================

mesh_path = "/home/ntinos/Documents/FEnics/heat equation/checkpoints/Dogbone3D.xdmf"
output_dir = "results_explicit_evaporation_latent"
output_filename = "laser_evaporation_latent_explicit2.xdmf"

with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
    domain = xdmf.read_mesh(name="dogbone_mesh")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    facet_tags = xdmf.read_meshtags(domain, name="facet_tags")

#===============================
#         GEOMETRY SETUP       #
#===============================

unit = 0.001
gauge_radius = 5 * unit / 2
x_gauge_half = 8 * unit / 2
delta_half_width = 13.5 * unit / 2 - gauge_radius
dx_fillet = math.sqrt(delta_half_width * (2 * 13.5 * unit - delta_half_width))
x_fillet_end = x_gauge_half + dx_fillet

#===============================
#        FUNCTION SPACE        #
#===============================

Vt = fem.functionspace(domain, ("Lagrange", 1))
T_room = 293.0

def initial_condition(x): return np.full(x.shape[1], T_room, dtype=PETSc.ScalarType)

T_n = Function(Vt)
T_n.name = "Temperature"
T_n.interpolate(initial_condition)

T_test = TestFunction(Vt)

#===============================
#       DIRICHLET BOUNDARY     #
#===============================

fdim = domain.topology.dim - 1
left_dofs = locate_dofs_topological(Vt, fdim, facet_tags.indices[facet_tags.values == 2])
right_dofs = locate_dofs_topological(Vt, fdim, facet_tags.indices[facet_tags.values == 3])
bcs = [
    fem.dirichletbc(PETSc.ScalarType(T_room), left_dofs, Vt),
    fem.dirichletbc(PETSc.ScalarType(T_room), right_dofs, Vt),
]

#===============================
#        MATERIAL PARAMS       #
#===============================

material = {
    "rho": 7800.0,
    "Cp": 500.0,
    "k_therm": 15.0,
    "T_solidus": 1674.0,
    "T_liquidus": 1697.0,
    "latent_heat_fusion": 2.67e5  # J/kg
}

rho = material["rho"]
Cp = material["Cp"]
k = material["k_therm"]
alpha = k / (rho * Cp)

#===============================
#         TIME SETTINGS        #
#===============================

time_params = {"t_end": 0.2, "Nsteps": 50000, "h_fine": 0.00002}
dt = time_params["t_end"] / time_params["Nsteps"]
dt1 = 3.0 * dt  # Adjusted time step below the solidus temperature
dt2 = 0.01 * dt  # Further adjusted time step bove the solidus temperature

CFL = time_params["h_fine"] ** 2 / (2 * alpha)
print(f"Thermal diffusivity α = {alpha:.3e} m²/s")
print(f"CFL condition requires dt < {CFL:.6e}, using dt = {dt:.6e}")

#===============================
#          MASS        #
#===============================

one = Function(Vt)
one.interpolate(lambda x: np.ones_like(x[0]))
mass_form = fem.form(rho * Cp * one * T_test * dx)
M_vec = assemble_vector(mass_form)
M_vec.assemble()
M_inv = 1.0 / M_vec.array

#===============================
#         LASER HEAT SOURCE    #
#===============================

class MovingLaser:
    def __init__(self, params):
        self.A = params["Absorptivity"]
        self.P = params["Power"]
        self.R = params["Radius"]
        self.v = params["Scan_speed"]
        self.y0 = params["y0"]
        self.peak = 2 * self.A * self.P / (2 * np.pi * self.R ** 2)

    def __call__(self, x, t):
        x0 = -x_gauge_half + self.v * t
        r2 = (x[0] - x0)**2 + (x[1] - self.y0)**2 + x[2]**2
        return self.peak * np.exp(-2 * r2 / self.R**2)

laser_params = {
    "Absorptivity": 0.30,
    "Power": 200.0,
    "Radius": 60e-6,
    "Scan_speed": 0.8,
    "y0": gauge_radius
}
laser = MovingLaser(laser_params)
g_fun = Function(Vt)

# --- Radiation constants ---
sigma_sb = Constant(domain, PETSc.ScalarType(5.670374419e-8))  # Stefan–Boltzmann constant
T_amb = Constant(domain, PETSc.ScalarType(T_room))             # Ambient temp (room)
emissivity = Constant(domain, PETSc.ScalarType(0.35))           # Material emissivity (adjust as needed)


#===============================
#      EVAPORATION TERM        #
#===============================

Rv = Constant(domain, PETSc.ScalarType(150.774))
PA_atm = Constant(domain, PETSc.ScalarType(101325.0))
delta_Hv = Constant(domain, PETSc.ScalarType(7.416e6))
T_boil = Constant(domain, PETSc.ScalarType(3134.0))
prefactor = 0.01 * PA_atm * delta_Hv / sqrt(2 * np.pi * Rv * T_boil)
Q_evap_expr = prefactor * exp((delta_Hv / (Rv * T_boil)) * (1.0 - T_boil / T_n))

#===============================
#      LATENT HEAT TERM        #
#===============================

T_solidus = Constant(domain, PETSc.ScalarType(material["T_solidus"]))
T_liquidus = Constant(domain, PETSc.ScalarType(material["T_liquidus"]))
L_fusion = material["latent_heat_fusion"]

T_sol = Function(Vt)
X_melt = Function(Vt)
X_melt_prev = Function(Vt)
X_melt_prev.interpolate(lambda x: np.zeros(x.shape[1], dtype=PETSc.ScalarType))
dXdt = Function(Vt)

def melt_fraction(T):
    return conditional(lt(T, T_solidus), PETSc.ScalarType(0.0),
        conditional(gt(T, T_liquidus), PETSc.ScalarType(1.0),
            (T - T_solidus) / (T_liquidus - T_solidus)
        )
    )

#===============================
#        RHS FORMULATION       #
#===============================

ds_laser = ds(domain=domain, subdomain_data=facet_tags, subdomain_id=5)

Q_rad = emissivity * sigma_sb * (T_n**4 - T_amb**4)
L_form_expr = (
    -k * dot(grad(T_n), grad(T_test)) * dx
    + g_fun * T_test * ds_laser
    - Q_evap_expr * T_test * ds_laser
    - Q_rad * T_test * ds_laser
    - rho * L_fusion * dXdt * T_test * dx
)

rhs_form = fem.form(L_form_expr)

#===============================
#           OUTPUT SETUP       #
#===============================

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_filename)

with io.XDMFFile(domain.comm, output_path, "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(T_n, 0.0)

#===============================
#         PROBE MONITORING     #
#===============================

def probe_point(x, tol=1e-8):
    return (
        (np.abs(x[0] - 0.004) < tol) &
        (np.abs(x[1] - 0.0025) < tol) &
        (np.abs(x[2]) < tol)
    )

top_dof = fem.locate_dofs_geometrical(Vt, probe_point)
top_temp, time_series = [], []
T_max = -np.inf
start_time = time.time()

#===============================
#          TIME LOOP           #
#===============================

t = 0.0
n = 0
t_end = time_params["t_end"]

with io.XDMFFile(domain.comm, output_path, "a") as xdmf:
    while t < t_end:
        # Choose time step based on temperature at probe (or global max)
        T_probe = T_n.x.array[top_dof[0]] if len(top_dof) > 0 else T_room
        if T_probe < material["T_solidus"]:
            dt_current = dt1
        else:
            dt_current = dt2

        # Prevent overshooting final time
        if t + dt_current > t_end:
            dt_current = t_end - t

        t += dt_current
        n += 1

        # Laser interpolation
        g_fun.interpolate(lambda x: laser(x, t).astype(PETSc.ScalarType))

        # Update melt fraction and latent heat rate
        T_sol.x.array[:] = T_n.x.array
        X_melt_prev.x.array[:] = X_melt.x.array
        X_melt.interpolate(fem.Expression(melt_fraction(T_sol), Vt.element.interpolation_points()))
        dXdt.x.array[:] = (X_melt.x.array - X_melt_prev.x.array) / dt_current

        # RHS assembly
        b = assemble_vector(rhs_form)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)

        # Explicit update
        start, end = Vt.dofmap.index_map.local_range
        owned = np.arange(start, end, dtype=np.int32)
        T_n.x.array[owned] += dt_current * b.array[owned] * M_inv
        T_n.x.scatter_forward()

        for bc in bcs:
            bc.set(T_n.x.array, None)

        # Monitor probe temperature
        if len(top_dof) > 0:
            T_val = T_n.x.array[top_dof[0]]
            top_temp.append(T_val)
            time_series.append(t)
            if n % 100 == 0 or t >= t_end:
                print(f"Step {n:05d} | Time: {t:.6f} s | T_center: {T_val:.2f} K | dt = {dt_current:.2e}")

        if n % 10 == 0 or t >= t_end:
            xdmf.write_function(T_n, t)


if MPI.COMM_WORLD.rank == 0:
    #txt_path = os.path.join(output_dir, "nonlin_h0.02.txt")
    txt_path = os.path.join(output_dir, "nonlin_all_dynamic_tstep.txt")
    with open(txt_path, "w") as f:
        f.write("# Time [s] \t Temperature [K]\n")
        for t_val, T_val in zip(time_series, top_temp):
            f.write(f"{t_val:.6e}\t{T_val:.6f}\n")
    print(f"Saved probe data to: {txt_path}")

elapsed_time = time.time() - start_time
print(f"\nExplicit simulation complete. {len(top_temp)} steps recorded.")
print(f"Maximum temperature reached: {np.max(top_temp):.2f} K")
print(f"Total CPU time: {elapsed_time:.2f} seconds")

#===============================
#            PLOT              #
#===============================

if MPI.COMM_WORLD.rank == 0:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(time_series, top_temp, label="T at probe point", lw=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature at Probe Point with Latent Heat")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "T_vs_time_probe_point_latent.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Saved plot to: {plot_path}")
