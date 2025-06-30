# ===================================================
#              VALIDATION WITH ROSENTHAL SOLUTION
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -------------------------------
# Load FEM simulation data
# -------------------------------
file_path = "/home/ntinos/Documents/FEnics/heat_equation_3D/checkpoints/top_center_temperature.txt"
data = np.loadtxt(file_path, comments="#")
time_sim = data[:, 0]
temp_sim = data[:, 1]

# -------------------------------
# Rosenthal Analytical Solution
# -------------------------------
T0 = 293                # Ambient temperature [K]
rho = 7850              # Density [kg/m^3]
Cp = 500                # Specific heat [J/(kg K)]
A = 0.15                # Absorptivity
P = 150                 # Laser power [W]
k = 15                  # Thermal conductivity [W/(m K)]
alpha = k / (rho * Cp)  # Thermal diffusivity [m^2/s]
V = 0.1                 # Laser travel speed [m/s]
R = 130e-6

# Probe point
x0 = 0.04               # 40 mm from origin along laser path
y0 = 0.012               # Small offset to avoid singularity (0.1 mm)
z0 = 0.0                # No offset in z, can add if needed

# Time vector for Rosenthal calculation
t = np.linspace(0.0, 0.5, 240)

def rosenthal_temp_physical(t):
    xi = x0 - V * t
    y_prime = y0
    z_prime = z0
    r_squared = xi**2 + y_prime**2 + z_prime**2
    r = np.sqrt(r_squared)
    return T0 + (A * P / (2.0 * np.pi * k * r)) * np.exp(-V * (r + xi) / (2 * alpha))

# Evaluate Rosenthal solution
T_analytical = rosenthal_temp_physical(t)
# -------------------------------
# Plot FEM vs Rosenthal
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(t, T_analytical, label='Rosenthal Solution', color='darkred', lw=2)
#plt.plot(time_sim, temp_sim, label='FEM Simulation',  linestyle='--', color='blue', lw=2)

plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.title('Laser verification: FEM vs Rosenthal')
plt.xlim(0.3, 0.5)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("fem_vs_rosenthal_combined.png", dpi=300)
plt.show()


rosenthal_interp = interp1d(t, T_analytical, kind='linear', fill_value='extrapolate')

# Compute Rosenthal temperatures at FEM time steps
T_rosenthal_on_FEM = rosenthal_interp(time_sim)

# ---------------------------------------
# Compute absolute eror (percentage)
# ---------------------------------------

error_abs = np.abs(temp_sim - T_rosenthal_on_FEM)/temp_sim

MAE = np.mean(error_abs)

print(f"Mean Absolute Error (MAE): {MAE:.3f}%")

