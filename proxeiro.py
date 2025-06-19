import numpy as np
import matplotlib.pyplot as plt

# --- Material and laser parameters (match your FEniCS setup) ---
T0 = 293.0                     # Ambient temperature [K]
P = 70.0                       # Laser power [W]
A = 0.15                       # Absorptivity
k = 15.0                       # Thermal conductivity [W/m·K]
rho = 7850.0                   # Density [kg/m³]
Cp = 500.0                     # Specific heat [J/kg·K]
v = 0.1                        # Scan speed [m/s]
alpha = k / (rho * Cp)        # Thermal diffusivity [m²/s]

q = 2 * A * P / (np.pi * (120e-6)**2)  # Peak flux from Gaussian laser [W/m²]

# --- Point to evaluate: center of domain ---
x_obs = 0.0  # x = 0
y_obs = 0.004  # y = 0.004 (top center point)

# --- Rosenthal solution function (steady-state) ---
def rosenthal_temperature(t):
    x_rel = x_obs - v * t
    R = np.sqrt(x_rel**2 + y_obs**2)
    temp = T0 + q / (2 * np.pi * k * R) * np.exp(-v * (x_rel + R) / (2 * alpha))
    return temp

# --- Time evaluation ---
t_series = np.linspace(0, 0.08, 1000)  # Match your simulation timespan
T_rosenthal = np.array([rosenthal_temperature(t) for t in t_series])

# --- Plot for comparison ---
plt.figure(figsize=(8, 5))
plt.plot(t_series, T_rosenthal, label="Rosenthal (analytical)", color='tab:red')
plt.xlabel("Time [s]")
plt.ylabel("Temperature [K]")
plt.title("Temperature at Domain Center: Simulation vs. Rosenthal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
