# 03.07.2025
# this code is the post process code of the validation of the numerical solution with the rosenthal equation 

import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from scipy.interpolate import interp1d  

# -------------------------------
# Load FEM simulation data
# -------------------------------
file_path = "/home/ntinos/Documents/FEnics/heat_equation_3D/checkpoints/top_center_temperature.txt"  # Define the path where the output of the numerics exists
data = np.loadtxt(file_path, comments="#")  # Load data from the specified file, ignoring lines starting with '#'
time_sim = data[:, 0]  # Extract time values from the first column of the loaded data
temp_sim = data[:, 1]  # Extract temperature values from the second column of the loaded data

# -------------------------------
# Rosenthal Analytical Solution
# -------------------------------
T0 = 293                # Ambient temperature [K]
rho = 7850              # Density [kg/m^3]
Cp = 500                # Specific heat [J/(kg K)]
A = 0.15                # Absorptivity (dimensionless)
P = 150                 # Laser power [W]
k = 15                  # Thermal conductivity [W/(m K)]
alpha = k / (rho * Cp)  # Thermal diffusivity [m^2/s] - a material property
V = 0.1                 # Laser travel speed [m/s]
R = 130e-6              # Laser beam radius (though not directly used in the point source Rosenthal solution here, often part of the problem context)

# Probe point for the analytical solution
# These coordinates define the fixed point in the material where temperature is being calculated
# Careful: Rosenthal equation trcks the points from the point where the laser is added, so in our case where we want to track the temperature at a point at the surface of the 
# domain, y ad z xoordinates need to be equal to zero and only the x xoordinate needs to be set properly (these coordinates differ from the prob point fo the solver because there
# the solver trackes from the bottom where the rectangular domain is created)
x0 = 0.04               # X-coordinate of the probe point (40 mm from origin along laser path)
y0 = 0.0                # Y-coordinate of the probe point (small offset to avoid singularity at r=0)
z0 = 0.0                # Z-coordinate of the probe point (no offset in z, can be adjusted if needed)

# Time vector for Rosenthal calculation
# The number of points chosen aims to limit numerical issues (e.g., singularities in the Rosenthal solution) 
# that can arise at very small time steps or when the probe point is coincident with the heat source.
t = np.linspace(0.0, 0.5, 240)  # Create a linearly spaced time array for analytical calculation

# Define the Rosenthal analytical solution function for a moving point heat source
def rosenthal_temp_physical(t):
    """
    Calculates the temperature at a fixed point (x0, y0, z0) due to a
    moving point heat source using the Rosenthal equation (for 3D steady-state,
    but applied here in a transient sense by changing relative position with time).
    """
    xi = x0 - V * t  # Relative coordinate in the direction of laser travel
    y_prime = y0     # Y-coordinate (perpendicular to travel, in the plane of the surface)
    z_prime = z0     # Z-coordinate (depth into the material, perpendicular to the surface)
    r_squared = xi**2 + y_prime**2 + z_prime**2  # Squared distance from the instantaneous laser position
    r = np.sqrt(r_squared)  # Distance from the instantaneous laser position
    # The Rosenthal equation for temperature
    return T0 + (A * P / (2.0 * np.pi * k * r)) * np.exp(-V * (r + xi) / (2 * alpha))

# Evaluate the Rosenthal solution over the defined time vector
T_analytical = rosenthal_temp_physical(t)

# -------------------------------
# Plot FEM vs Rosenthal
# -------------------------------
plt.figure(figsize=(8, 5))  # Create a new figure with a specified size
# Plot the Rosenthal analytical solution
plt.plot(t, T_analytical, label='Rosenthal Solution', color='darkred', lw=2)
# Plot the FEM simulation results
plt.plot(time_sim, temp_sim, label='FEM Simulation',  linestyle='--', color='blue', lw=2)

plt.xlabel('Time (s)')  # Set the x-axis label
plt.ylabel('Temperature (K)')  # Set the y-axis label
plt.title('Laser verification: FEM vs Rosenthal')  # Set the plot title
plt.xlim(0.3, 0.5)  # Set the x-axis limits to focus on a relevant time window

plt.grid(True)  # Display a grid on the plot
plt.legend()  # Display the legend to identify the plotted lines
plt.tight_layout()  # Adjust plot parameters for a tight layout
plt.savefig("fem_vs_rosenthal_combined.png", dpi=300)  # Save the plot to a PNG file
plt.show()  # Display the plot

# Interpolate the Rosenthal solution to match the time steps of the FEM simulation
rosenthal_interp = interp1d(t, T_analytical, kind='linear', fill_value='extrapolate')

# Compute Rosenthal temperatures at the exact time steps of the FEM simulation
T_rosenthal_on_FEM = rosenthal_interp(time_sim)

# ---------------------------------------
# Compute absolute error (percentage)
# ---------------------------------------

# Calculate the absolute percentage error at each time step
# (abs(FEM_temp - Rosenthal_temp) / FEM_temp)
error_abs = np.abs(temp_sim - T_rosenthal_on_FEM) / temp_sim

# Calculate the Mean Absolute Error (MAE)
MAE = np.mean(error_abs)

# Print the Mean Absolute Error as a percentage
print(f"Mean Absolute Error (MAE): {100*MAE:.3f}%")