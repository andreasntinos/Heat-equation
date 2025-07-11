# 01.07.2025
# the following code is a post-rocessing code of the Laser simulations on the 3D dogbone specimen. It contains the :
# a. size convergence analysis
# b. plots of the effect of laser power and speed to the temperature of a point at the end of the gauge area.

# =========================================================
#        L2 error plot for the simple diffusion           #
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# Corrected data: both arrays must have the same length
h = np.array([0.1, 0.05, 0.02, 0.014, 0.0083, 0.0066])
L2_implicit = np.array([7.08e-3, 1.77e-3, 2.85e-4, 1.46e-4, 5.01e-5, 3.23e-5])

# Log–Log plot
# the theoretical line is: logE = logC + 2log(h), however here we do not know C, so we choose such as the line passes through one of the points
plt.figure(figsize=(6, 5))
plt.loglog(h, L2_implicit, 'o-', label="L2 implicit", linewidth=2, markersize=6)

# Reference line for 2nd order convergence
slope_ref = h**2 * (L2_implicit[0] / h[0]**2)
plt.loglog(h, slope_ref, 'k--', label=r'O($h^2$) reference')

# Labels and style
plt.xlabel('Mesh size $h$')
plt.ylabel(r'$L^2$ error')
plt.title('Mesh Convergence Plot')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# =========================================================
#                 LASER 3D POST PROCESSING                #
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import os

data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/txt"
# Change working directory
os.chdir(data_folder)

## === Load the data (Mesh Size Effect) ===
# Ensure these files exist in the same directory as your script
data1_mesh = np.loadtxt("mesh_size0.1_3D_0.16.txt", comments="#", delimiter=None)
data2_mesh = np.loadtxt("mesh_size0.07_3D_0.16.txt", comments="#", delimiter=None)
data3_mesh = np.loadtxt("mesh_size0.05_3D_0.16.txt", comments="#", delimiter=None)
data4_mesh = np.loadtxt("mesh_size0.02_3D_0.16.txt", comments="#", delimiter=None)
data5_mesh = np.loadtxt("mesh_size0.01_3D_0.16.txt", comments="#", delimiter=None)

time1_mesh = 1000 * data1_mesh[:, 0]
time2_mesh = 1000 * data2_mesh[:, 0]
time3_mesh = 1000 * data3_mesh[:, 0]
time4_mesh = 1000 * data4_mesh[:, 0]
time5_mesh = 1000 * data5_mesh[:, 0]

# Original temperature data for Mesh Size Effect
temperature1_mesh_orig = data1_mesh[:, 1]
temperature2_mesh_orig = data2_mesh[:, 1]
temperature3_mesh_orig = data3_mesh[:, 1]
temperature4_mesh_orig = data4_mesh[:, 1]
temperature5_mesh_orig = data5_mesh[:, 1]

# === Normalize temperatures for Mesh Size Effect plot (0-1 range) ===
# Combine all temperature data to find global min and max for consistent normalization
all_temperatures_mesh = np.concatenate([
    temperature1_mesh_orig,
    temperature2_mesh_orig,
    temperature3_mesh_orig,
    temperature4_mesh_orig,
    temperature5_mesh_orig
])

min_temp_mesh = np.min(all_temperatures_mesh)
max_temp_mesh = np.max(all_temperatures_mesh)

# Apply min-max normalization
temperature1_mesh_norm = (temperature1_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
temperature2_mesh_norm = (temperature2_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
temperature3_mesh_norm = (temperature3_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
temperature4_mesh_norm = (temperature4_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
temperature5_mesh_norm = (temperature5_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)

# === Plotting Temperature vs. Time (Normalized) for Mesh Size Effect ===
plt.figure(figsize=(6, 5))

plt.plot(time1_mesh, temperature1_mesh_norm, '^-', color='tab:blue', lw=1.2, markersize=6, markevery=50, label="h_fine = 0.0001")
plt.plot(time2_mesh, temperature2_mesh_norm, 'o-', color='tab:orange', lw=1.2, markersize=6, markevery=50, label="h_fine = 0.00007")
plt.plot(time3_mesh, temperature3_mesh_norm, 'v-', color='tab:green', lw=1.2, markersize=6, markevery=50, label="h_fine = 0.00005")
plt.plot(time4_mesh, temperature4_mesh_norm, 's-', color='tab:red', lw=1.2, markersize=6, markevery=500, label="h_fine = 0.00002")
plt.plot(time5_mesh, temperature5_mesh_norm, 'D-', color='tab:purple', lw=1.2, markersize=6, markevery=5000, label="h_fine = 0.00001")

plt.xlabel("Time [ms]")
plt.ylabel("Temperature")
plt.title("Centerline temperature evolution (Mesh Study)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Extract final probe temperatures (using original for consistency with original script's intent) ===
# Note: These will be the original final temperatures, not normalized, as the subsequent plots
# (CPU time, Tmax/Tfinal vs Mesh Size) typically use absolute values.
final_Ts_mesh = np.array([
    temperature1_mesh_orig[-1],
    temperature2_mesh_orig[-1],
    temperature3_mesh_orig[-1],
    temperature4_mesh_orig[-1],
    temperature5_mesh_orig[-1]
])
print("Final probe temperatures (original values for mesh study):", final_Ts_mesh)

# === Define mesh sizes ===
mesh_sizes = np.array([0.0001, 0.00007, 0.00005, 0.00002, 0.00001])

# === Define CPU times: must match mesh_sizes length ===
cpu_times = np.array([8.37, 22.14, 25.36, 474.77, 9370.0])  

# === Compute error vs finest mesh ===
ref_T_mesh = final_Ts_mesh[-1]
error_mesh = np.abs(final_Ts_mesh - ref_T_mesh)

# === Max and final temperatures [K] (Original values for these plots) ===
T_max_orig = np.array([1770.69, 1898.17, 1980.38, 2109.48, 2116.95])
T_final_orig = np.array([315.22, 314.27, 314.6, 313.81, 313.91])

# === Normalize T_max and T_final for plotting (0-1 range) ===
all_T_values_for_mesh_plot = np.concatenate([T_max_orig, T_final_orig])
min_T_mesh_plot = np.min(all_T_values_for_mesh_plot)
max_T_mesh_plot = np.max(all_T_values_for_mesh_plot)

T_max_norm = (T_max_orig - min_T_mesh_plot) / (max_T_mesh_plot - min_T_mesh_plot)
T_final_norm = (T_final_orig - min_T_mesh_plot) / (max_T_mesh_plot - min_T_mesh_plot)

# === 1 CPU Time vs Mesh Size ===
plt.figure(figsize=(6, 5))
plt.loglog(mesh_sizes, cpu_times, 'o-', lw=2)
plt.xlabel("Mesh size [m]")
plt.ylabel("CPU time [s]")
plt.title("CPU Time vs Mesh Size")
plt.grid(True, which="both")
plt.tight_layout()
plt.show()
print(f" CPU vs mesh size plot saved: {os.path.abspath('cpu_vs_mesh_size_loglog.png')}")

# === 2 Tmax & Tfinal vs Mesh Size (Normalized) ===
plt.figure(figsize=(6, 5))
plt.semilogx(mesh_sizes, T_max_norm, 'o-', label="Normalized T_max", lw=2) # Use normalized data
plt.semilogx(mesh_sizes, T_final_norm, 's-', label="Normalized T_final", lw=2) # Use normalized data
plt.xlabel("Mesh size [m]")
plt.ylabel("Normalized Temperature") # Updated label
plt.title(" Peak & Final Probe Temperature vs Mesh Size") # Updated title
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.show()
print(f"Temperature vs mesh size plot saved: {os.path.abspath('temperature_vs_mesh_size_normalized.png')}")

# === Absolute & Relative errors for T_max and T_final (using original values) ===
# Use the finest (last) value as reference
T_max_ref = T_max_orig[-1]
T_final_ref = T_final_orig[-1]

# Compute absolute errors
error_Tmax = np.abs(T_max_orig - T_max_ref)
error_Tfinal = np.abs(T_final_orig - T_final_ref)

print("\n=== Absolute & Relative errors (Mesh Study) ===")
for h, e1, e2 in zip(mesh_sizes, error_Tmax, error_Tfinal):
    rel1 = (e1 / T_max_ref) * 100  # relative to finest Tmax
    rel2 = (e2 / T_final_ref) * 100  # relative to finest Tfinal
    print(f"h = {h:.5e} | ΔT_max = {e1:.3f} K ({rel1:.2f}%) | ΔT_final = {e2:.3f} K ({rel2:.2f}%)")

# ===================================================
#            Speed and Power Effect
# ===================================================

# === Load data (Speed Effect) ===
data1_speed = np.loadtxt("LaserP70_v0.1.txt")
data2_speed = np.loadtxt("LaserP70_v0.2.txt")
data3_speed = np.loadtxt("LaserP70_v0.5.txt")
data4_speed = np.loadtxt("LaserP70_v0.8.txt")
data5_speed = np.loadtxt("LaserP70_v1.0.txt") # This one is loaded but not plotted in the loop

time1_speed = 1000 * data1_speed[:, 0]  # ms
time2_speed = 1000 * data2_speed[:, 0]
time3_speed = 1000 * data3_speed[:, 0]
time4_speed = 1000 * data4_speed[:, 0]
time5_speed = 1000 * data5_speed[:, 0]

# Original temperature data for Speed Effect
temperature1_speed_orig = data1_speed[:, 1]
temperature2_speed_orig = data2_speed[:, 1]
temperature3_speed_orig = data3_speed[:, 1]
temperature4_speed_orig = data4_speed[:, 1]
temperature5_speed_orig = data5_speed[:, 1] # This one is loaded but not plotted in the loop

# === Normalize temperatures for Speed Effect plot (0-1 range) ===
# Combine only the temperatures that will be plotted to find global min and max
all_temperatures_speed_plot = np.concatenate([
    temperature1_speed_orig,
    temperature2_speed_orig,
    temperature3_speed_orig,
    temperature4_speed_orig
])

min_temp_speed = np.min(all_temperatures_speed_plot)
max_temp_speed = np.max(all_temperatures_speed_plot)

# Apply min-max normalization to the plotted temperatures
temperature1_speed_norm = (temperature1_speed_orig - min_temp_speed) / (max_temp_speed - min_temp_speed)
temperature2_speed_norm = (temperature2_speed_orig - min_temp_speed) / (max_temp_speed - min_temp_speed)
temperature3_speed_norm = (temperature3_speed_orig - min_temp_speed) / (max_temp_speed - min_temp_speed)
temperature4_speed_norm = (temperature4_speed_orig - min_temp_speed) / (max_temp_speed - min_temp_speed)


# === Create 1x4 subplots for Speed Effect ===
fig_speed, axes_speed = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

# Define titles for each scenario
titles_speed = [
    "Speed = 100 mm/s",
    "Speed = 200 mm/s",
    "Speed = 500 mm/s",
    "Speed = 800 mm/s"
]

# Plot each dataset in its own subplot with dark red color
for ax, time, temp, title in zip(
    axes_speed,
    [time1_speed, time2_speed, time3_speed, time4_speed],
    [temperature1_speed_norm, temperature2_speed_norm, temperature3_speed_norm, temperature4_speed_norm], # Use normalized data
    titles_speed
):
    ax.plot(time, temp, color="darkred", lw=2, label="P = 70 W")
    ax.set_title(title)
    ax.set_xlabel("Time [ms]")
    ax.grid(True)
    ax.legend()

axes_speed[0].set_ylabel("Normalized Temperature") # Updated label

plt.tight_layout()
plt.show()
print(f" Saved: {os.path.abspath('temperature_profiles_speeds_normalized.png')}")


# === Load data (Power Effect) ===
data1_power = np.loadtxt("Speedv0.1_P40.txt")
data2_power = np.loadtxt("Speedv0.1_P70.txt")
data3_power = np.loadtxt("Speedv0.1_P100.txt")
data4_power = np.loadtxt("Speedv0.1_P150.txt")
data5_power = np.loadtxt("Speedv0.1_P210.txt") # This one is loaded but not plotted in the loop

time1_power = 1000 * data1_power[:, 0]  # ms
time2_power = 1000 * data2_power[:, 0]
time3_power = 1000 * data3_power[:, 0]
time4_power = 1000 * data4_power[:, 0]
time5_power = 1000 * data5_power[:, 0] # This one is loaded but not plotted in the loop

# Original temperature data for Power Effect
temperature1_power_orig = data1_power[:, 1]
temperature2_power_orig = data2_power[:, 1]
temperature3_power_orig = data3_power[:, 1]
temperature4_power_orig = data4_power[:, 1]
temperature5_power_orig = data5_power[:, 1] # This one is loaded but not plotted in the loop

# === Normalize temperatures for Power Effect plot (0-1 range) ===
# Combine only the temperatures that will be plotted to find global min and max
all_temperatures_power_plot = np.concatenate([
    temperature1_power_orig,
    temperature2_power_orig,
    temperature3_power_orig,
    temperature4_power_orig
])

min_temp_power = np.min(all_temperatures_power_plot)
max_temp_power = np.max(all_temperatures_power_plot)

# Apply min-max normalization to the plotted temperatures
temperature1_power_norm = (temperature1_power_orig - min_temp_power) / (max_temp_power - min_temp_power)
temperature2_power_norm = (temperature2_power_orig - min_temp_power) / (max_temp_power - min_temp_power)
temperature3_power_norm = (temperature3_power_orig - min_temp_power) / (max_temp_power - min_temp_power)
temperature4_power_norm = (temperature4_power_orig - min_temp_power) / (max_temp_power - min_temp_power)


# === Create 1x4 subplots for Power Effect ===
fig_power, axes_power = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

# Define titles for each scenario (all same speed)
titles_power = [
    "Power = 40 W",
    "Power = 70 W",
    "Power = 100 W",
    "Power = 150 W"
]

# Define unique power labels for legend
power_labels = [
    "v=100 mm/s",
    "v=100 mm/s",
    "v=100 mm/s",
    "v=100 mm/s"
]

# Plot each dataset in its own subplot with dark red color and unique legend
for ax, time, temp, title, label in zip(
    axes_power,
    [time1_power, time2_power, time3_power, time4_power],
    [temperature1_power_norm, temperature2_power_norm, temperature3_power_norm, temperature4_power_norm], # Use normalized data
    titles_power,
    power_labels
):
    ax.plot(time, temp, color="darkred", lw=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time [ms]")
    ax.grid(True)
    ax.legend()

axes_power[0].set_ylabel("Normalized Temperature") # Updated label

plt.tight_layout()
plt.show()
print(f"Saved: {os.path.abspath('temperature_profiles_powers_normalized.png')}")
