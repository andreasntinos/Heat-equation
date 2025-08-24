# 01.07.2025
# the following code is a post-rocessing code of the Laser simulations on the 3D dogbone specimen. It contains the :
# a. size convergence analysis
# b. plots of the effect of laser power and speed to the temperature of a point at the end of the gauge area.

# =========================================================
#                 LASER 3D POST PROCESSING                #
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import os


#data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/txt"
data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/results_linear_explicit"
# Change working directory
os.chdir(data_folder)


## === Load the data (Mesh Size Effect) ===
# Ensure these files exist in the same directory as your script
#data1_mesh = np.loadtxt("mesh_size0.08_3D_0.16.txt", comments="#", delimiter=None)
#data2_mesh = np.loadtxt("mesh_size0.05_3D_0.16.txt", comments="#", delimiter=None)
#ata3_mesh = np.loadtxt("mesh_size0.02_3D_0.16.txt", comments="#", delimiter=None)
#ata4_mesh = np.loadtxt("mesh_size0.01_3D_0.16.txt", comments="#", delimiter=None)
#data5_mesh = np.loadtxt("mesh_size0.005_3D_0.16.txt", comments="#", delimiter=None)

data1_mesh = np.loadtxt("lin_h0.03.txt", comments="#", delimiter=None)
data2_mesh = np.loadtxt("lin_h0.02.txt", comments="#", delimiter=None)
data3_mesh = np.loadtxt("lin_h0.015.txt", comments="#", delimiter=None)
data4_mesh = np.loadtxt("lin_h0.01.txt", comments="#", delimiter=None)
data5_mesh = np.loadtxt("lin_h0.005.txt", comments="#", delimiter=None)
data6_mesh = np.loadtxt("lin_h0.0025.txt", comments="#", delimiter=None)


time1_mesh = 1000 * data1_mesh[:, 0]
time2_mesh = 1000 * data2_mesh[:, 0]
time3_mesh = 1000 * data3_mesh[:, 0]
time4_mesh = 1000 * data4_mesh[:, 0]
time5_mesh = 1000 * data5_mesh[:, 0]
time6_mesh = 1000 * data6_mesh[:, 0]


# Original temperature data for Mesh Size Effect
temperature1_mesh_orig = data1_mesh[:, 1]
temperature2_mesh_orig = data2_mesh[:, 1]
temperature3_mesh_orig = data3_mesh[:, 1]
temperature4_mesh_orig = data4_mesh[:, 1]
temperature5_mesh_orig = data5_mesh[:, 1]
temperature6_mesh_orig = data6_mesh[:, 1]


# === Normalize temperatures for Mesh Size Effect plot (0-1 range) ===
# Combine all temperature data to find global min and max for consistent normalization
all_temperatures_mesh = np.concatenate([
    temperature1_mesh_orig,
    temperature2_mesh_orig,
    temperature3_mesh_orig,
    temperature4_mesh_orig,
    temperature5_mesh_orig,
    temperature6_mesh_orig,
   
])

#min_temp_mesh = np.min(all_temperatures_mesh)
#max_temp_mesh = np.max(all_temperatures_mesh)#

# Apply min-max normalization
#temperature1_mesh_norm = (temperature1_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
#temperature2_mesh_norm = (temperature2_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
#temperature3_mesh_norm = (temperature3_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
#emperature4_mesh_norm = (temperature4_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)
#temperature5_mesh_norm = (temperature5_mesh_orig - min_temp_mesh) / (max_temp_mesh - min_temp_mesh)

# === Plotting Temperature vs. Time for Mesh Size Effect ===
plt.figure(figsize=(6, 5))

plt.plot(time1_mesh, temperature1_mesh_orig, '^-', color='blue', lw=1.2, markersize=6, markevery=50, label="h_fine = 30 µm")
plt.plot(time2_mesh, temperature2_mesh_orig, 'o-', color='orange', lw=1.2, markersize=6, markevery=50, label="h_fine = 20 µm")
plt.plot(time3_mesh, temperature3_mesh_orig, 'v-', color='green', lw=1.2, markersize=6, markevery=50, label="h_fine = 15 µm")
plt.plot(time4_mesh, temperature4_mesh_orig, 's-', color='red', lw=1.2, markersize=6, markevery=50, label="h_fine = 10 µm")
plt.plot(time5_mesh, temperature5_mesh_orig, 'D-', color='deepskyblue', lw=1.2, markersize=6, markevery=50, label="h_fine = 5 µm")
plt.plot(time6_mesh, temperature6_mesh_orig, 'X-', color='tab:purple', lw=1.2, markersize=6, markevery=50, label="h_fine = 2.5 µm")



# === Final Touches ===
plt.xlabel("Time [ms]", fontsize=14)
plt.ylabel("Temperature [K]", fontsize=14)
plt.title("Centerline Temperature Evolution", fontsize=16)
plt.xlim(1.570, 1.7)
plt.grid(True, linestyle=':')
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# === Define correct mesh sizes corresponding to the 8 data files ===
mesh_sizes_full = np.array([30, 20, 15, 10, 5, 2.5])

# Extract peak temperatures from 6 datasets
T_max_values = np.array([
    np.max(temperature1_mesh_orig),
    np.max(temperature2_mesh_orig),
    np.max(temperature3_mesh_orig),
    np.max(temperature4_mesh_orig),
    np.max(temperature5_mesh_orig),
    np.max(temperature6_mesh_orig)
])

# === Plot: Peak Temperature vs Mesh Size ===
plt.figure(figsize=(5, 4))
plt.plot(mesh_sizes_full, T_max_values, 'o-', 
         markerfacecolor='white', 
         markeredgecolor='red', 
         markeredgewidth=1.5, 
         color='black')  # black line, red-outlined hollow markers

plt.xlabel("Mesh Size [µm]", fontsize=12)
plt.ylabel("Peak Temperature [K]", fontsize=12)
plt.title("Mesh Convergence: Peak Temperature", fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
plt.tight_layout()
plt.show()


# === Reference value and relative error computation ===
T_ref = T_max_values[-1]  # Use finest mesh result as reference
rel_error_percent = np.abs((T_max_values - T_ref) / T_ref) * 100

# === Plot: Relative Error (%) vs Mesh Size ===
plt.figure(figsize=(5, 4))
plt.plot(mesh_sizes_full, rel_error_percent, 'o-', 
         markerfacecolor='white', 
         markeredgecolor='red', 
         markeredgewidth=1.5, 
         color='black')  # black line, red-outlined hollow markers

plt.xlabel("Mesh Size [µm]", fontsize=12)
plt.ylabel("Relative Error in Peak Temperature [%]", fontsize=12)
plt.title("Mesh Convergence: Relative Error", fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='lightgray')
plt.tight_layout()
plt.show()



# ===================================================
#            Speed and Power Effect
# ===================================================


data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/txt"
# Change working directory
os.chdir(data_folder)

data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/txt"
# Change working directory
os.chdir(data_folder)

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
fig_speed, axes_speed = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

titles_speed = ["(a)", "(b)", "(c)", "(d)"]
speed_labels = ["v = 100 mm/s", "v = 200 mm/s", "v = 500 mm/s", "v = 800 mm/s"]

for ax, time, temp, title, label in zip(
    axes_speed,
    [time1_speed, time2_speed, time3_speed, time4_speed],
    [temperature1_speed_norm, temperature2_speed_norm, temperature3_speed_norm, temperature4_speed_norm],
    titles_speed,
    speed_labels
):
    ax.plot(time, temp, color="darkred", lw=2.5, label=label)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Time [ms]", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=14)

axes_speed[0].set_ylabel("Normalized Temperature", fontsize=18)

plt.tight_layout()
plt.show()

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
fig_power, axes_power = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

titles_power = ["(a)", "(b)", "(c)", "(d)"]
power_labels = ["P = 40 W", "P = 70 W", "P = 100 W", "P = 150 W"]

# Plot each dataset in its own subplot with enhanced styling
for ax, time, temp, title, label in zip(
    axes_power,
    [time1_power, time2_power, time3_power, time4_power],
    [temperature1_power_norm, temperature2_power_norm, temperature3_power_norm, temperature4_power_norm],
    titles_power,
    power_labels
):
    ax.plot(time, temp, color="darkred", lw=2.5, label=label)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Time [ms]", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=14)

# Shared Y-axis label on the first subplot
axes_power[0].set_ylabel("Normalized Temperature", fontsize=18)

plt.tight_layout()
plt.show()
