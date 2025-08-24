## 09/07/2025 ##
# this code is the post processing of the different laser scenaria, and the mean value of the cooling rates.

import os
import numpy as np
import matplotlib.pyplot as plt

#=============================
#        LINEAR PROBLEM      =
#=============================
data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/results_linear_explicit"
os.chdir(data_folder)

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Figure creation with extra vertical space

# Font/spacing settings
plt.rcParams.update({
    'axes.titlesize': 12,     # subplot titles
    'axes.titlepad': 12,      # space between title and plot
    'axes.labelsize': 14,     # BIGGER axis labels (x and y titles)
    'axes.labelpad': 10,      # padding between labels and ticks
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

T_threshold = 296  # Ambient temperature [K]

# ~10x12 in figure ≈ six panels that read like 5x4 in each
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.35)
axes = axes.flatten()

# Helper to style all subplots the same
def style_axes(ax, title, xlim):
    ax.set_title(title)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Temperature [K]")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(*xlim)

# === Scenario 1: P = 24 W ===
data1a = np.loadtxt("lin_Scenario1_P24_v0.15.txt")
data1b = np.loadtxt("lin_Scenario1_P24_v0.25.txt")
time1a, temp1a = 1000 * data1a[:, 0], data1a[:, 1]
time1b, temp1b = 1000 * data1b[:, 0], data1b[:, 1]

ax = axes[0]
ax.plot(time1a, temp1a, lw=2, color='red', label="v = 0.15 m/s")
ax.plot(time1b, temp1b, lw=2, color='black', label="v = 0.25 m/s")
ax.axhline(1700, color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold, color='black', linestyle=':', label='Ambient Temperature')
style_axes(ax, "P = 24 W", (0, 100))

# === Scenario 2: P = 70 W ===
data2a  = np.loadtxt("lin_Scenario2_P70_v0.5.txt")
data2a2 = np.loadtxt("lin_Scenario2_P70_v0.1.txt")
time2a,  temp2a  = 1000 * data2a[:, 0],  data2a[:, 1]
time2a2, temp2a2 = 1000 * data2a2[:, 0], data2a2[:, 1]

ax = axes[1]
ax.plot(time2a2, temp2a2, lw=2, color='red',  label="v = 0.10 m/s")
ax.plot(time2a,  temp2a,  lw=2, color='black', label="v = 0.50 m/s")
ax.axhline(1700, color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold, color='black', linestyle=':', label='Ambient Temperature')
style_axes(ax, "P = 70 W", (0, 100))

# === Scenario 3: P = 45 W ===
data3a = np.loadtxt("lin_Scenario3_P45_v0.15.txt")
data3b = np.loadtxt("lin_Scenario3_P45_v0.25.txt")
time3a, temp3a = 1000 * data3a[:, 0], data3a[:, 1]
time3b, temp3b = 1000 * data3b[:, 0], data3b[:, 1]

ax = axes[2]
ax.plot(time3a, temp3a, lw=2, color='red',  label="v = 0.15 m/s")
ax.plot(time3b, temp3b, lw=2, color='black',  label="v = 0.25 m/s")
ax.axhline(1700, color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold, color='black', linestyle=':', label='Ambient Temperature')
style_axes(ax, "P = 45 W", (0, 100))

# === Scenario 4: P = 120 W ===
data4a  = np.loadtxt("lin_Scenario4_P120_v0.5.txt")
data4a2 = np.loadtxt("lin_Scenario4_P120_v0.8.txt")
time4a,  temp4a  = 1000 * data4a[:, 0],  data4a[:, 1]
time4a2, temp4a2 = 1000 * data4a2[:, 0], data4a2[:, 1]

ax = axes[3]
ax.plot(time4a,  temp4a,  lw=2, color='red',   label="v = 0.50 m/s")
ax.plot(time4a2, temp4a2, lw=2, color='black', label="v = 0.80 m/s (with heat losses)")
ax.axhline(1700, color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold, color='black', linestyle=':', label='Ambient Temperature')
style_axes(ax, "P = 120 W", (0, 40))

# === Scenario 5: P = 200 W ===
data5a = np.loadtxt("lin_Scenario5_P200_v0.8.txt")
data5b = np.loadtxt("lin_Scenario5_P200_v1.0.txt")
time5a, temp5a = 1000 * data5a[:, 0], data5a[:, 1]
time5b, temp5b = 1000 * data5b[:, 0], data5b[:, 1]

ax = axes[4]
ax.plot(time5a, temp5a, lw=2, color='red', label="v = 0.80 m/s")
ax.plot(time5b, temp5b, lw=2, color='black', label="v = 1.00 m/s")
ax.axhline(1700, color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold, color='black', linestyle=':', label='Ambient Temperature')
style_axes(ax, "P = 200 W", (0, 40))

# Remove unused panel
fig.delaxes(axes[5])

plt.show()









#=============================
#    NON-LINEAR PROBLEM      =
#=============================
# === Setup ===
data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/results_nonlinear_evaporation"
os.chdir(data_folder)

import numpy as np
import matplotlib.pyplot as plt

# Font/spacing settings (same as linear)
plt.rcParams.update({
    'axes.titlesize': 12,     # subplot titles
    'axes.titlepad': 12,      # space between title and plot
    'axes.labelsize': 14,     # BIGGER axis labels (x and y titles)
    'axes.labelpad': 10,      # padding between labels and ticks
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

T_threshold = 296  # Ambient temperature [K]

# ~10x12 in figure ≈ six panels that read like 5x4 in each
fig, axes = plt.subplots(3, 2, figsize=(10, 12))
fig.subplots_adjust(hspace=0.5, wspace=0.35)
axes = axes.flatten()

# Helper to style all subplots the same
def style_axes(ax, title, xlim):
    ax.set_title(title)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Temperature [K]")
    ax.grid(True)
    ax.legend()
    ax.set_xlim(*xlim)

# === Scenario 1: P = 24 W ===
data1a = np.loadtxt("nonlin_Scenario1_P24_v0.15.txt")
data1b = np.loadtxt("nonlin_Scenario1_P24_v0.25.txt")
time1a, temp1a = 1000 * data1a[:, 0], data1a[:, 1]
time1b, temp1b = 1000 * data1b[:, 0], data1b[:, 1]

ax = axes[0]
ax.plot(time1a, temp1a, lw=2, color='red',   label="v = 0.15 m/s")
ax.plot(time1b, temp1b, lw=2, color='black', label="v = 0.25 m/s")
ax.axhline(1700,       color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold,color='black', linestyle=':',  label='Ambient Temperature')
style_axes(ax, "P = 24 W", (0, 100))

# === Scenario 2: P = 70 W ===
data2a  = np.loadtxt("nonlin_Scenario2_P70_v0.5.txt")
data2a2 = np.loadtxt("nonlin_Scenario2_P70_v0.1.txt")
time2a,  temp2a  = 1000 * data2a[:, 0],  data2a[:, 1]   # 0.50 m/s
time2a2, temp2a2 = 1000 * data2a2[:, 0], data2a2[:, 1]  # 0.10 m/s

ax = axes[1]
ax.plot(time2a2, temp2a2, lw=2, color='red',   label="v = 0.10 m/s")
ax.plot(time2a,  temp2a,  lw=2, color='black', label="v = 0.50 m/s")
ax.axhline(1700,       color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold,color='black', linestyle=':',  label='Ambient Temperature')
style_axes(ax, "P = 70 W", (0, 100))

# === Scenario 3: P = 45 W ===
data3a = np.loadtxt("nonlin_Scenario3_P45_v0.15.txt")
data3b = np.loadtxt("nonlin_Scenario3_P45_v0.25.txt")
time3a, temp3a = 1000 * data3a[:, 0], data3a[:, 1]
time3b, temp3b = 1000 * data3b[:, 0], data3b[:, 1]

ax = axes[2]
ax.plot(time3a, temp3a, lw=2, color='red',   label="v = 0.15 m/s")
ax.plot(time3b, temp3b, lw=2, color='black', label="v = 0.25 m/s")
ax.axhline(1700,       color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold,color='black', linestyle=':',  label='Ambient Temperature')
style_axes(ax, "P = 45 W", (0, 100))

# === Scenario 4: P = 120 W ===
data4a  = np.loadtxt("nonlin_Scenario4_P120_v0.5.txt")
data4a2 = np.loadtxt("nonlin_Scenario4_P120_v0.8.txt")
time4a,  temp4a  = 1000 * data4a[:, 0],  data4a[:, 1]   # 0.50 m/s
time4a2, temp4a2 = 1000 * data4a2[:, 0], data4a2[:, 1]  # 0.80 m/s

ax = axes[3]
ax.plot(time4a,  temp4a,  lw=2, color='red',   label="v = 0.50 m/s")
ax.plot(time4a2, temp4a2, lw=2, color='black', label="v = 0.80 m/s (with heat losses)")
ax.axhline(1700,       color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold,color='black', linestyle=':',  label='Ambient Temperature')
style_axes(ax, "P = 120 W", (0, 40))

# === Scenario 5: P = 200 W ===
data5a = np.loadtxt("nonlin_Scenario5_P200_v0.8.txt")
data5b = np.loadtxt("nonlin_Scenario5_P200_v1.0.txt")
time5a, temp5a = 1000 * data5a[:, 0], data5a[:, 1]
time5b, temp5b = 1000 * data5b[:, 0], data5b[:, 1]

ax = axes[4]
ax.plot(time5a, temp5a, lw=2, color='red',   label="v = 0.80 m/s")
ax.plot(time5b, temp5b, lw=2, color='black', label="v = 1.00 m/s")
ax.axhline(1700,       color='blue', linestyle='--', label='Melting Point')
ax.axhline(T_threshold,color='black', linestyle=':',  label='Ambient Temperature')
style_axes(ax, "P = 200 W", (0, 40))

# Remove unused panel
fig.delaxes(axes[5])




import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === Cooling Rate and Interpass Time Extraction ===
T_threshold = 300  # Ambient temperature threshold [K]
Power_pts = []
CoolingRate_pts = []
Speed_pts = []
InterpassTime_pts = []

print("\n=== Cooling Rate & Interpass Time Analysis ===")

# List of datasets: (data_array, laser_power[W], scan_speed[m/s])
datasets = [
    (data1a,  24, 0.15),
    (data1b,  24, 0.25),
    (data2a2, 70, 0.10),
    (data2a,  70, 0.50),
    (data3a,  45, 0.50),
    (data3b,  45, 0.80),
    (data4a, 120, 0.50),
    (data4a2,120, 0.80),
    (data5a, 200, 0.80),  
    (data5b, 200, 1.00)   
]

# --- Calculate cooling rate (CR) and interpass time (Δt) ---
for data, P, v in datasets:
    time = data[:, 0]  # [s]
    temp = data[:, 1]  # [K]

    idx_peak = np.argmax(temp)       # index of peak temperature
    T_peak = temp[idx_peak]          # peak temperature [K]
    t_peak = time[idx_peak]          # peak time [s]

    try:
        # Find first time after peak when temperature drops below threshold
        idx_thresh = np.where(temp[idx_peak:] < T_threshold)[0][0] + idx_peak
        T_thresh = temp[idx_thresh]
        t_thresh = time[idx_thresh]

        CR = (T_peak - T_thresh) / (t_thresh - t_peak)   # Cooling rate [K/s]
        delta_t = t_thresh - t_peak                      # Cooling time [s]

        CoolingRate_pts.append(CR)
        InterpassTime_pts.append(delta_t)
        Power_pts.append(P)
        Speed_pts.append(v)

        print(f"P = {P} W, v = {v:.2f} m/s → CR = {CR:.1f} K/s, Δt = {delta_t*1000:.1f} ms")

    except IndexError:
        print(f"P = {P} W, v = {v:.2f} m/s → did not cool below {T_threshold} K")

# =========================================
#   Plot: Cooling Rate vs Laser Power
# =========================================
unique_speeds = sorted(set(Speed_pts))
colors = cm.viridis(np.linspace(0, 1, len(unique_speeds)))
markers = ['o', 's', '^', 'D']

plt.figure(figsize=(6, 4))
for i, v in enumerate(unique_speeds):
    Ps  = [p  for p,  sv in zip(Power_pts, Speed_pts) if sv == v]
    CRs = [cr for cr, sv in zip(CoolingRate_pts, Speed_pts) if sv == v]
    plt.scatter(Ps, CRs, color=colors[i], marker=markers[i % len(markers)],
                s=70, label=f"v = {v} m/s")

plt.xlabel("Laser Power [W]", fontsize=14)
plt.ylabel("Cooling Rate [K/s]", fontsize=14)
plt.title("Cooling Rate vs Laser Power", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# =========================================
#   Plot: Cooling Time vs Laser Power
# =========================================
plt.figure(figsize=(6, 4))
for i, v in enumerate(unique_speeds):
    Ps  = [p   for p,  sv in zip(Power_pts, Speed_pts) if sv == v]
    IPs = [ip*1000 for ip, sv in zip(InterpassTime_pts, Speed_pts) if sv == v]  # [ms]
    plt.scatter(Ps, IPs, color=colors[i], marker=markers[i % len(markers)],
                s=70, label=f"v = {v} m/s")

plt.xlabel("Laser Power [W]", fontsize=14)
plt.ylabel("Cooling Time [ms]", fontsize=14)
plt.title("Cooling Time vs Laser Power", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
