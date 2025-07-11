## 09/07/2025 ##
# this code is the post processing of the different laser scenaria, and the mean value of the cooling rates.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

data_folder = "/home/ntinos/Documents/FEnics/heat equation/demos_heat equation/txt"
# Change working directory
os.chdir(data_folder)

# === INPUT ===
scenario_files = [
    ["Scenario1_P24_v0.1.txt0", "Scenario1_P24_v0.15.txt0"],
    # Add your P50 files here
    ["Scenario2_P66_v0.10.txt0", "Scenario2_P66_v0.15.txt0"],
    ["Scenario3_P70_v0.1.txt0", "Scenario3_P70_v0.5.txt0"],
    ["Scenario4_P120_v0.5.txt0", "Scenario4_P120_v0.8.txt0"],
    ["Scenario5_P200_v0.8.txt0", "Scenario5_P200_v1.0.txt0"]
]

powers = [24, 66, 70, 120, 200]  # [W] 

speeds = [                       # m/s
    [0.1, 0.15],
    [0.10, 0.15],
    [0.1, 0.5],
    [0.5, 0.8],
    [0.8, 1.0]
]

T_threshold = 478  # [K], this is the interpass temperature below which the sample can be rescanned

markers = ['o', 's']

# === Make multi-panel subplots ===
fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows x 3 cols
axes = axes.flatten()  # Easy indexing

for i, (files, P, vs) in enumerate(zip(scenario_files, powers, speeds)):
    ax = axes[i]

    for file, v in zip(files, vs):
        data = np.loadtxt(file)
        time_ms = 1000 * data[:, 0]
        temp = data[:, 1]

        ax.plot(
            time_ms, temp,
            lw=2,
            label=f"v = {v} m/s"
        )

    ax.axhline(y=1700, color='k', linestyle='--', label='Melting Point')
    ax.axhline(y=T_threshold, color='blue', linestyle=':', label='Interpass Threshold')

    ax.set_title(f"P = {P} W")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Temperature [K]")
    ax.grid(True)
    ax.legend(fontsize=8)

# Remove empty last subplot if any
if len(scenario_files) < len(axes):
    fig.delaxes(axes[-1])

fig.suptitle("Centerline Temperature Evolution ", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle


# === 2. Calculate cooling rates AND interpass times ===
Power_pts = []
CoolingRate_pts = []
Speed_pts = []
InterpassTime_pts = []

for files, P, vs in zip(scenario_files, powers, speeds):
    for file, v in zip(files, vs):
        data = np.loadtxt(file)
        time = data[:, 0]  # [s]
        temp = data[:, 1]  # [K]

        idx_peak = np.argmax(temp)
        T_peak = temp[idx_peak]
        t_peak = time[idx_peak]

        try:
            idx_thresh = np.where(temp[idx_peak:] < T_threshold)[0][0] + idx_peak
            T_thresh = temp[idx_thresh]
            t_thresh = time[idx_thresh]

            CR = (T_peak - T_thresh) / (t_thresh - t_peak)  # [K/s]
            interpass_time = t_thresh - t_peak  # [s]

            Power_pts.append(P)
            CoolingRate_pts.append(CR)
            Speed_pts.append(v)
            InterpassTime_pts.append(interpass_time)

            print(f"P = {P} W, v = {v} m/s → Cooling Rate = {CR:.1f} K/s, Interpass Time = {interpass_time*1000:.2f} ms")

        except IndexError:
            print(f" P = {P} W, v = {v} m/s → Did not cool below threshold!")

# === 3. Make Cooling Rate vs Power plot === 
unique_speeds = sorted(set([v for sublist in speeds for v in sublist]))
colors = cm.viridis(np.linspace(0, 1, len(unique_speeds)))
markers = ['o', 's']

plt.figure(figsize=(6, 4))
for i, uspeed in enumerate(unique_speeds):
    P_group = [p for p, v in zip(Power_pts, Speed_pts) if v == uspeed]
    CR_group = [c for c, v in zip(CoolingRate_pts, Speed_pts) if v == uspeed]
    plt.scatter(
        P_group, CR_group,
        color=colors[i],
        marker=markers[i % len(markers)],
        s=70,
        label=f"v = {uspeed} m/s"
    )

plt.xlabel("Laser Power [W]", fontsize=12)
plt.ylabel("Cooling Rate [K/s]", fontsize=12)
plt.title("Average Cooling Rate vs Laser Power", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Scan Speed")
plt.tight_layout()


# === 4. Make Interpass Time vs Power plot ===
plt.figure(figsize=(6, 4))
for i, uspeed in enumerate(unique_speeds):
    P_group = [p for p, v in zip(Power_pts, Speed_pts) if v == uspeed]
    IP_group = [ip for ip, v in zip(InterpassTime_pts, Speed_pts) if v == uspeed]
    IP_group_ms = [ip * 1000 for ip in IP_group]  # Convert to ms

    plt.scatter(
        P_group, IP_group_ms,
        color=colors[i],
        marker=markers[i % len(markers)],
        s=70,
        label=f"v = {uspeed} m/s"
    )

plt.xlabel("Laser Power [W]", fontsize=12)
plt.ylabel("Interpass Cooling Time [ms]", fontsize=12)
plt.title("Interpass Cooling Time vs Laser Power", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title="Scan Speed")
plt.tight_layout()
plt.show()
plt.close()