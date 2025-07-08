import numpy as np
import os
from processOutputs import make_qlcfs_file
import matplotlib.pyplot as plt

"""
call from the temptrials directory. parses the qlcfs files in the
directories hardcoded into this file, and produces a plot of the neoclassical
and turbulent heat fluxes for each of the configurations at the temperatures
specified
"""
if not os.path.exists("./plots/"):
    os.mkdir("plots")

temperatures = np.array([7.5, 11.25, 15]) #eV
cnt0_5folders = ["cnt_0.5_7", "cnt_0.5_11", "cnt_0.5_15"]
csx0_5folders = ["csx7", "csx11", "csx15"]
csx0_1folders = ["csx_0.1_7", "csx_0.1_11", "csx_0.1_15"]
cnt0_1folders = [name.replace("csx", "cnt") for name in csx0_1folders]
cnt0_2folders = [name.replace("0.1", "0.2341") for name in cnt0_1folders]

configsnames = ["CSX (0.5 T)", "CSX (0.1 T)", "CNT (0.5 T)", "CNT (0.2341 T)", "CNT (0.1 T)"]
configs = [csx0_5folders, csx0_1folders, cnt0_5folders, cnt0_2folders, cnt0_1folders]
colors = ["mediumblue", "red", "green", "gold", "darkorchid"]

def parseqlcfs(folder):
    os.chdir(f"./{folder}/ambipolar/")
    make_qlcfs_file()
    qlcfs = np.load("qlcfs.npy")
    os.chdir("../../")
    return qlcfs

fluxes = []
for i, config in enumerate(configs):
    Qn = []
    Qt = []
    for temp in config:
        print(f"File: {temp}")
        qn, qt = parseqlcfs(temp)
        Qn.append(qn)
        Qt.append(qt)
    fluxes.append([Qn, Qt])

# Making neoclassical figure
fig = plt.figure()
ax = fig.add_subplot()
for i, configname in enumerate(configsnames):
    ax.plot(temperatures, fluxes[i][0], marker='o', label=configname, color=colors[i])
    ax.set_xlabel("on-axis electron temperature $T_e$ [eV]")
    ax.set_ylabel("Power from neoclassical heat flux [W]")
    ax.set_yscale('log')
    ax.legend()
fig.tight_layout()
fig.savefig("./plots/neoclassical.jpeg")

fig = plt.figure()
ax = fig.add_subplot()
for i, configname in enumerate(configsnames):
    ax.plot(temperatures, fluxes[i][1], marker='o', label=configname, color=colors[i])
    ax.set_xlabel("on-axis electron temperature $T_e$ [eV]")
    ax.set_ylabel("Power from turbulent heat flux [W]")
    ax.set_yscale('log')
    ax.legend()
fig.tight_layout()
fig.savefig("./plots/turbulent.jpeg")

fig = plt.figure()
ax = fig.add_subplot()
for i, configname in enumerate(configsnames):
    ax.plot(temperatures, fluxes[i][0], marker='o', label=configname+" NC", color=colors[i])
    ax.plot(temperatures, fluxes[i][1], marker='x', label=configname+" GB", color=colors[i])
    ax.set_xlabel("on-axis electron temperature $T_e$ [eV]")
    ax.set_ylabel("Power from heat flux [W]")
    ax.set_yscale('log')
    ax.legend()
fig.tight_layout()
fig.savefig("./plots/both.jpeg")

