import os
import numpy as np
from processOutputs import *

"""
helper script for plotting results of tempscan

run in a directory containing folders named like
TXXXeV
where XXX is the temperature T_e of the plasma.
Each such folder should contain a raderscan and 
ambipolar directory
"""

def getTemperatureFromFileName(filename):
    return float(filename.replace("T", "").replace("eV", ""))

def makeQ_NvsTemperaturePlot():
    main_dir = os.getcwd()
    
    os.system("mkdir plots") if not os.path.exists("plots")

    tempfolders = [folder for folder in os.listdir() if folder.endswith("eV")]
    Ts, tempfolders = zip(*sorted(zip([getTemperatureFromFileName(file) for file in tempfolders], tempfolders), key=lambda x: x[0]))

    Q_Ns = []
    for tempfolder in tempfolder:
        os.chdir(f"{tempfolder}/ambipolar/")
        Q_N, _ = valsafe(make_qlcfs_file())
        Q_Ns.append(Q_N)
        os.chdir(main_dir)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    ax.plot(Ts, Q_Ns, marker='o', color=columbia)
    ax.set_xlabel("Core electron temperature $T_e$ [eV]", fontsize=22)
    ax.set_ylabel("$Q_N$ [W]", fontsize=22)
    fig.savefig(f"./plots/Q_NvsTemp.jpeg", dpi=360)

if __name__ == "__main__":
    makeQ_NvsTemperaturePlot()
