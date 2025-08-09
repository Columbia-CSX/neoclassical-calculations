"""
Scans through the raderscan folder made by runRadErScan.py
and organizes the radial current vs Er data on each flux surface into a file
to be read by interpret.py.
"""
import os
from processOutputs import *
import numpy as np
from tqdm import tqdm
import argparse

# --- Enables and disables progress bar --- #
disabletqdm = None

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--disabletqdm', type=str, help='put any string to disable progres sbar')

disabletqdm = parser.parse_args().disabletqdm
if disabletqdm is None:
    disabletqdm=False
else:
    disabletqdm=True
# ----------------------------------------- #

main_dir = os.getcwd()
assert "raderscan" in os.listdir(), "parseRadErScan.py should be called in the directory containing 'raderscan' folder"

os.chdir("raderscan")
os.system("mkdir determineEr")

# ---------- Main part ------------------------------------------------------------#
# for each rN_0.xx directory in raderscan, sorts through the subdirectories Er___
# stores a .dat file of Ir vs Er data in determineEr for each radial position. Also
# saves a figure showing Ir vs Er for each flux surface. Spits out details about
# and skips over failed sfincs runs.

rN_folders = [folder for folder in os.listdir() if folder.startswith("rN_")]

for folder in tqdm(rN_folders, disable=disabletqdm, desc="Parsing data from radius and Er scan..."):
    os.chdir(folder)
    Er_folders = [file for file in os.listdir() if file.startswith("Er")]
    Ers = []
    Irs = []
    failed = []
    zeroes = []
    for file in Er_folders:
        if not os.path.exists(f"{file}/sfincsOutput.h5"):
            print(f"sfincs run {folder} {file} seems to have failed.")
            failed.append(file)
            continue
        try:
            er = valsafe(parseHDF5(file, Er).data)
            Ir = valsafe(getRadialCurrent(file))
        except:
            failed.append(file)
            continue
        if Ir != 0.0:
            Ers.append(er)
            Irs.append(Ir)
        else:
            zeroes.append(file)
    if len(failed)>0:
        print(f"Accessing sfincs parameters failed for the following files in {folder}.")
        print("This likely indicates a failed or incomplete sfincs run:")
        print(failed)
    if len(zeroes)>0:
        print(f"sfincs calculated a 0 radial current for the following files in {folder}.")
        print("This indicates a completed but unphysical sfincs result:")
        print(zeroes)
    if len(Er_folders) == len(failed) + len(zeroes):
        print(f"All calculations on {folder} failed.")
        # raise ValueError("All calculations on a flux surface failed")
        os.chdir("..")
        continue
    os.chdir("..")
    Ers, Irs = zip(*sorted(zip(Ers, Irs), key=lambda pair: pair[0]))
    Ers = list(Ers)
    Irs = list(Irs)
    np.savetxt(f"./determineEr/raderscan-{folder}-Ir-vs-Er.dat", np.array([Ers, Irs]).T)
    
    #plotting the output
    fig = plt.figure()
    ax = fig.add_subplot()
    Irs_plus = np.ma.masked_less_equal(Irs, 0)
    Irs_minus = np.ma.masked_greater(Irs, 0)
    ax.plot(Ers, Irs_plus, marker='o', color='red', label='+')
    ax.plot(Ers, Irs_minus, marker='o', color='blue', label='-')
    ax.set_xlabel("$E_r$ [V/m]")
    ax.set_ylabel("$I_r$ [A]")
    ax.set_title(f"{folder}")
    fig.legend()
    fig.show()
    fig.savefig(f"./determineEr/raderscan-{folder}-Ir-vs-Er.jpeg")

print(f"Data from radius and Er scan stored in:\n\t{os.getcwd()}/determineEr.")

