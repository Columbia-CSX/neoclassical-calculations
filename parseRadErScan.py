import os
from processOutputs import *
import numpy as np
from tqdm import tqdm
import argparse

disabletqdm = None

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--disabletqdm', type=str, help='put any string to disable progres sbar')

disabletqdm = parser.parse_args().disabletqdm
if disabletqdm is None:
    disabletqdm=False
else:
    disabletqdm=True

main_dir = os.getcwd()
assert "raderscan" in os.listdir(), "parseRadErScan.py should be called in the directory containing 'raderscan' folder"

os.chdir("raderscan")
os.system("mkdir determineEr")

# for each rN_0.xx directory in raderscan, sorts through the subdirectories Er___
# stores a .dat file of Jr vs Er data in determineEr for each radial position

rN_folders = [folder for folder in os.listdir() if folder.startswith("rN_")]

for folder in tqdm(rN_folders, disable=disabletqdm, desc="Parsing data from radius and Er scan..."):
    os.chdir(folder)
    Er_folders = [file for file in os.listdir() if file.startswith("Er")]
    Ers = []
    Irs = []
    for file in Er_folders:
        Ers.append(valsafe(parseHDF5(file, Er).data))
        Irs.append(valsafe(getRadialCurrent(file)))
    os.chdir("..")
    Ers, Irs = zip(*sorted(zip(Ers, Irs), key=lambda pair: pair[0]))
    Ers = list(Ers)
    Irs = list(Irs)
    np.savetxt(f"./determineEr/raderscan-{folder}-Ir-vs-Er.dat", np.array([Ers, Irs]).T)

print(f"Data from radius and Er scan stored in:\n\t{os.getcwd()}/determineEr.")
