import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--equilibrium', type=str, required=True, help='name of .bc or .nc file specifiying the equilibrium, should be in profiles folder')
parser.add_argument('--profile', type=str, required=True, help='name of .opt file specifying the profiles in the profiles folder')
parser.add_argument('--numRad', type=str, required=True, help='number of radii at which to perform sfincs calculations')
parser.add_argument('--rN_min', type=str, required=True, help='innermost rN to perform calculation')
parser.add_argument('--rN_max', type=str, required=True, help='outermost rN to perform calculation')
parser.add_argument('--NErs', type=str, required=True, help='number of samples to scan between -ErRange and ErRange')
parser.add_argument('--ErRange', type=str, required=True, help='range of Er to scan-- ErMin = -ErRange, ErMax = ErRange (in V)')
parser.add_argument('--NSB', type=str, required=True, help=r'number of values -1 for \epsilon_{sb}. Ex. "4" will result in values 0.0, 0.25, 0.5, 0.75, 1.0')

eq = parser.parse_args().equilibrium
profile = parser.parse_args().profile
numRad = parser.parse_args().numRad
rN_min = parser.parse_args().rN_min
rN_max = parser.parse_args().rN_max
NErs = parser.parse_args().NErs
ErRange = float(parser.parse_args().ErRange)
NSB = int(parser.parse_args().NSB)

main_dir = os.getcwd()

esbs = np.linspace(0.0, 1.0, NSB+1)

for esb in esbs:
    print(f"### Setting up scan over radius and Er for esb = {esb}... ###")
    os.mkdir(f"esb_{esb}")
    os.chdir(f"./esb_{esb}")
    os.system(f"python $UTILS_DIR/runRadErScan.py --equilibrium {eq} --profile {profile} --numRad {numRad} --rN_min {rN_min} --rN_max {rN_max} --NErs {NErs} --ErRange {ErRange} --SB")
    os.chdir(f"./raderscan")
    with open("./input.namelist", "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "min_Bmn_to_load" in line:
            new_lines.append("\tsymm_breaking = 2 ! Fourier modes of 1/B^2 used for symmetry breaking\n")
            new_lines.append("\tsymm_type = 0 ! Applies scaling to non-QA modes\n")
            new_lines.append(f"\tepsilon_symmbreak = {esb} ! Factor by which to scale non-QS modes\n")
    with open("./input.namelist", "w") as f:
        f.writelines(new_lines)
    print(f"### Adjusted namelist to scale non-QA modes of 1/B^2 by {esb} ###")
    print(f"### Launching scan over radius and Er... ###")
    os.system("echo y | python $SCAN_DIR/sfincsScan")
    os.chdir(main_dir)

os.chdir(os.environ['SCAN_DIR'])
with open(f"./sfincsScan_5", "r") as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    modified_line = line.replace(f"!ss NErs = {NErs}", f"!ss NErs = 101")
    modified_line = modified_line.replace(f"!ss ErMax = {ErRange/1000}", f"!ss ErMax = 0.020")
    modified_line = modified_line.replace(f"!ss ErMin = {-ErRange/1000}", f"!ss ErMin = -0.020")
    new_lines.append(modified_line)
with open(f"./sfincsScan_5", "w") as f:
    f.writelines(new_lines)
