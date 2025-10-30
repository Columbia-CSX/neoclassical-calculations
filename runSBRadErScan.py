"""See runRadErScan.py for the full documentation. This script serves as a wrapper for runRadErScan for the case where
one would like to investigate what happens when the symmetry breaking factor on the asymmetric modes of 1/B^2 are 
introduced. The only additional argument this script accepts is --NSB, the number of additional symmetry scenarios to 
include. Ex. -NSB 5 would include the cases where the symmetry breaking factor is 0, 1/4, 1/2, 3/4, and 1.

What is meant by symmetry breaking factor e_sb: As an example for QA, 1/B^2 can be expressed F(theta) + G(theta, zeta)
for some F and G where G depends nontrivially on zeta. To obtain the configuration with adjusted symmetry, we'll redefine
B to be such that 1/B^2 = F(theta) + e_sb * G(theta, zeta). In this way, e_sb corresponds to the original configuration, 
and e_sb = 0 corresponds to an adjusted configuration with perfect quasiaxisymmetry. This code currently works only for
QA, not QP or QH."""

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
parser.add_argument('--ErRange', type=str, required=True, nargs="+", help='range of Er to scan-- ErMin = -ErRange, ErMax = ErRange (in V)\nif instead of a single number, two are provided then they will be interpreted as "ErMin ErMax"')
parser.add_argument('--NSB', type=str, required=True, help=r'number of values -1 for \epsilon_{sb}. Ex. "4" will result in values 0.0, 0.25, 0.5, 0.75, 1.0')
parser.add_argument('--Ntheta', type=str, required=True, help='resolution for SFINCS')
parser.add_argument('--Nzeta', type=str, required=True, help='resolution for SFINCS for 1 field period')
parser.add_argument('--Nxi', type=str, required=True, help='resolution for SFINCS')

eq = parser.parse_args().equilibrium
profile = parser.parse_args().profile
numRad = parser.parse_args().numRad
rN_min = parser.parse_args().rN_min
rN_max = parser.parse_args().rN_max
NErs = parser.parse_args().NErs
ErRange = parser.parse_args().ErRange
NSB = int(parser.parse_args().NSB)
Ntheta = parser.parse_args().Ntheta
Nzeta = parser.parse_args().Nzeta
Nxi = parser.parse_args().Nxi

try:
    ErRange = float(parser.parse_args().ErRange)
    ErMax = ErRange
    ErMin = -ErRange
except:
    try:
        ErMin, ErMax = parser.parse_args().ErRange
        ErMin = float(ErMin)
        ErMax = float(ErMax)
    except:
        raise ValueError("Invalid input provided for ErRange")

if isinstance(ErRange, list):
    ErRange = ErRange[0] + " " + ErRange[1]

main_dir = os.getcwd()

esbs = np.linspace(0.0, 1.0, NSB+1)

for esb in esbs:
    print(f"### Setting up scan over radius and Er for esb = {esb}... ###")
    if not os.path.exists(f"./esb_{esb}"):
        os.mkdir(f"esb_{esb}")
    os.chdir(f"./esb_{esb}")
    os.system(f"python $UTILS_DIR/runRadErScan.py --equilibrium {eq} --profile {profile} --numRad {numRad} --rN_min {rN_min} --rN_max {rN_max} --NErs {NErs} --ErRange {ErRange} --SB --Ntheta {Ntheta} --Nzeta {Nzeta} --Nxi {Nxi}")
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
    modified_line = modified_line.replace(f"!ss ErMax = {ErMax/1000}", f"!ss ErMax = 0.004")
    modified_line = modified_line.replace(f"!ss ErMin = {-ErMax/1000}", f"!ss ErMin = -0.004")
    new_lines.append(modified_line)
with open(f"./sfincsScan_5", "w") as f:
    f.writelines(new_lines)
