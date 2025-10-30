"""
Runs a scan over radial variable and radial electric field for a given configuration (equilibrium) and set of profiles.
In the working directory, a folder will appear called "raderscan". This will contain folders that looks like "rN_0.xx", 
one for each flux surface. Inside of these will be folders that look like "ErX" corresponding to each Er (X is in kV).

Example use: python runRadErScan.py --numRad 2 --rN_min 0.95 --rN_max 0.99 --NErs 121 --ErRange 60 --Nzeta 31 --Ntheta 19 --Nxi 71 --equilibrium wout_csx_ls_4.5_0.5T.bc --profile kh72_11.2_eV.opt

For more information: python runRadErScan.py --help
"""

import os
import argparse

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--equilibrium', type=str, required=True, help='name of .bc or .nc file specifiying the equilibrium, should be in profiles folder')
parser.add_argument('--profile', type=str, required=True, help='name of .opt file specifying the profiles in the profiles folder')
parser.add_argument('--numRad', type=str, required=True, help='number of radii at which to perform sfincs calculations')
parser.add_argument('--rN_min', type=str, required=True, help='innermost rN to perform calculation')
parser.add_argument('--rN_max', type=str, required=True, help='outermost rN to perform calculation')
parser.add_argument('--NErs', type=str, required=True, help='number of samples to scan between -ErRange and ErRange')
parser.add_argument('--ErRange', type=str, required=True, nargs="+", help='range of Er to scan-- ErMin = -ErRange, ErMax = ErRange (in V/m) if ErRange is a number\nif instead a tuple is provided, ex. "ErMin ErMax" then the scan will happen between these values')
parser.add_argument('--SB', action=argparse.BooleanOptionalAction)
parser.add_argument('--Nzeta', type=str, default="51", help="Number of points along zeta to use for the 1 field period mesh")
parser.add_argument('--Ntheta', type=str, default="15", help="Number of points along theta to use")
parser.add_argument('--Nxi', type=str, default="149", help="Number of points in xi in phase space")
parser.set_defaults(SB=False)

eq = parser.parse_args().equilibrium
profile = parser.parse_args().profile
numRad = parser.parse_args().numRad
rN_min = parser.parse_args().rN_min
rN_max = parser.parse_args().rN_max
NErs = parser.parse_args().NErs
SB = bool(parser.parse_args().SB)
Nzeta = parser.parse_args().Nzeta
Ntheta = parser.parse_args().Ntheta
Nxi = parser.parse_args().Nxi

if int(NErs) % 2 == 0:
    NErs = str(int(NErs)+1)

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

eqFile = "/global/homes/m/michaelc/stelloptPlusSfincs/equilibria/"+eq
proFile = "/global/homes/m/michaelc/stelloptPlusSfincs/profiles/"+profile

# modifies sfincsScan_5 to agree with the desired Er Range
main_dir = os.getcwd()
os.chdir(os.environ['SCAN_DIR'])
with open(f"./sfincsScan_5", "r") as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    modified_line = line.replace(f"!ss NErs = 81", f"!ss NErs = {NErs}")
    modified_line = modified_line.replace(f"!ss ErMax = 0.004", f"!ss ErMax = {ErMax/1000}")
    modified_line = modified_line.replace(f"!ss ErMin = -0.004", f"!ss ErMin = {ErMin/1000}")
    new_lines.append(modified_line)
with open(f"./sfincsScan_5", "w") as f:
    f.writelines(new_lines)
os.chdir(main_dir)

# makes a new directory and prepares the scan
os.system("mkdir raderscan")
os.chdir("./raderscan")
os.system("python /global/homes/m/michaelc/stelloptPlusSfincs/stelloptPlusSfincs/run.py --profilesIn "+proFile+" --eqIn "+eqFile+" --radialVar 3 --Nzeta "+Nzeta+" --Ntheta "+Ntheta+" --Nxi "+Nxi+" --driftScheme 2 --saveLoc "+os.getcwd()+" --time 00-0:20:00 --nNodes 1 --notifs all --noRun")

# adjust all of the prepared files to agree with the range of radii
with open(f"./input.namelist", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    modified_line = line.replace("!ss scanType = 4", f"!ss scanType = 5")
    modified_line = modified_line.replace("!ss Nradius = 16", f"!ss Nradius = {numRad}")
    modified_line = modified_line.replace("!ss rN_min = 0.15", f"!ss rN_min = {rN_min}")
    modified_line = modified_line.replace("!ss rN_max = 0.95", f"!ss rN_max = {rN_max}")
    new_lines.append(modified_line)

with open(f"./input.namelist", "w") as f:
    f.writelines(new_lines)

if not SB:
    # launches jobs
    os.system("python $SCAN_DIR/sfincsScan")

    # rewrites sfincsScan_5 so that it is net unchanged
    os.chdir(os.environ['SCAN_DIR'])
    with open(f"./sfincsScan_5", "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        modified_line = line.replace(f"!ss NErs = {NErs}", f"!ss NErs = 81")
        modified_line = modified_line.replace(f"!ss ErMax = {ErMax/1000}", f"!ss ErMax = 0.004")
        modified_line = modified_line.replace(f"!ss ErMin = {ErMin/1000}", f"!ss ErMin = -0.004")
        new_lines.append(modified_line)
    with open(f"./sfincsScan_5", "w") as f:
        f.writelines(new_lines)

