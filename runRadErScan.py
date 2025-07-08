import os
import argparse

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--equilibrium', type=str, required=True, help='name of .bc or .nc file specifiying the equilibrium, should be in profiles folder')
parser.add_argument('--profile', type=str, required=True, help='name of .opt file specifying the profiles in the profiles folder')
parser.add_argument('--numRad', type=str, required=True, help='number of radii at which to perform sfincs calculations')
parser.add_argument('--rN_min', type=str, required=True, help='innermost rN to perform calculation')
parser.add_argument('--rN_max', type=str, required=True, help='outermost rN to perform calculation')
parser.add_argument('--NErs', type=str, required=True, help='number of samples to scan between -ErRange and ErRange')
parser.add_argument('--ErRange', type=str, required=True, help='range of Er to scan-- ErMin = -ErRange, ErMax = ErRange (in V)')
parser.add_argument('--SB', action=argparse.BooleanOptionalAction)
parser.set_defaults(SB=False)

eq = parser.parse_args().equilibrium
profile = parser.parse_args().profile
numRad = parser.parse_args().numRad
rN_min = parser.parse_args().rN_min
rN_max = parser.parse_args().rN_max
NErs = parser.parse_args().NErs
ErRange = float(parser.parse_args().ErRange)
SB = bool(parser.parse_args().SB)

if int(NErs) % 2 == 0:
    NErs = str(int(NErs)+1)

eqFile = "/global/homes/m/michaelc/stelloptPlusSfincs/equilibria/"+eq
proFile = "/global/homes/m/michaelc/stelloptPlusSfincs/profiles/"+profile

main_dir = os.getcwd()
os.chdir(os.environ['SCAN_DIR'])
with open(f"./sfincsScan_5", "r") as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    modified_line = line.replace(f"!ss NErs = 101", f"!ss NErs = {NErs}")
    modified_line = modified_line.replace(f"!ss ErMax = 0.020", f"!ss ErMax = {ErRange/1000}")
    modified_line = modified_line.replace(f"!ss ErMin = -0.020", f"!ss ErMin = {-ErRange/1000}")
    new_lines.append(modified_line)
with open(f"./sfincsScan_5", "w") as f:
    f.writelines(new_lines)
os.chdir(main_dir)

os.system("mkdir raderscan")
os.chdir("./raderscan")
os.system("python /global/homes/m/michaelc/stelloptPlusSfincs/stelloptPlusSfincs/run.py --profilesIn "+proFile+" --eqIn "+eqFile+" --radialVar 3 --Nzeta 37 --Ntheta 27 --Nxi 81 --driftScheme 0 --saveLoc "+os.getcwd()+" --time 00-0:20:00 --nNodes 1 --notifs all --noRun")

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
    os.system("python $SCAN_DIR/sfincsScan")

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

