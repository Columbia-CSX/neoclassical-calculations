import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--rootChoice', type=str, required=True, help='low middle or high, which root to use (if only two, use low or middle for electron root)')
parser.add_argument('--Ir', type=float, default=0.0, help='radial current to enforce, in Amperes')
parser.add_argument('--getIr', action=argparse.BooleanOptionalAction)
parser.set_defaults(feature=False)

rootChoice = parser.parse_args().rootChoice
Ir = parser.parse_args().Ir
getIr = parser.parse_args().getIr

"""
to be run after running runRadErScan.py (in the same configuration directory)
"""
main_dir = os.getcwd()
print(f"~~ runAmbipolar.py starting in the following directory ~~")
print(f"\t {main_dir}")

if getIr:
    try:
        os.chdir("raderscan/determineEr")
    except:
        raise SystemExit("An ambipolar scan should be done before any other radial current scan")
    files = os.listdir()
    maxdatas = []
    mindatas = []
    for file in files:
        if file.endswith('Ir-vs-Er.dat'):
            Irs = np.transpose(np.loadtxt(file))[1]
            maxdatas.append(max(Irs))
            mindatas.append(min(Irs))
    
    Ir_max = min(maxdatas)
    Ir_min = max(mindatas)
    
    if abs(Ir_max) >= abs(Ir_min):
        Ir_to_use = Ir_max - Ir_max*0.05

    if abs(Ir_min) > abs(Ir_max):
        Ir_to_use = Ir_min - Ir_min*0.05

    os.chdir(main_dir)

if not getIr:
    Ir_to_use = Ir

os.system("python $UTILS_DIR/parseRadErScan.py")
os.chdir("./raderscan/determineEr")
os.system(f"python $UTILS_DIR/interpret.py --rootChoice {rootChoice} --Ir {Ir_to_use}")
user = None
while not user in ["y", "n"]:
    user = input("\n Check the Er value -- would you like to run sfincs at this Er? [y/n]")
if user == "n":
    raise SystemExit
os.chdir(main_dir)
print(Ir)
if Ir_to_use == 0.0:
    new_dir_name = 'ambipolar'
if Ir_to_use != 0.0:
    new_dir_name = f'Ir_{Ir_to_use}'
os.system(f"mkdir {new_dir_name}")
os.chdir(f"./{new_dir_name}")
for file in ["input.namelist", "profiles", "job.sfincsScan", "determineEr/scan_info.npy"]:
    os.system(f"cp ../raderscan/{file} ./")

os.system("cp $UTILS_DIR/adjustScan.py ./")

with open("./input.namelist", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    modified_line = line.replace("!ss scanType = 5", "!ss scanType = 4")
    new_lines.append(modified_line)

with open("./input.namelist", "w") as f:
    f.writelines(new_lines)

os.system("python $SCAN_DIR/sfincsScan")
