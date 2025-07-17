import os
import argparse

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--rootChoice', type=str, required=True, help='low middle or high, which root to use (if only two, use low or middle for electron root)')
parser.add_argument('--Ir', type=float, default=None, help='radial current to enforce, in Amperes')

rootChoice = parser.parse_args().rootChoice
Ir = parser.parse_ars().Ir

"""
to be run after running runRadErScan.py (in the same configuration directory)
"""
main_dir = os.getcwd()
print(f"~~ runAmbipolar.py starting in the following directory ~~")
print(f"\t {main_dir}")
os.system("python $UTILS_DIR/parseRadErScan.py")
os.chdir("./raderscan/determineEr")
os.system(f"python $UTILS_DIR/interpret.py --rootChoice {rootChoice} --Ir {Ir}")
user = None
while not user in ["y", "n"]:
    user = input("\n Check the Er value -- would you like to run sfincs at this Er? [y/n]")
if user == "n":
    raise SystemExit
os.chdir(main_dir)
if Ir is None:
    new_dir_name = 'ambipolar'
if Ir is not None:
    new_dir_name = f'Ir_{Ir}'
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
