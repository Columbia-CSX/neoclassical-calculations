import os
import argparse

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--rootChoice', type=str, required=True, help='low middle or high, which root to use (if only two, use low or middle for electron root)')

rootChoice = parser.parse_args().rootChoice

"""
to be run after running runRadErScan.py (in the same configuration directory)
"""
main_dir = os.getcwd()
os.system("python $UTILS_DIR/parseRadErScan.py")
os.chdir("./raderscan/determineEr")
os.system(f"python $UTILS_DIR/interpret.py --rootChoice {rootChoice}")
os.chdir(main_dir)
os.system("mkdir ambipolar")
os.chdir("./ambipolar")
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
