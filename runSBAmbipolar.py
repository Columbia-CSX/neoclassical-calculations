import os
import argparse

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--rootChoice', type=str, required=True, help='low middle or high, which root to use (if only two, use low or middle for electron root)')

rootChoice = parser.parse_args().rootChoice

main_dir = os.getcwd()

esb_dirs = os.listdir()
for folder in esb_dirs:
    os.chdir(f"./{folder}")
    os.system("ls")
    os.system(f"python $UTILS_DIR/runAmbipolar.py --rootChoice {rootChoice}")
    os.chdir(main_dir)

