import os
import argparse

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--rootChoice', type=str, required=True, help='low middle or high, which root to use (if only two, use low or middle for electron root)')
parser.add_argument('--Ir', type=float, default=0.0, help='radial current to enforce, in Amperes')

rootChoice = parser.parse_args().rootChoice
Ir = parser.parse_args().Ir

main_dir = os.getcwd()

print("=== Beginning runSBAmbipolar ===")
esb_dirs = [folder for folder in os.listdir() if folder.startswith("esb")]
for folder in esb_dirs:
    os.chdir(f"./{folder}")
    print(f"=== Starting runAmbipolar on {folder} ===")
    os.system(f"python $UTILS_DIR/runAmbipolar.py --rootChoice {rootChoice} --Ir {Ir}")
    os.chdir(main_dir)
    print(f"=== runAmbipolar ended on {folder} ===")

print("=== End runSBAmbipolar ===")

