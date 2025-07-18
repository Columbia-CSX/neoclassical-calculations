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

main_dir = os.getcwd()

print("=== Beginning runSBAmbipolar ===")
esb_dirs = [folder for folder in os.listdir() if folder.startswith("esb")]

if getIr:
    print("=== Parsing scanned radial current values ===")
    Maxdatas = []
    Mindatas = []
    for folder in esb_dirs:
        os.chdir(f"./{folder}/raderscan/determineEr")
        files = os.listdir()
        maxdatas = []
        mindatas = []
        for file in files:
            if file.endswith('Ir-vs-Er.dat'):
                Ir = np.transpose(np.loadtxt(file))[1]
                maxdatas.append(max(Ir))
                mindatas.append(min(Ir))
        Maxdatas.append(min(maxdatas))
        Mindatas.append(max(mindatas))
        os.chdir(main_dir)

    Ir_max = min(Maxdatas)
    Ir_min = max(Mindatas)

    if abs(Ir_max) >= abs(Ir_min):
        Ir_to_use = Ir_max - Ir_max*0.05

    if abs(Ir_min) > abs(Ir_max):
        Ir_to_use = Ir_min - Ir_min*0.05

    print(f"=== Enforcing Ir: {Ir_to_use} [A] ===")

if not getIr:
    Ir_to_use = Ir

for folder in esb_dirs:
    os.chdir(f"./{folder}")
    print(f"=== Starting runAmbipolar on {folder} ===")
    os.system(f"python $UTILS_DIR/runAmbipolar.py --rootChoice {rootChoice} --Ir {Ir_to_use}")
    os.chdir(main_dir)
    print(f"=== runAmbipolar ended on {folder} ===")

print("=== End runSBAmbipolar ===")

