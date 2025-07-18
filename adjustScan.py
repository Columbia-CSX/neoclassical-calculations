import os
import numpy as np
import time

print("##### ADJUSTING ER #####")

rN, Er = np.load("scan_info.npy")

def fstring(x):
    return "{:.15e}".format(x).replace('e', 'd')

def edit_namelist(new_dir_name, rN_wish, Er):
    """
    goes into new_dir_name, edits the already present
    input.namelist file to have the correct Er
    """
    
    with open(f"./{new_dir_name}/input.namelist", "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        modified_line = line.replace("Er = 0.001 ! Seed", f"Er = {Er} ! Seed")
        new_lines.append(modified_line)

    with open(f"./{new_dir_name}/input.namelist", "w") as f:
        f.writelines(new_lines)


for i in range(0, len(rN)):
    rN_wish = rN[i]
    er = Er[i]
    new_dir_name = f"rN_{rN_wish:.4g}"
    edit_namelist(new_dir_name, rN_wish, er)
    print(f"##### Run {new_dir_name} adjusted to Er = {er*1000} [V] #####")  


