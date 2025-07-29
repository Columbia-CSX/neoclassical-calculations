"""
This script can be run standalone by editing the user section.
Another script, profiles.py, also calls this one and can be used
similarly. Using profiles.py is recommended (it handles more stuff for you)

If used on its own, this code generates a .opt BEAMS3D format
input file containing only information about profiles. This can
then processed by the stelloptPlusSfincs code to produce a
"profiles" file that sfincs can interpret.
"""
import numpy as np

example_array = np.array([0.0, 0.1, 0.2, 0.3])
# ─────── USER SECTION ─────────────────────────────────────────────────────────
# name for output file
outputfile = "outputfile.opt"
# input profiles specified as arrays
NE_AUX_S = example_array # XX_AUX_S files are radial coordinates for XX_AUX_F
NE_AUX_F = example_array # electron density in m^-3
TE_AUX_S = example_array
TE_AUX_F = example_array # electron temperature in eV
NI_AUX_S = example_array
# 2D array (need a profile for each particle species):
NI_AUX_F = np.vstack([   # ion density
    example_array,
    example_array,
    example_array
])
TI_AUX_S = example_array
TI_AUX_F = example_array # ion temperature (should be same shape as NI_AUX_F probably)
# 1D array of masses for each ion species -- order matters (should match NI_AUX_F)
# (note that electrons are not included here)
NI_AUX_M = np.array([3.343583746e-27, 5.008267660e-27, 6.646479070e-27])
NI_AUX_Z = np.array([1, 1, 2], dtype=np.int32) # charge numbers Z for each ion species
# ─────── END USER SECTION ─────────────────────────────────────────────────────

def fmt_fortran(vals):
    if isinstance(vals, (list, tuple, np.ndarray)):
        if vals.dtype != np.int32:
            return [f"{x:.10E}" for x in vals]
        return [f"{x}" for x in vals]
    else:
        raise ValueError("Unsupported parameter type provided.")

def write_profiles(
    outfile,
    NE_AUX_S, NE_AUX_F,
    TE_AUX_S, TE_AUX_F,
    NI_AUX_S, NI_AUX_F,
    TI_AUX_S, TI_AUX_F,
    NI_AUX_M, NI_AUX_Z
):
    header = (
        "&BEAMS3D_INPUT\n"
        "!--------PROFILES ----\n"
    )
    with open(outfile, 'w') as f:
        f.write(header)
        params = [
            ('NE_AUX_S', NE_AUX_S, 1),
            ('NE_AUX_F', NE_AUX_F, 1),
            ('TE_AUX_S', TE_AUX_S, 1),
            ('TE_AUX_F', TE_AUX_F, 1),
            ('NI_AUX_S', NI_AUX_S, 1),
            ('NI_AUX_F', NI_AUX_F, 2),
            ('TI_AUX_S', TI_AUX_S, 1),
            ('TI_AUX_F', TI_AUX_F, 1),
            ('NI_AUX_M', NI_AUX_M, 1),
            ('NI_AUX_Z', NI_AUX_Z, 1),
        ]
        for name, arr, ndim in params:
            arr = np.asarray(arr)
            if ndim == 1:
                formatted = fmt_fortran(arr)
                line = f"  {name} = {'     '.join(formatted)}\n"
                f.write(line)
            elif ndim == 2:
                for idx, row in enumerate(arr, start=1):
                    formatted = fmt_fortran(row)
                    line = f"  {name}({idx},:) = {'     '.join(formatted)}\n"
                    f.write(line)
            else:
                raise ValueError(f"Unsupported ndim for {name}: {arr.ndim}")
        f.write("/\n")

if __name__ == "__main__":
    write_profiles(
        outputfile,
        NE_AUX_S, NE_AUX_F,
        TE_AUX_S, TE_AUX_F,
        NI_AUX_S, NI_AUX_F,
        TI_AUX_S, TI_AUX_F,
        NI_AUX_M, NI_AUX_Z
    )
    print(f"Written {outputfile}")
