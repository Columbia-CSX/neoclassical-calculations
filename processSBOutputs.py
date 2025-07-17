from processOutputs import *

"""
meant to be called from the same folder runSBRadErScan.py was called in,
the directory containing esb_* folders.

contains plotting routines for comparing plasma parameters across different
values of esb
"""

if not os.path.exists("./plots"):
    os.system("mkdir plots")

def plotVvsesb(speciesIndex=0, omitPar=False, omitPerp=False, makesubplots=True):
    esbfiles = [file for file in os.listdir() if file.startswith("esb")]
    main_dir = os.getcwd()
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()
    for esbfile in esbfiles:
        try:
            os.chdir("esbfile/ambipolar")
        except:
            raise SystemExit("No ambipolar directory detected.")

        radfiles = [file for file in os.listdir() if file.startswith("rN")]
        radialCoordinate, v = getVprofile(speciesIndex=0, omitPar=omitPar, omitPerp=omitPerp, plot=makesubplots)
        _, esb = esbfile.split("_")
        esb = float(esb)
        esb = f"{esb:.3f}"
        ax.plot(radialCoordinate, v, label=f"$\epsilon_{sb} = ${esb}")
        os.chdir(main_dir)

    ylabel = ""
    species = ["Electron ", "Ion "][speciesIndex]
    ylabel = ylabel + species
    if omitPar:
        ylabel = ylabel + "perpendicular "
    if omitPerp:
        ylabel = ylabel + "parallel "
    filename = ylabel + "flow velocity and esb"
    ylabel = ylabel + "flow velocity [km/s]"

    ax.set_xlabel("$\sqrt{\psi_N}$", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    fig.tight_layout()
    fig.savefig(f"plots/{filename}_vs_rN.jpeg")


if __name__ == "__main__":
    plotVvsesb()

