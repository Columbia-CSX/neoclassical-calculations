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
            os.chdir(f"{esbfile}/ambipolar")
        except:
            raise SystemExit("No ambipolar directory detected.")

        radfiles = [file for file in os.listdir() if file.startswith("rN")]
        radialCoordinate, v = getVprofile(speciesIndex=0, omitPar=omitPar, omitPerp=omitPerp, plot=makesubplots)
        _, esb = esbfile.split("_")
        esb = float(esb)
        esb = f"{esb:.3f}"
        plot_label = "$\epsilon_{sb}$ = "+f"{esb}"
        ax.plot(radialCoordinate, np.array(valsafe(v))/1000, marker='o', label=plot_label)
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
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"plots/{filename}_vs_rN.jpeg")

def plotHeat():
    main_dir = os.getcwd()
    Q_Ns = []
    Q_Ts = []
    esbs = []

    for file in [file for file in os.listdir() if file.startswith("esb")]:
        if file == "esb_0.8":
            continue
        # hardcoded fix to a failed file
        os.chdir(f"{file}/ambipolar")
        Q_N, Q_T = make_qlcfs_file()
        _, esb = file.split("_")
        esb = float(esb)
        Q_Ns.append(valsafe(Q_N))
        Q_Ts.append(valsafe(Q_T))
        esbs.append(esb)
        os.chdir(main_dir)

    _, Q_Ns = zip(*sorted(zip(esbs, Q_Ns), key=lambda pair: pair[0]))
    esbs, Q_Ts = zip(*sorted(zip(esbs, Q_Ts), key=lambda pair: pair[0]))

    esbs = list(esbs)
    Q_Ns = np.array(list(Q_Ns))
    Q_Ts = np.array(list(Q_Ts))

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()
    ax.plot(esbs, Q_Ns, marker='o', color=columbia)
    ax.set_xlabel("$\epsilon_{sb}$", fontsize=22)
    ax.set_ylabel("Neoclassical power $Q_N$ [W]", fontsize=22)
    fig.tight_layout()
    fig.savefig("./plots/Q_N_vs_esb.jpeg", dpi=360)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()
    ax.plot(esbs, Q_Ts, marker='o', color=columbia)
    ax.set_xlabel("$\epsilon_{sb}$", fontsize=22)
    ax.set_ylabel("Turbulent power $Q_T$ [W]", fontsize=22)
    fig.tight_layout()
    fig.savefig("./plots/Q_T_vs_esb.jpeg", dpi=360)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()
    ax.plot(esbs, Q_Ns/Q_Ts, marker='o', color=columbia)
    ax.set_xlabel("$\epsilon_{sb}$", fontsize=22)
    ax.set_ylabel("Neoclassical power $Q_N$ [$Q_T$]", fontsize=22)
    fig.tight_layout()
    fig.savefig("./plots/Q_N_over_Q_T_vs_esb.jpeg", dpi=360)


if __name__ == "__main__":
    """
    plotVvsesb()
    plotVvsesb(speciesIndex=1)
    plotVvsesb(speciesIndex=0, omitPar=True)
    plotVvsesb(speciesIndex=1, omitPar=True)
    plotVvsesb(speciesIndex=0, omitPerp=True)
    plotVvsesb(speciesIndex=1, omitPerp=True)
    """
    plotHeat()

