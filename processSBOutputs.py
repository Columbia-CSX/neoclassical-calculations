from processOutputs import *
from tqdm import tqdm

"""
meant to be called from the same folder runSBRadErScan.py was called in,
the directory containing esb_* folders.

contains plotting routines for comparing plasma parameters across different
values of esb
"""

if not os.path.exists("./plots"):
    os.system("mkdir plots")

def plotVvsesb(speciesIndex=0, omitPar=False, omitPerp=False, makesubplots=True):
    def extract_esb(file):
        try:
            return float(file.split("_")[1])
        except:
            return 999.9

    esbfiles = sorted([file for file in os.listdir() if file.startswith("esb")], key=extract_esb)
    main_dir = os.getcwd()
    fig = plt.figure(figsize=(10.5, 7))
    ax = fig.add_subplot()
    for esbfile in esbfiles:
        try:
            os.chdir(f"{esbfile}/ambipolar")
        except:
            raise SystemExit("No ambipolar directory detected.")

        radfiles = [file for file in os.listdir() if file.startswith("rN")]
        radialCoordinate, v = getVprofile(speciesIndex=speciesIndex, omitPar=omitPar, omitPerp=omitPerp, plot=makesubplots)
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
    ax.tick_params(axis="both", labelsize=17)
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
        # todo remove this two lines ^^ hardcoded fix for a failed file
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
    ax.tick_params(axis="both", labelsize=19)

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
    ax.set_ylabel("Neoclassical heat flux $Q_N$ [$Q_T$]", fontsize=22)
    ax.tick_params(axis="both", labelsize=19)
    fig.tight_layout()
    fig.savefig("./plots/Q_N_over_Q_T_vs_esb.jpeg", dpi=360)

    return esbs, Q_Ns, Q_Ts

def asNumber(string):
    try:
        num = float(string)
        num_int = int(string)
        if num - num_int == 0.0:
            num = num_int
        return num
    except:
        return string

def selectRadialCurrent(options, prompt):
    print(prompt)
    print("Select from the following options by using the index of the")
    print("desired option in the list (starting with 0) or by copying ")
    print("the entire string in full. Here are the options: ")
    print(options)
    user = 0.5
    while not asNumber(user) in options and not isinstance(asNumber(user), int):
        user = input("Your selection: ")

    if user in options:
        return user

    return options[asNumber(user)]

def plot_DeltaT_vs_rN_and_esb(speciesIndex=0):
    
    main_dir = os.getcwd()
    esbfiles = [file for file in os.listdir() if file.startswith("esb")]
    
    radialCurrentOptions = []
    for i, esbfile in enumerate(esbfiles):
        os.chdir(esbfile)
        currentFiles = [file for file in os.listdir() if file == "ambipolar" or file.startswith("Ir")]
        radialCurrentOptions.extend(currentFiles)
        os.chdir(main_dir)
    
    radialCurrentOptions = list(set(radialCurrentOptions))

    radialCurrentOption1 = selectRadialCurrent(radialCurrentOptions, "First radial current to use for plot_DeltaT_vs_rN_and_esb")
    radialCurrentOption2 = selectRadialCurrent(radialCurrentOptions, "Second radial current to use for plot_DeltaT_vs_rN_and_esb")
    print("Selected options:")
    print("\t I_r 1:", radialCurrentOption1)
    print("\t I_r 2:", radialCurrentOption2)
    
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot()
    for esbfile in esbfiles:
        try:
            os.chdir(f"{esbfile}/{radialCurrentOption1}")
        except:
            continue
        radialVar, vals1 = getDeltaTProfile(speciesIndex=speciesIndex)
        os.chdir(main_dir)
        try:
            os.chdir(f"{esbfile}/{radialCurrentOption2}")
        except:
            continue
        radialVar, vals2 = getDeltaTProfile(speciesIndex=speciesIndex)
        vals = []
        iters = max([len(vals1), len(vals2)])
        for i in range(iters):
            try:
                vals.append(vals2[i]-vals1[i])
            except:
                vals.append(None)
        _, esb = esbfile.split("_")
        esb = float(esb)
        esb = f"{esb:.3f}"
        plot_label = "$\epsilon_{sb}$ = "+f"{esb}"
        ax.plot(radialVar, vals, marker='o', label=plot_label)
        os.chdir(main_dir)

    ax.set_xlabel(r"$\sqrt{\psi_N}$", fontsize=22)
    ax.set_ylabel(r"$t_{fd}$ [$\mu$s]", fontsize=22)

    try:
        _, ir1 = radialCurrentOption1.split("_")
        ir1 = float(ir1)
        ir1 = f"{ir1:.8f}"
    except:
        ir1 = "0"
    
    try:
        _, ir2 = radialCurrentOption2.split("_")
        ir2 = float(ir2)
        ir2 = f"{ir2:.8f}"
    except:
        ir2 = "0"

    ax.set_title(f"Starting $I_r$ = {ir1} [A]\nEnding $I_r$ = {ir2} [A]", fontsize=22)
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"./plots/{speciesIndex}_DeltaT_{ir1.replace('.','_')}_{ir2.replace('.','_')}.jpeg", dpi=360)

def plotNTVvsErAndEsb(rNfile="rN_0.75"):
    main_dir = os.getcwd()
    def extract_esb(file):
        try:
            return float(file.split("_")[1])
        except:
            return 999.9

    esbfiles = sorted([file for file in os.listdir() if file.startswith("esb")], key=extract_esb)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()
    for esbfile in esbfiles:
        os.chdir(f"{esbfile}/raderscan")
        Ers, taus = getNTVvsEr(rNfile)
        taus = [tau*1000 for tau in taus]
        esb = extract_esb(esbfile)
        ax.plot(Ers, taus, label=r"$\epsilon_{sb} = $"+f"{esb:.3f}")
        os.chdir(main_dir)

    ax.set_xlabel("$E_r$ [V]", fontsize=22)
    ax.set_ylabel(r"$\tau_{NTV}$ [g$\text{m}^{-1}\text{s}^{-2}$]", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"plots/NTVvsErvsEsb_{rNfile}.jpeg", dpi=360)

def plotDeltaTvsErAndEsb(rNfile="rN_0.75", speciesIndex=0, spinUp=False):
    """
    spinUp == True   : \Delta t = l - l_ambipolar / tau
    spinUp == False  : \Delta t = l_ambipolar - l / tau_ambipolar	
    """
    main_dir = os.getcwd()
    def extract_esb(file):
        try:
            return float(file.split("_")[1])
        except:
            return 999.9
    rNval = float(rNfile.split("_")[1])
    esbfiles = sorted([file for file in os.listdir() if file.startswith("esb")], key=extract_esb)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot()
    for esbfile in tqdm(esbfiles, desc=f"Processing time scale at {rNfile}..."):
        os.chdir(f"{esbfile}/ambipolar")
        try:
            amdambipolar = valsafe(getFSAAngularMomentumDensity(rNfile, speciesIndex=speciesIndex))
            tauambipolar = valsafe(getNTVTorque(rNfile))
        except:
            os.chdir(main_dir)
            continue
        os.chdir(main_dir)
        os.chdir(f"{esbfile}/raderscan")
        Ers, taus, amd = getNTVvsEr(rNfile, returnAMD=True, speciesIndex=speciesIndex)
        DeltaTs = []
        if spinUp:
            for i in range(len(Ers)):
                DeltaTs.append(((amd[i]-amdambipolar)/taus[i]))
        if not spinUp:
            for i in range(len(Ers)):
                DeltaTs.append(abs((amdambipolar - amd[i])/tauambipolar))
        esb = extract_esb(esbfile)
        ax.plot(Ers, DeltaTs, label=r"$\epsilon_{sb} = $"+f"{esb:.3f}")
        os.chdir(main_dir)

    extra_plot_label = ["Electron", "Ion"][speciesIndex] + " " + ["spin down", "spin up"][int(spinUp)] + " "
    ax.set_xlabel("$E_r$ [V]", fontsize=22)
    ax.set_ylabel(extra_plot_label + r"$t_{fd}$ [s]", fontsize=22)
    ax.set_yscale('log')
    ax.set_title(r"$\sqrt{\psi_N}$ ="+f"{rNval}", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"plots/DeltaTvsErvsEsb_{speciesIndex}_{rNfile}_{spinUp}.jpeg", dpi=360)

if __name__ == "__main__":
    """
    plotVvsesb()
    plotVvsesb(speciesIndex=1)
    plotVvsesb(speciesIndex=0, omitPar=True)
    plotVvsesb(speciesIndex=1, omitPar=True)
    plotVvsesb(speciesIndex=0, omitPerp=True)
    plotVvsesb(speciesIndex=1, omitPerp=True)
    """
    plotDeltaTvsErAndEsb(rNfile="rN_0.35", speciesIndex=1, spinUp=False)
    plotDeltaTvsErAndEsb(rNfile="rN_0.75", speciesIndex=1, spinUp=False)
    plotDeltaTvsErAndEsb(rNfile="rN_0.35", speciesIndex=1, spinUp=True)
    plotDeltaTvsErAndEsb(rNfile="rN_0.75", speciesIndex=1, spinUp=True)
    plotDeltaTvsErAndEsb(rNfile="rN_0.35", speciesIndex=0, spinUp=False)
    plotDeltaTvsErAndEsb(rNfile="rN_0.75", speciesIndex=0, spinUp=False)
    plotDeltaTvsErAndEsb(rNfile="rN_0.35", speciesIndex=0, spinUp=True)
    plotDeltaTvsErAndEsb(rNfile="rN_0.75", speciesIndex=0, spinUp=True)

