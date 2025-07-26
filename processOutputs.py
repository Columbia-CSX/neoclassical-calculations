import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from tqdm import tqdm
from astropy import units as u
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from scipy.io import savemat
from simsoptutils import *
from scipy.fft import fft2, ifft2
import pickle
import matplotlib.colors as color

# uses the parula colormap if nmmn library is available (I like it)
try:
    from nmmn.plots import parulacmap
    parula = parulacmap()
    parulacmap = parula
except:
    parulacmap = "plasma"

# cu blue
columbia = "#012169"

"""
processes 'sfincsOutput.h5' files present in the directory.
this script should be called from the directory containing the
rN_0.xxxx directories (sfincsScan_4 run with adjustScript)
"""

if not os.path.exists("./plots"):
    os.system("mkdir plots")

### utility functions ###

def index_along_dim(arr, dim, idx):
    indexer = [slice(None)] * arr.ndim
    indexer[dim] = idx
    return arr[tuple(indexer)]

def getPathToWout():
    folders = os.listdir()
    main_dir = os.getcwd()

    for folder in folders:
        if folder.startswith("rN") or folder == "input.namelist":
            try:
                os.chdir(f"./{folder}")
            except:
                pass
            with open(f"./input.namelist", "r") as f:
                lines = f.readlines()

            for line in lines:
                line_list = line.split(" ")
                if line_list[0] == "\tequilibriumFile":
                    wout_file = line_list[2].replace('"', "").replace(".bc\n", ".nc")
                    os.chdir(main_dir)
                    return wout_file

"""
Each parameter of interest from sfincs is given a corresponding Parameter object.
It stores information about what the name of the key for the parameter is in the HDF5
file, what the .data of the parameter actually returns after processing, and the scaling 
which is applied to the parameter as it is processed.
"""
class Parameter:
    def __init__(self, name, label, scaling=1.0, speciesIndex=None):
        self.name = name
        self.label = label
        self.scaling = scaling
        self.speciesIndex = speciesIndex
        self.data = None

    def speciate(self):
        assert self.data is not None, "No data to separate into species"
        try:
            shape = list(self.data.shape)
            speciesDim = shape.index(2)
            self.data = np.squeeze(index_along_dim(self.data, speciesDim, self.speciesIndex))
        except:
            try:
                self.data = np.squeeze(self.data) # necessary to remove 'iterations' dimension of size 1, may need updating to include nonlinear solving method
            except:
                pass
        return self


### Normalization constants and other constants used in sfincs ###
c = 299792458.0*u.m/u.s#m/s
e = 1.602176634e-19*u.C #C
RBar = 1.0*u.m #m
nBar = 1e20/u.m/u.m/u.m #m^-3
BBar = 1.0*u.T #Tesla
TBar = 1000.0*u.eV #eV
phiBar = 1000*u.V #Volts
mBar = 1.67262192595e-27*u.kg #kg
vBar = np.sqrt(2*TBar/mBar).to(u.m/u.s) #m/s

### Parameters supported ###
rN = Parameter("rN", r"$\sqrt{\psi_N}$")
psiN = Parameter("psiN", r"$\psi_N$")
FSAbootstrapCurrentDensity = Parameter("FSABjHatOverB0", r"Bootstrap Current $\langle \mathbf{j}\cdot\mathbf{B} \rangle/B_{00}$ [A$\mathrm{m}^{-2}$]"                                       , scaling=e*vBar*nBar)
Er = Parameter("Er", r"$E_r$ [V/m]", phiBar/RBar)
flow_i = Parameter("flow", r"Ion parallel flow velocity $\int d^3v v_{||} f_i$ [m/s]", scaling=nBar*vBar, speciesIndex=1)
flow_e = Parameter("flow", r"Electron parallel flow velocity $\int d^3v v_{||} f_i$ [m/s]", scaling=nBar*vBar, speciesIndex=0)
theta = Parameter("theta", r"$\theta$")
zeta = Parameter("zeta", r"$\zeta$")
n_e = Parameter("totalDensity", r"Electron density $n_e$ [\mathrm{m}^{-3}]", scaling=nBar, speciesIndex=0)
n_i = Parameter("totalDensity", r"Ion density $n_i$ [\mathrm{m}^{-3}]", scaling=nBar, speciesIndex=1)
vPar_i = Parameter("velocityUsingTotalDensity", r"Ion parallel flow velocity $\frac{1}{n_i}\int d^3v v_{||} f_i$ [m/s]", scaling=vBar, speciesIndex = 1)
vPar_e = Parameter("velocityUsingTotalDensity", r"Electron parallel flow velocity $\frac{1}{n_e}\int d^3v v_{||} f_e$ [m/s]", scaling=vBar, speciesIndex = 0)
VPrime = Parameter("VPrimeHat", r"$V'$ [m/T]", scaling=RBar/BBar)
heatFlux_vm_psi_e = Parameter("heatFlux_vm_psiHat", r"Electron heat flux Q = whatever", scaling=nBar*vBar*vBar*vBar*BBar*mBar*RBar, speciesIndex = 0)
heatFlux_vm_psi_i = Parameter("heatFlux_vm_psiHat", r"Ion heat flux Q = whatever", scaling=nBar*vBar*vBar*vBar*BBar*mBar*RBar, speciesIndex = 1)
B = Parameter("BHat", r"Magnetic field strength $|B|$ [T]", scaling=BBar)
B0 = Parameter("B0OverBBar", r"$|B_0|$ [T]", scaling=BBar)
T_i = Parameter("THats", r"Ion temperature $T_i$ [eV]", scaling=TBar, speciesIndex=1)
T_e = Parameter("THats", r"Electron temperature $T_e$ [eV]$", scaling=TBar, speciesIndex=0)
rough_n_i = Parameter("nHats", r"Ion density $n_i$ [$\mathrm{m}^{-3}$]", scaling=nBar, speciesIndex=1)
a = Parameter("aHat", r"Minor radius $a$ [m]", scaling=RBar)
dT_idr = Parameter("dTHatdrHat", r"Ion temperature gradient $dT_i/dr$ [eV/m]", scaling=TBar/RBar, speciesIndex=1)
Delta = Parameter("Delta", r"Reference parameter")
Bsuptheta = Parameter("BHat_sup_theta", r"$B^\theta$ [T/M]", scaling = BBar/RBar)
Bsupzeta = Parameter("BHat_sup_zeta", r"$B^\zeta$ [T/M]", scaling = BBar/RBar)
iota = Parameter("iota", r"Rotational transform $\iota$")
G = Parameter("GHat", r"Covariant toroidal Boozer component", scaling = BBar * RBar)
I = Parameter("IHat", r"Covariant poloidal Boozer component", scaling = BBar * RBar)
dPhidpsi = Parameter("dPhiHatdpsiHat", r"$\frac{\partial \Phi}{\partial \psi}$", scaling = phiBar / (BBar * RBar * RBar))
dn_edpsi = Parameter("dnHatdpsiHat", r"blah blah", scaling = nBar / (BBar * RBar * RBar), speciesIndex=0)
dn_idpsi = Parameter("dnHatdpsiHat", r"blah blah", scaling = nBar / (BBar * RBar * RBar), speciesIndex=1)
dT_edpsi = Parameter("dTHatdpsiHat", r"blah blah", scaling = TBar / (BBar * RBar * RBar), speciesIndex=0)
dT_idpsi = Parameter("dTHatdpsiHat", r"blah blah", scaling = TBar / (BBar * RBar * RBar), speciesIndex=1)
eFlux_vm_psi = Parameter("particleFlux_vm_psiHat", r"Electron radial flux $\langle \int d^3 v f_e \mathbf{v}\cdot \nabla\psi\rangle$ [T/ms]", scaling=nBar*vBar*RBar*BBar, speciesIndex=0)
iFlux_vm_psi = Parameter("particleFlux_vm_psiHat", r"Ion radial flux $\langle \int d^3 v f_i \mathbf{v}\cdot \nabla\psi\rangle$ [T/ms]", scaling=nBar*vBar*RBar*BBar, speciesIndex=1)
totalHeatFlux_e = Parameter("heatFluxBeforeSurfaceIntegral_vm", r"Total heat flux", scaling=nBar*vBar*vBar*vBar*BBar*mBar*RBar, speciesIndex=0)

def valsafe(quantity):
    try:
        return quantity.value
    except:
        try:
            ls = []
            for q in quantity:
                ls.append(valsafe(q))
            try:
                return np.array(ls)
            except:
                return ls
        except:
            return quantity

def parseHDF5(folder, parameter, ignoreFileFalure=False):
    f = h5py.File(f"./{folder}/sfincsOutput.h5", 'r')
    assert parameter.name in list(f.keys()), f"invalid sfincs parameter {parameter.name} provided to parseHDF5\n if this is a valid parameter, this error indicates a failed sfincs run."
    paramDataset = f[parameter.name]
    parameter.data = paramDataset[()]*parameter.scaling
    return parameter.speciate()
    
def plotProfile(parameter, radialCoordinate, plot=True, sort=True):
    dirfiles = os.listdir()
    radialCoords = []
    vals = []
    for file in dirfiles:
        if not file.startswith("rN"):
            continue
        radialCoords.append(parseHDF5(f"{file}", radialCoordinate).data)
        vals.append(parseHDF5(f"{file}", parameter).data)
    
    if sort:
        radialCoords, vals = zip(*sorted(zip(radialCoords, vals), key=lambda pair: pair[0]))
        radialCoords = list(radialCoords)
        vals = list(vals)

    if plot:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot()
        ax.plot(valsafe(radialCoords), valsafe(vals), color='#012169', alpha=0.7)
        ax.plot(valsafe(radialCoords), valsafe(vals), marker='o', linestyle='None', color='#012169')
        ax.set_xlabel(radialCoordinate.label, fontsize=24)
        ax.set_ylabel(parameter.label, fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        fig.savefig(f"./plots/{parameter.name}_vs_{radialCoordinate.name}.jpeg", dpi=320)
        print(f"Plot {parameter.name}_vs_{radialCoordinate.name} saved.")

    return radialCoords, vals

def plotHeatmap(folder, parameter, save=True, specificDirectory=None, verb=True, nametag=None, contourLevels=None, cmap="viridis"):
    parameter = parseHDF5(folder, parameter)
    zetas = parseHDF5(folder, zeta)
    thetas = parseHDF5(folder, theta)
    Z, T = np.meshgrid(zetas.data, thetas.data)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot()
    ax.set_title(folder, fontsize=24)
    if contourLevels is not None:
        p=plt.contourf(Z, T, np.transpose(valsafe(parameter.data)), levels=contourLevels, cmap=cmap)
    else:
        p=plt.contourf(Z, T, np.transpose(valsafe(parameter.data)), cmap=cmap)
    cbar = plt.colorbar(p, cmap=cmap)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(parameter.label, size=24)
    ax.set_xlabel(zetas.label, fontsize=24)
    ax.set_ylabel(thetas.label, fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    if specificDirectory is None:
        specificDirectory = "plots"
    if save:
        fig.savefig(f"./{specificDirectory}/{parameter.name}{parameter.speciesIndex}_heatmap{nametag}.png", dpi=320)
        if verb:
            print(f"Plot {parameter.name}{parameter.speciesIndex}_heatmap{nametag} saved.")
    plt.close(fig)
    return Z, T, parameter

def makeHeatmapGif(parameter, radialCoordinate, contourLevels=None, cmap="turbo"):
    dirfiles = os.listdir()
    radialCoords = []
    imageFiles = []
    if os.path.exists("./gif_files"):
        os.system("rm -r gif_files")
    os.system("mkdir gif_files")
    for i, file in enumerate(tqdm(dirfiles, desc="    Writing image files...")):
        if not file.startswith("rN"):
            continue
        radialCoords.append(parseHDF5(f"{file}", radialCoordinate).data)
        plotHeatmap(file, parameter, specificDirectory="gif_files", nametag=f"{i}", verb=False, contourLevels=contourLevels, cmap=cmap)
        imageFiles.append(f"./gif_files/{parameter.name}{parameter.speciesIndex}_heatmap{i}.png")

    radialCoords, imageFiles = zip(*sorted(zip(radialCoords, imageFiles), key=lambda pair: pair[0]))
    radialCoords = list(radialCoords)
    imageFiles = list(imageFiles)
    
    with imageio.get_writer(f'{parameter.name}{parameter.speciesIndex}.gif', mode='I', fps=2) as writer:
        for filename in tqdm(imageFiles, desc="    Assembling .gif...    "):
            image = imageio.imread(filename)
            writer.append_data(image)
        print("    Writing to file...")

    os.system("rm -r gif_files")
    print(f"Created {parameter.name}{parameter.speciesIndex}.gif.")
    
def getLCFS():
    return "_".join(sorted([s.split("_") for s in [f for f in os.listdir() if f.startswith("rN")]], key= lambda pair: float(pair[1]))[-1])

def getNeoclassicalHeatFlux(folder):
    hf_e = parseHDF5(folder, heatFlux_vm_psi_e)
    print(hf_e.data)
    hf_i = parseHDF5(folder, heatFlux_vm_psi_i)
    print(hf_i.data)
    V = parseHDF5(folder, VPrime)
    return ((hf_e.data+hf_i.data)*V.data).to(u.W)
    
def getGyroBohmHeatFluxEstimate(folder):
    Q_GB = 1.0
    surf_area = 2*u.m*u.m # guess of surface area of LCFS
    Ti = parseHDF5(folder, T_i).data.to(u.kg*u.m*u.m/u.s/u.s)
    mi = 3.343e-27*u.kg
    dTidr = parseHDF5(folder, dT_idr).data.to(u.kg*u.m/u.s/u.s)
    ni = parseHDF5(folder, rough_n_i).data
    mr = parseHDF5(folder, a).data
    mf = parseHDF5(folder, B0).data
    Q_GB *= (mi*np.sqrt(2*Ti/mi)/(e*mf))**2
    Q_GB *= np.sqrt(2*Ti/mi)
    Q_GB *= abs(dTidr)**3
    Q_GB *= abs(Ti)**(-2)
    Q_GB *= ni*mr*surf_area
    #print("L_T", Ti/dTidr, "T_i", Ti, "dT_i/dr", dTidr, "n_i", ni, "a", mr, "B", mf)

    # new estimate
    """
    Q_GB = 1.0
    Q_GB *= (mi/(e*mf))**2
    Q_GB *= (2*Ti/mi)**(3/2)
    Q_GB *= abs(dTidr)*ni*mr
    """
    return Q_GB.to(u.W)
    
def make_qlcfs_file(lcfs=None):
    print("Making qlcfs.npy...")
    if lcfs is None:
        lcfs = getLCFS()
    Q_N = getNeoclassicalHeatFlux(lcfs)
    Q_T = getGyroBohmHeatFluxEstimate(lcfs)
    print(f"  Neoclassical heat flux (LCFS)-- {Q_N}")
    print(f"  Turbulent heat flux (LCFS)----- {Q_T}")
    np.save("qlcfs.npy", np.array(valsafe([Q_N, Q_T])))
    print("qlcfs.npy saved.")
    return Q_N, Q_T

def makeCSXSurface(folder, colorparam=None, savematlab=True, plotname="defaultplotname"):
    psi_N = parseHDF5(folder, psiN).data
    thetas = parseHDF5(folder, theta).data
    zetas1 = parseHDF5(folder, zeta).data
    zetas2 = zetas1 + np.pi
    zetas = np.concatenate((zetas1, zetas2))
    thetas = np.append(thetas, thetas[0])
    zetas = np.append(zetas, zetas[0])

    THETA, ZETA = np.meshgrid(thetas, zetas)
    PSIN = np.ones_like(ZETA)*psi_N

    if colorparam is not None:
        if isinstance(colorparam, Parameter):
            c = valsafe(parseHDF5(folder, colorparam).data)
            c = np.vstack((c, c))
            plotlabel = colorparam.name
        else:
            c = valsafe(colorparam)
            plotlabel = plotname.replace(" ", "")
            plotlabel = plotlabel.replace("[","")
            plotlabel = plotlabel.replace("]", "")
            plotlabel = plotlabel.replace("/", "_")
        c_h = np.atleast_2d(c[:, 0]).T
        c = np.hstack((c, c_h))
        c_v = np.atleast_2d(c[0, :])
        C = np.vstack((c, c_v))
        
    bri = getBoozerRadialInterpolant(getPathToWout())
    points = unrollMeshgrid(PSIN, THETA, ZETA)
    points = np.ascontiguousarray(points, dtype=np.float64)
    points = cartesian(bri, points)
    points = np.ascontiguousarray(points, dtype=np.float64)
    X, Y, Z = rollMeshgrid(len(zetas), len(thetas), points)
    if colorparam is None:
        C = X #just to give it some color
        plotlabel = "X"
    if savematlab is True:
        print(os.getcwd())
        mdic = {"X": X, "Y": Y, "Z": Z, "C": C}
        savemat(f"./plots/csxSurface_{folder}_{plotlabel}.mat", mdic)

def makePeriodic(thetas, zetas, paramdata):
    thetas = np.append(thetas, 2*np.pi)
    zetas = np.concatenate((zetas, zetas+np.pi))
    zetas = np.append(zetas, 2*np.pi)
    c = paramdata
    c = np.vstack((c, c))
    c_h = np.atleast_2d(c[:, 0]).T
    c = np.hstack((c, c_h))
    c_v = np.atleast_2d(c[0, :])
    C = np.vstack((c, c_v))
    return thetas, zetas, C

def scale_by_epsb(B, epsb):
    """
    given a magnetic field B, returns the B corresponding
    to the 1/B^2 modified by scaling the symmetry-breaking 
    modes by epsb
    """
    FinvB2 = fft2(1/(B**2))
    FinvB2_copy = np.ones_like(FinvB2)*epsb
    FinvB2_copy[:, 0] = np.ones_like(FinvB2_copy[:, 0])
    return np.real(ifft2(FinvB2*FinvB2_copy))**(-0.5)

def getQSError(B):
    """
    compute [ \sum_{m, n\neq 0} (B_{mn}/B_{00})^2 ]^1/2
    note that positive and negative m and n are included
    here, QS means QA
    """
    Bmn = fft2(B)
    return np.sqrt(np.sum(abs((Bmn/Bmn[0, 0])[:, 1:])**2))

def getUnitVectorB(folder, rollover=False):
    psi_N = parseHDF5(folder, psiN).data
    thetas = parseHDF5(folder, theta).data
    zetas1 = parseHDF5(folder, zeta).data
    zetas2 = zetas1 + np.pi
    zetas = np.concatenate((zetas1, zetas2))
    if rollover:
        thetas = np.append(thetas, thetas[0])
        zetas = np.append(zetas, zetas[0])

    THETA, ZETA = np.meshgrid(thetas, zetas)
    PSIN = np.ones_like(ZETA)*psi_N
    
    parameter_outputs_list = []
    for param in [B, Bsupzeta, Bsuptheta]:
        c = valsafe(parseHDF5(folder, param).data)
        c = np.vstack((c, c))
        if rollover:
            c_h = np.atleast_2d(c[:, 0]).T
            c = np.hstack((c, c_h))
            c_v = np.atleast_2d(c[0, :])
            C = np.vstack((c, c_v))
        parameter_outputs_list.append(C)
    
    modB, Bszeta, Bstheta = parameter_outputs_list
    bri = getBoozerRadialInterpolant(getPathToWout())
    points = unrollMeshgrid(PSIN, THETA, ZETA)
    points = np.ascontiguousarray(points, dtype=np.float64)
    moddrdtheta, moddrdzeta = getGradientMagnitudes(bri, points)
    moddrdtheta = rollMeshgrid(len(zetas), len(thetas), moddrdtheta)
    moddrdzeta = rollMeshgrid(len(zetas), len(thetas), moddrdzeta)
    btheta = Bstheta*moddrdtheta/modB
    bzeta = Bszeta*moddrdzeta/modB
    return THETA, ZETA, btheta, bzeta

def makeStreamPlot(folder, colorparam=B):
    THETA, ZETA, btheta, bzeta = getUnitVectorB(folder)
    modB = valsafe(parseHDF5(folder, colorparam).data)
    modB = np.vstack((modB, modB)).T
    thetas = THETA[0]
    zetas = ZETA[:,0]
    thetas = thetas[:-1]
    zetas = zetas[:-1]
    bzeta = bzeta[:-1, :-1].T
    btheta = btheta[:-1, :-1].T
    C = np.hypot(bzeta, btheta)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot()
    strm = ax.streamplot(zetas, thetas, modB*bzeta, modB*btheta, color=modB)
    fig.colorbar(strm.lines)
    fig.show()
    fig.savefig(f"./plots/stream_{colorparam.name}_{colorparam.speciesIndex}.jpeg")

def makeQuiverPlotUnitB(folder):
    THETA, ZETA, btheta, bzeta = getUnitVectorB(folder)
    C = np.hypot(bzeta, btheta)
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot()
    quiv = ax.quiver(ZETA, THETA, bzeta/C, btheta/C, C, headlength=3.0)
    fig.colorbar(quiv)
    fig.show()
    fig.savefig("./plots/unitbquivC.jpeg")

def getFullV(folder, speciesIndex=0, omitPar=False, omitPerp=False, plot=False, forceRedo=False):
    filename = f"./flows/{folder.replace('.', '_')}_fullV_{speciesIndex}_perp_{omitPerp}_par_{omitPar}.pkl"
    if os.path.exists(filename) and not forceRedo:
        with open(filename, "rb") as f:
            try:
                v_theta, v_zeta, modv, _, __, ___ = pickle.load(f)
                return v_theta, v_zeta, modv
            except:
                print("Unable to recover flow data from the following file:")
                print("\t"+f"{filename}")
                print("(Re)calculating flow data...")

    thetas = parseHDF5(folder, theta).data
    zetas = parseHDF5(folder, zeta).data
    zetas = np.concatenate((zetas, zetas + np.pi))
    psi_N = parseHDF5(folder, psiN).data
    
    THETAS, ZETAS = np.meshgrid(thetas, zetas)
    PSIN = np.ones_like(ZETAS)*psi_N

    # Computing the parallel flow components
    # stored as the arrays multiplying dr/dtheta, dr/dtheta
    
    assert speciesIndex in [0, 1], "must give a valid speciesIndex -- 0 (electrons), 1 (ions)"
    if speciesIndex == 0:
        vPar = parseHDF5(folder, vPar_e).data
    else:
        vPar = parseHDF5(folder, vPar_i).data

    vPar_unit = vPar.unit
    vPar = valsafe(vPar)
    vPar = np.vstack((vPar, vPar))
    
    Gval = parseHDF5(folder, G).data
    G_unit = Gval.unit
    G_array = np.ones_like(ZETAS)*valsafe(Gval)

    Ival = parseHDF5(folder, I).data
    I_unit = Ival.unit
    I_array = np.ones_like(ZETAS)*valsafe(Ival)

    iotaval = parseHDF5(folder, iota).data

    modB = parseHDF5(folder, B).data
    modB_unit = modB.unit
    modB = valsafe(modB)
    modB = np.vstack((modB, modB))

    vPar_theta = vPar*modB*iotaval / (G_array + iotaval*I_array)
    vPar_theta_unit = vPar_unit*modB_unit / (G_unit)
    
    vPar_zeta = vPar*modB / (G_array + iotaval*I_array)
    vPar_zeta_unit = vPar_theta_unit

    # Now computing the perpendicular component of the flow velocity

    # First, the B cross nabla psi over B^2 components are computed
    
    diamag_theta = G_array / (G_array + iotaval*I_array)
    diamag_zeta = I_array / (G_array + iotaval*I_array)

    # next the scaling factor (w factor in Helander + Sigmar)
    
    dPhi_dpsi = parseHDF5(folder, dPhidpsi).data

    label_for_plot = ["Ion", " ", "flow velocity [km/s]"]
    
    if speciesIndex == 0:
        dTdpsi = dT_edpsi
        dndpsi = dn_edpsi
        n = n_e
        T = T_e
        Z = -1.0
        label_for_plot[0] = "Electron"
    else:
        dTdpsi = dT_idpsi
        dndpsi = dn_idpsi
        n = n_i
        T = T_i
        Z = 1.0

    dn_dpsi = parseHDF5(folder, dndpsi).data
    dT_dpsi = parseHDF5(folder, dTdpsi).data
    Tval = parseHDF5(folder, T).data

    n_array = parseHDF5(folder, n).data
    n_array = np.vstack((n_array, n_array))

    #print("n_array", n_array)
    #print("Tval", Tval)
    #print("dn_dpsi", dn_dpsi)
    #print("dT_dpsi", dT_dpsi)
    #print("dPhi_dpsi", dPhi_dpsi.to(u.V/(u.T*u.m*u.m)))
    
    w = dPhi_dpsi + (1/ (Z * e * n_array))*(n_array*dT_dpsi + Tval*dn_dpsi)

    #print("w", w)
    #print("first component", ((1/ (Z * e * n_array))*(n_array*dT_dpsi)).to(u.V/(u.T*u.m*u.m)))
    #print("second component", ((1/ (Z * e * n_array))*(Tval*dn_dpsi)).to(u.V/(u.T*u.m*u.m)))
    
    # combining the scale factor with the directional arrays

    vPerp_theta = valsafe(diamag_theta * w)
    vPerp_zeta = valsafe(diamag_zeta * w)
    
    # combining parallel and perpendicular parts
    scalePerp = 1.0
    if omitPerp is True:
        scalePerp = 0.0
        label_for_plot[1] = " parallel "  

    scalePar = 1.0
    if omitPar is True:
        scalePar = 0.0
        label_for_plot[1] = " perpendicular "

    #print("VPAR THETA", vPar_theta)
    #print("VPAR ZETA", vPar_zeta)
    #print("VPERP THETA", vPerp_theta)
    #print("VPERP ZETA", vPerp_zeta)

    v_theta = vPar_theta*vPar_theta_unit*scalePar + vPerp_theta*scalePerp*w.unit
    v_zeta = vPar_zeta*vPar_zeta_unit*scalePar + vPerp_zeta*scalePerp*w.unit
    
    print("BRI Checkpoint")
    print(getPathToWout())
    bri = getBoozerRadialInterpolant(getPathToWout())
    points = unrollMeshgrid(PSIN, THETAS, ZETAS)
    moddrdtheta, moddrdzeta = getGradientMagnitudes(bri, points)
    dotproduct = get_drdzeta_dot_drdtheta(bri, points)
    moddrdtheta = rollMeshgrid(len(zetas), len(thetas), moddrdtheta)
    moddrdzeta = rollMeshgrid(len(zetas), len(thetas), moddrdzeta)
    dotproduct = rollMeshgrid(len(zetas), len(thetas), dotproduct)

    modv = np.sqrt( v_theta*v_theta*moddrdtheta*moddrdtheta + v_zeta*v_zeta*moddrdzeta*moddrdzeta + 2*v_theta*v_zeta*dotproduct )
    vPar_dot_vPerp = vPar_theta*vPerp_theta*moddrdtheta*moddrdtheta
    vPar_dot_vPerp += vPar_zeta*vPerp_zeta*moddrdzeta*moddrdzeta
    vPar_dot_vPerp += (vPar_theta*vPerp_zeta + vPar_zeta*vPerp_theta)*dotproduct
    #print("DOT PRODUCT VPAR DOT VPERP", vPar_dot_vPerp)
    #print("average", np.mean(vPar_dot_vPerp))

    # making sure that diamagnetic dir dot b dir = 0

    bHat_dot_diamag = (iotaval/modB)*diamag_theta*moddrdtheta*moddrdtheta
    #print(np.mean(bHat_dot_diamag))
    bHat_dot_diamag += diamag_zeta*moddrdzeta*moddrdzeta
    #print(np.mean(bHat_dot_diamag))
    bHat_dot_diamag += ( (iotaval/modB)*diamag_zeta + diamag_theta)*dotproduct

    #print("DOT PRODUCT BHAT DOT DIAMAG", bHat_dot_diamag)
    #print("average", np.mean(bHat_dot_diamag))
    
    if not os.path.exists("flows"):
        os.system("mkdir flows")

    flows = (v_theta, v_zeta, modv, moddrdtheta, moddrdzeta, dotproduct)

    with open(filename, 'wb') as f:
        pickle.dump(flows, f)

    if plot is True:
        # gets magnitude
        ll = label_for_plot
        label = ll[0]+ll[1]+ll[2]
        bri = getBoozerRadialInterpolant(getPathToWout())
        points = unrollMeshgrid(PSIN, THETAS, ZETAS)
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot()
        tickpositions = [0, np.pi]
        ticklabels = ["0", "$\pi$"]
        ax.set_xlabel("$\zeta$", fontsize=22)
        ax.set_ylabel(r"$\theta$", fontsize=22)
        ax.set_xticks(tickpositions, ticklabels, fontsize=16)
        ax.set_yticks(tickpositions, ticklabels, fontsize=16)
        lw = 0.2+4*valsafe(modv).T/np.max(valsafe(modv))
        strm = ax.streamplot(ZETAS[:,0], THETAS[0], valsafe(v_zeta.T), valsafe(v_theta.T), color=valsafe(modv).T/1000, linewidth=lw, cmap=parulacmap)
        #quiv = ax.quiver(ZETAS, THETAS, valsafe(v_zeta.T/np.max(modv)), valsafe(v_theta.T/np.max(modv)), valsafe(modv).T/1000, cmap=parulacmap)
        cbar = fig.colorbar(strm.lines)
        #cbar = fig.colorbar(quiv)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(label, size=22)
        fig.tight_layout()
        if not os.path.exists("./plots"):
            os.system("mkdir plots")
        fig.savefig(f"./plots/{folder.replace('.', '_')}_fullV_{speciesIndex}_perp_{omitPerp}_par_{omitPar}.jpeg", dpi=360)
        makeCSXSurface(folder, colorparam=modv, plotname=label)

        # makes plot of 1 >> | \iota + Gw/vB | criterion

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot()
        tickpositions = [0, np.pi]
        ticklabels = ["0", "$\pi$"]
        ax.set_xlabel(r"$\zeta$", fontsize=22)
        ax.set_ylabel(r"$\theta$", fontsize=22)
        ax.set_xticks(tickpositions, ticklabels, fontsize=16)
        ax.set_yticks(tickpositions, ticklabels, fontsize=16)
        criterion = valsafe(iotaval) + valsafe((G_array*w)/(vPar*modB))
        criterion = abs(valsafe(criterion))
        cont = ax.contourf(ZETAS, THETAS, criterion, levels=50, cmap=parula, norm=color.LogNorm(vmin=criterion.min(), vmax=criterion.max()))
        cbar = fig.colorbar(cont)
        cbar.ax.set_ylabel(r"$|\iota + \frac{Gw}{v_{||}B}|$")
        fig.tight_layout()
        fig.savefig(f"./plots/{folder.replace('.', '_')}_criterion_{speciesIndex}_perp_{omitPerp}_par_{omitPar}.jpeg", dpi=360)

    return v_theta, v_zeta, modv

def getAngularMomentumDensity(folder, speciesIndex=0):
    
    v_theta, v_zeta, modv = getFullV(folder, speciesIndex=speciesIndex)

    # need to multiply by mass density

    thetas = parseHDF5(folder, theta).data
    zetas = parseHDF5(folder, zeta).data
    zetas = np.concatenate((zetas, zetas + np.pi))
    psi_N = parseHDF5(folder, psiN).data

    THETAS, ZETAS = np.meshgrid(thetas, zetas)
    PSIN = np.ones_like(ZETAS)*psi_N

    if speciesIndex == 0:
        n = n_e
        m = 9.10938e-31*u.kg
    else:
        n = n_i
        m = 4.6518341428e-26*u.kg

    filename = f"./flows/{folder.replace('.', '_')}_fullV_{speciesIndex}_perp_False_par_False.pkl"

    mass_dens = m*parseHDF5(folder, n).data
    mass_dens = np.vstack((valsafe(mass_dens), valsafe(mass_dens)))*mass_dens.unit

    if os.path.exists(filename):
        with open(filename, "rb") as f:
            _, __, ___, moddrdtheta, moddrdzeta, dotproduct = pickle.load(f)
        print("Loaded pickle AMD")
    else:
        points = np.ascontiguousarray(unrollMeshgrid(PSIN, THETAS, ZETAS), dtype=np.float64)
        bri = getBoozerRadialInterpolant(getPathToWout())
        _, moddrdzeta = getGradientMagnitudes(bri, points)
        dotproduct = get_drdzeta_dot_drdtheta(bri, points)
        dotproduct = rollMeshgrid(len(zetas), len(thetas), dotproduct)
        moddrdzeta = rollMeshgrid(len(zetas), len(thetas), moddrdzeta)

    AngularMomentumField = mass_dens*(v_theta*dotproduct*u.m*u.m + v_zeta*moddrdzeta*moddrdzeta*u.m*u.m)
    return AngularMomentumField

def getRadialCurrent(folder):
    # FSA < J dot \nabla \psi > = fsa
    eFlux = parseHDF5(folder, eFlux_vm_psi).data
    iFlux = parseHDF5(folder, iFlux_vm_psi).data
    vprime = parseHDF5(folder, VPrime).data
    fsaj = e*iFlux - e*eFlux
    radial_current = fsaj*vprime
    radial_current.to(u.A)
    return radial_current

def fluxSurfaceAverageOfArray(folder, ARRAY, justIntegral=False):
    thetas = parseHDF5(folder, theta).data
    zetas = parseHDF5(folder, zeta).data
    zetas = np.concatenate((zetas, zetas+np.pi))

    assert ARRAY.shape == (len(zetas), len(thetas))
    
    modB = valsafe(parseHDF5(folder, B).data)
    modB2 = np.vstack((modB, modB))**2

    Gval = valsafe(parseHDF5(folder, G).data)
    Ival = valsafe(parseHDF5(folder, I).data)
    iotaval = valsafe(parseHDF5(folder, iota).data)
    vprime = valsafe(parseHDF5(folder, VPrime).data)

    dtheta = thetas[1]-thetas[0]
    dzeta = zetas[1]-zetas[0]

    sqrtg = (Gval+iotaval*Ival)/modB2
    if justIntegral:
        sqrtg = 1.0
    integrand = sqrtg*ARRAY
    integral = dtheta*dzeta*np.sum(integrand)

    if justIntegral:
        vprime = 1.0

    return integral/vprime

def getVprofile(radialCoordinate=rN, speciesIndex=0, omitPar=False, omitPerp=False, plot=False):
    dirfiles = os.listdir()
    radialCoords = []
    vals = []
    for file in dirfiles:
        if not file.startswith("rN"):
            continue
        radialCoords.append(parseHDF5(file, rN).data)
        v_theta, v_zeta, modv = getFullV(file, speciesIndex=speciesIndex, omitPar=omitPar, omitPerp=omitPerp, plot=plot)
        v = fluxSurfaceAverageOfArray(file, modv)
        vals.append(v)

    radialCoords, vals = zip(*sorted(zip(radialCoords, vals), key=lambda pair: pair[0]))
    radialCoords = list(radialCoords)
    vals = list(vals)
    
    ylabel = ""
    species = ["Electron ", "Ion "][speciesIndex]
    ylabel = ylabel + species
    if omitPar:
        ylabel = ylabel + "perpendicular "
    if omitPerp:
        ylabel = ylabel + "parallel "
    filename = ylabel + "flow velocity"
    ylabel = filename + " [km/s]"

    if plot:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot()
        ax.plot(valsafe(radialCoords), valsafe(vals), color='#012169', alpha=0.7)
        ax.plot(valsafe(radialCoords), valsafe(vals), marker='o', linestyle='None', color='#012169')
        ax.set_xlabel(radialCoordinate.label, fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        fig.savefig(f"./plots/{filename}_vs_rN.jpeg", dpi=320)

    return radialCoords, vals

def getFSAAngularMomentumDensity(folder, speciesIndex=0):
    L_density = getAngularMomentumDensity(folder, speciesIndex=speciesIndex)
    return fluxSurfaceAverageOfArray(folder, L_density).to(u.kg/u.m/u.s)

def getFSAAngularMomentumDensityProfile(speciesIndex=0, plot=True):
    dirfiles = os.listdir()
    radialCoords = []
    vals = []
    for file in dirfiles:
        if not file.startswith("rN"):
            continue
        radialCoords.append(parseHDF5(file, rN).data)
        v = valsafe(getFSAAngularMomentumDensity(file, speciesIndex=speciesIndex))
        vals.append(v)

    radialCoords, vals = zip(*sorted(zip(radialCoords, vals), key=lambda pair: pair[0]))
    radialCoords = list(radialCoords)
    vals = list(vals)

    ylabel = ""
    species = ["Electron ", "Ion "][speciesIndex]
    ylabel = ylabel + species
    filename = ylabel + "angular momentum density"
    ylabel = filename + r" [$\frac{\text{kg}}{\text{m}\cdot\text{s}}]"

    if plot:
        if not os.path.exists("./plots"):
            os.system("mkdir plots")

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot()
        ax.plot(valsafe(radialCoords), valsafe(vals), color='#012169', alpha=0.7)
        ax.plot(valsafe(radialCoords), valsafe(vals), marker='o', linestyle='None', color='#012169')
        ax.set_xlabel(radialCoordinate.label, fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        fig.savefig(f"./plots/{filename}_vs_{radialCoordinate.name}.jpeg", dpi=320)
    return radialCoords, vals

def getNTVTorque(folder, speciesIndex=0):
    eFlux = parseHDF5(folder, eFlux_vm_psi).data
    iFlux = parseHDF5(folder, iFlux_vm_psi).data
    fsaj = e*iFlux - e*eFlux
    iotaVal = parseHDF5(folder, iota).data
    NTV = iotaVal*fsaj
    return NTV.to(u.kg/u.m/u.s/u.s)

def getDeltaT(folder, speciesIndex=0):
    try:
        return ( getFSAAngularMomentumDensity(folder, speciesIndex=speciesIndex)/getNTVTorque(folder, speciesIndex=speciesIndex) ).to(u.us)
    except AssertionError:
        print(f"Skipping Delta T calculation for {folder}")

def getDeltaTProfile(speciesIndex=0, plot=True):
    dirfiles = os.listdir()
    radialCoords = []
    vals = []
    for file in dirfiles:
        if not file.startswith("rN"):
            continue
        radialCoords.append(parseHDF5(file, rN).data)
        v = valsafe(getDeltaT(file, speciesIndex=speciesIndex))
        vals.append(v)

    radialCoords, vals = zip(*sorted(zip(radialCoords, vals), key=lambda pair: pair[0]))
    radialCoords = list(radialCoords)
    vals = list(vals)

    ylabel = ""
    species = ["Electron ", "Ion "][speciesIndex]
    ylabel = ylabel + species
    filename = ylabel + r"Delta t"
    ylabel =  r"$\Delta t$ [s]"

    if plot:
        if not os.path.exists("./plots"):
            os.system("mkdir plots")

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot()
        ax.plot(valsafe(radialCoords), valsafe(vals), color='#012169', alpha=0.7)
        ax.plot(valsafe(radialCoords), valsafe(vals), marker='o', linestyle='None', color='#012169')
        ax.set_xlabel(r"$\sqrt{\psi_N}$", fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        fig.savefig(f"./plots/{filename}_vs_rN.jpeg", dpi=320)
    return radialCoords, vals

def getTotalHeatFlux(folder, speciesIndex=0):
    # here "total" doesn't mean classical + neoclassical + turbulent
    # it means just the neoclassical heat flux but not flux surface averaged
    
    if speciesIndex==0:
        totalHeatFlux = totalHeatFlux_e
    else:
        totalHeatFlux = totalHeatFlux_i

    thetas = parseHDF5(folder, theta).data
    zetas = parseHDF5(folder, zeta).data
    zetas = np.concatenate((zetas, zetas + np.pi))

    heatFlux = valsafe(parseHDF5(folder, totalHeatFlux).data)
    heatFlux = np.vstack((heatFlux, heatFlux))

    print("flux surface averaged", fluxSurfaceAverageOfArray(folder, heatFlux, justIntegral=True))
    print("from sficns", valsafe(parseHDF5(folder, heatFlux_vm_psi_e).data))

def getNTVvsEr(folder, returnAMD=False, speciesIndex=0):
    # to be run in raderscan directory
    # folder is some radius
    main_dir = os.getcwd()
    os.chdir(folder)
    print(returnAMD)
    Ers = []
    taus = []
    amds = []
    Erfiles = [file for file in os.listdir() if file.startswith("Er")]
    for file in Erfiles:
        try:
            er = valsafe(parseHDF5(f"{file}", Er).data)
            tau = valsafe(getNTVTorque(f"{file}"))
            Ers.append(er)
            taus.append(tau)
            if returnAMD:
                amd = valsafe(getFSAAngularMomentumDensity(file, speciesIndex=speciesIndex))
                amds.append(amd)
        except:
            continue
    print(Ers)
    print(taus)
    print(amds)
    print(os.getcwd())
    if returnAMD:
        ___, amds = zip(*sorted(zip(Ers, amds), key=lambda pair: pair[0]))
        amds = list(amds)

    Ers, taus = zip(*sorted(zip(Ers, taus), key=lambda pair: pair[0]))
    Ers = list(Ers)
    taus = list(taus)

    if returnAMD:
        return Ers, taus, amds

    return Ers, taus


if __name__ == "__main__":

    # ensures a plots folder in outputsDir
    if not os.path.exists("./plots"):
        os.system("mkdir plots")

    # makeStreamPlot("rN_0.95", vPar_e)
    # makeStreamPlot("rN_0.95", vPar_i)

    make_qlcfs_file()
    for radius in [file for file in os.listdir() if file.startswith("rN_0.75")]:
        print(f"Analyzing file {radius}...")
        getFullV(radius)
        #getRadialCurrent(radius)
        #getFullV(radius, omitPerp=True, plot=True)
        #getFullV(radius, omitPar=True, plot=True)
        #getFullV(radius, plot=True, speciesIndex=0)
        #getFullV(radius, plot=True, speciesIndex=1)
        #getFullV(radius, omitPerp=True, plot=True, speciesIndex=1)
        #getFullV(radius, omitPar=True, plot=True, speciesIndex=1)
        #getAngularMomentumDensity(radius)

"""
folder = "rN_0.95"
plotProfile(FSAbootstrapCurrentDensity, rN)
plot
Profile(Er, rN)
plotProfile(FSAbootstrapCurrentDensity, Er)
plotProfile(rN, psiN)
plotHeatmap(folder, vPar_i)
plotHeatmap(folder, vPar_e)
makeHeatmapGif(vPar_i, rN, contourLevels=np.linspace(-125, 155, 225))
make_qlcfs_file()

makeCSXSurface("rN_0.95", colorparam=B)
makeCSXSurface("rN_0.95", colorparam=vPar_i)
makeCSXSurface("rN_0.95", colorparam=vPar_e)

# for making confirmation plots
folders = ["esym_0.1", "esym_0.6", "esym_0.9", "rN_0.95"]
epsbs = [0.1, 0.6, 0.9, 1.0]
QSErrors = []
for file, epsb in zip(folders, epsbs):
    thetas = parseHDF5(file, theta).data
    zetas1 = parseHDF5(file, zeta).data
    zetas = np.concatenate((zetas1, zetas1 + np.pi))
    modB = valsafe(parseHDF5(file, B).data)
    modB = np.vstack((modB, modB)).T
    QSErrors.append(getQSError(modB))

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(epsbs, QSErrors, marker='o', color="black")
ax.set_xlabel(r"$\epsilon_{sb}$", fontsize=18)
ax.set_ylabel(r"QS Error $S = \sqrt{\sum_{m, n\neq 0}\frac{B_{mn}^2}{B_{00}^2}}$", fontsize=18)
plt.tight_layout()
fig.savefig("./plots/epvserr.jpeg")
print(QSErrors)

t1, t2 = 0, 2*np.pi
fig = plt.figure(figsize=(9, 13))
plt.tight_layout()
ax = fig.add_subplot(411)
file = "rN_0.95"
thetas = parseHDF5(file, theta).data
zetas = parseHDF5(file, zeta).data
zetas = np.concatenate((zetas, zetas+np.pi))
modB = valsafe(parseHDF5(file, B).data)
modB = np.vstack((modB, modB))
modB = modB.T
t2z = zetas[-1]
t2t = thetas[-1]
plt.contourf(zetas, thetas, modB, levels=13)
plt.colorbar()
ax.set_title(r"$\epsilon_{sb} = 1$")
ax.set_xticks([t1, t2z])
ax.set_xticklabels(["0", r"$2\pi$"])
ax.set_yticks([t1, t2t])
ax.set_yticklabels(["0", r"$2\pi$"])

ax = fig.add_subplot(412)
modB = scale_by_epsb(modB, 0.1)
modB_michael=modB
plt.contourf(zetas, thetas, modB, levels=13)
plt.colorbar()
ax.set_title(r"$\epsilon_{sb} = 0.1$ (michael)")
ax.set_xticks([t1, t2z])
ax.set_xticklabels(["0", r"$2\pi$"])
ax.set_yticks([t1, t2t])
ax.set_yticklabels(["0", r"$2\pi$"])

ax = fig.add_subplot(413)
file = "esym_0.1"
thetas = parseHDF5(file, theta).data
zetas = parseHDF5(file, zeta).data
zetas = np.concatenate((zetas, zetas + np.pi))
modB = valsafe(parseHDF5(file, B).data)
modB = np.vstack((modB, modB))

modB = modB.T

plt.contourf(zetas, thetas, modB, levels=13)
plt.colorbar()
ax.set_title(r"$\epsilon_{sb} = 0.1$ (mike)")
ax.set_xticks([t1, t2z])
ax.set_xticklabels(["0", r"$2\pi$"])
ax.set_yticks([t1, t2t])
ax.set_yticklabels(["0", r"$2\pi$"])

ax = fig.add_subplot(414)
plt.contourf(zetas, thetas, modB - modB_michael, levels = 13)
plt.colorbar()
ax.set_title("mike - michael")

ax.set_xticks([t1, t2z])
ax.set_xticklabels(["0", r"$2\pi$"])
ax.set_yticks([t1, t2t])
ax.set_yticklabels(["0", r"$2\pi$"])

fig.supxlabel(r"$\zeta$", y = 0.04, fontsize = 18)
fig.supylabel(r"$\theta$", x = 0.05, fontsize = 18)
fig.suptitle(r"$|B|$ [T]", y = 0.95, fontsize = 18)
fig.savefig("./plots/comparison_0.1.jpeg")
#end confirmation plots
"""




