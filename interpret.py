"""
Interprets the results of the scan of radius and radial electric field
make with runRadErScan.py. Providing a rootChoice and Ir (run this script
with --help for more info) this script saves a numpy array 'scan_info.npy'
detailing the radial electric field profile necessary to enforce the given
Ir, which will be used by adjustScan.py. This script is auxiliary to 
runAmbipolar.py, and probably won't need to be run standalone.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import root
from tqdm import tqdm
import argparse


# ------ Parses input tags ------ #
parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--rootChoice', type=str, required=True, help='low middle or high, which root to use (if only two, use low or middle for electron root)')
parser.add_argument('--Ir', type=float, default=0.0, help='radial current to enforce, in Amperes')

rootChoice = parser.parse_args().rootChoice
Ir = parser.parse_args().Ir
# ------------------------------- #

print("~~~ Interpretting results from runRadErScan ~~~")
files = os.listdir('./')

# ----------------------------------------------------------------------- #
# goes into the raderscan directory, and for each radial value rN
# (rN = sqrt(psi/psiLCFS) where psi is the toroidal flux) makes an
# array "data" containing a 2D array of data from a scan of the radial
# current Ir over Er on the rN flux surface. "datas" is an array of these
# arrays of length len(rN).
rNs = []
datas = []
for file in files:
    if file.endswith('Ir-vs-Er.dat'):
        datas.append(np.transpose(np.loadtxt(file)))
        rNs.append(float(file.split('_')[1].split('-')[0]))

rNs, datas = zip(*sorted(zip(rNs, datas), key=lambda pair: pair[0]))
rNs = list(rNs)
datas = list(datas)
print("Data for the following rN values has been succesfully retrieved:")
print(rNs)
# ----------------------------------------------------------------------- #

def plot_data(data, ErRange=None):
    """
    data is a two-array [Er, Ir] of a scan of Ir against Er.
    plots Ir vs Er
    """
    Er_data = data[0]
    Jr_data = data[1]

    Jr = interp1d(Er_data, Jr_data, kind='linear')
    
    if ErRange is None:
        ErRange = np.max(abs(Er_data))

    Er = np.linspace(-ErRange, ErRange, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if np.max(Jr_data)/1e-3 > 100:
        ax.set_yscale('symlog', linthresh=1e-3)
    plt.plot(Er_data, Jr_data, linestyle=None, marker='o')
    plt.plot(Er, Jr(Er))
    fig.show()

def deriv(y, x0):
    """
    for a function y(x), returns an approximation
    of y'(x0)
    """
    return (y(x0+x0*1e-6)-y(x0-x0*1e-6))/(x0*2e-6)
    
def crossesZero(array):
    """
    boolean: True only if the first and final values
    of the array differ in sign
    """
    crosses = False
    if np.sign(array[0]/array[-1]) == -1:
        crosses = True
    return crosses
    
def zeroCrossings(x, y):
    """
    finds the values of x which approximate the roots
    of y(x). y and x are both arrays of the same length
    """
    s = np.sign(y)
    crossings = np.where(s[:-1] * s[1:] < -0.5)[0]
    return x[crossings]
    

def get_Er(data, rootChoice, scaledownFactor=0.99):
    """
    given data, a two element array of arrays [Er, Ir] containing a scan of radial current
    over Er, returns the value of Er which corresponds to the Ir provided to the script.
    If no value of Ir was provided, the ambipolar Er is calculated.

    scaledownFactor is the number the maximum and minimum Er considered for rootfinding
    will be multiplied by-- this is here so that 1. on the first recursion level, there are no 
    issues with extrapolation 2. on subsequent recursion levels, the considered range is 
    adjusted so that eventually we will end up with a maximum of 3 roots. This is useful
    if one over-scans the Er parameter in sfincs-- past a certain magnitude of Er, it will start 
    giving a lot of 0 answers for the particle fluxes (which we should ignore. This indicates
    a sfincs failure.) and thus the radial current.

    It is important to scan the Er at a high enough resolution so that we can be confident
    the values obtained from the linear interpolation here will be good.
    """
    assert rootChoice in ['low', 'middle', 'high']
    # subtracts away the value of Ir we want to find, and then looks
    # for roots. If Ir is 0 (or not provided), then these roots will
    # be the ambipolar ones.
    data = [data[0], np.array(data[1])-Ir]
    ErMin = np.min(data[0])
    ErMax = np.max(data[0])
    ErDiff = abs(data[0][1] - data[0][0]) # implicitly assumes len(Er) > 1, which is the
    print("ErMin, ErMax", ErMin, ErMax)   # only case this script is useful anyway
    # makes a linear interpolant. Ignore the use of "Jr", this value is Ir in Amperes
    Jr = interp1d(data[0], data[1], kind='linear')
    Er_range = np.linspace(ErMin*scaledownFactor, ErMax*scaledownFactor, 1000)
    Jrs = Jr(Er_range)
    # gets initial guesses for the zeroes based off where along Er that Ir changes sign
    zero_crossings = zeroCrossings(data[0], data[1])
    Er_candidates = [Er for Er in zero_crossings if abs(Er) <= np.max([abs(ErMin), abs(ErMax)])]
    
    print("Initial ambipolar Er candidates:", Er_candidates)
    aErs = []
    aEr = None

    try:
        if len(Er_candidates) > 3:
            aEr = get_Er(data, rootChoice=rootChoice, scaledownFactor=scaledownFactor*0.9)
    except RecursionError:
        print("Ers", data[0])
        print("Irs", data[1])
        raise ValueError("Maximum recursion depth exceeded.\nSee above what is going wrong")
    
    # zooms in, looks around the guesses for the root using
    # the interpolant. Returns the root asked for using rootChoice,
    # if more than 1 is found. If two roots, "low" and "middle" both
    # correspond to the lower root, and no more than 3 roots will be
    # handled (guaranteed by the above recursion)
    if aEr is None:
        for er in Er_candidates:
            er_range = np.linspace(max(er-ErDiff, ErMin), min(er+ErDiff, ErMax), 500000)
            jrs = Jr(er_range)
            if crossesZero(jrs):
                aErs.append(er_range[np.argmin(abs(jrs))])
            else:
                er_range = np.linspace(max(er-ErDiff/5, ErMin), min(er+ErDiff/5, ErMax), 500000)
                jrs = Jr(er_range)
                if crossesZero(jrs):
                    aErs.append(er_range[np.argmin(abs(jrs))])
                else:
                    raise ValueError("can't find exact zero crossing")
            
        if len(aErs) == 1:
            if rootChoice == 'middle':
                return aErs[0]
            print("Only a single root to choose from.")
            return aErs[0]
        
        if len(aErs) == 2:
            if rootChoice == 'low' or rootChoice == 'middle': #ion root
                return aErs[0]
            return aErs[1] #electron root, rootChoice == 'high'

        if rootChoice == 'low':
            aEr = aErs[0]
        if rootChoice == 'middle':
            aEr = aErs[1]
        if rootChoice == 'high':
            aEr = aErs[2]
            
    return aEr

def make_profile(rNs, datas, rootChoice):
    """
    generates Er profile corresponding to the Ir provided when
    interpret.py is called. If Ir is not provided, the default value is 0
    and the ambipolar Er profile is generated.

    rNs is an array of rN values (rN = sqrt{psi/psiLCFS}), datas
    is an array of arrays like [Er, Ir] containing scans of the 
    radial current over Er. rootChoice is "low" "middle" or "high",
    an option to select which ambipolar root to choose. For CSX,
    "middle" is the best choice.
    """
    Ers = []
    for i, data in enumerate(tqdm(datas, desc="Generating Er profile...")):
        Ers.append(get_Er(data, rootChoice))
        
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.plot(rNs, Ers, marker='o', linestyle='None', color='#012169')
    ax.set_ylabel(r"ambipolar $E_r$ (V/m)", fontsize=24)
    ax.set_xlabel(r"$\sqrt{\psi_N}$", fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    fig.show()
    return rNs, Ers
    
def make_3d_data(rNs, datas):
    """
    gets the Ir vs Er and rN data in the correct shape
    to plot -- this function and the following two plotting 
    functions aren't called anywhere else.
    """
    Er = np.linspace(-99.9, 99.9, 30) # todo hardcoded range
    Jr = []
    for i in range(0, len(datas)):
        Jr.append(interp1d(datas[i][0], datas[i][1], kind='linear')(Er))
    
    ER, RN = np.meshgrid(Er, rNs)
    JR = np.array(Jr)
    return ER, RN, JR
    
def contour(rNs, datas):
    """
    makes a contour plot of the radial current 
    as a function of Er and rN
    """
    ER, RN, JR = make_3d_data(rNs, datas)
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.contourf(ER, RN, JR)
    c = plt.colorbar()
    fig.show()
    
def surface(rNs, datas):
    """
    makes a surface plot of the radial current
    as a function of Er and rN
    """
    ER, RN, JR = make_3d_data(rNs, datas)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(ER, RN, abs(JR))
    ax.set_zscale('log')
    fig.show()
    
# finds Er as a function of rN
rN, Er = make_profile(rNs, datas, rootChoice)

# saves profile to a file which will later be read
# by adjustScan.py 
np.save("scan_info.npy", np.array([rN, np.array(Er)/1000])) # to kV/m

# prints out the profile so that you can double
# check that it's reasonable
if Ir == 0.0:
    print("Ambipolar results")
else:
    print(f"Radial current Ir = {Ir} results")

for i in range(0, len(rN)):
    print("rN:", rN[i], "Er:", Er[i])

print("~~~ End interpret.py ~~~")

## uncomment to make matlab file
## I prefer matlab plots sometimes
#from scipy.io import savemat

#ER, RN, JR = make_3d_data(rNs, datas)
#savemat('raderscan6.mat', {
#    'ER': ER,
#    'RN': RN,
#    'JR': JR
#})
